# Compute background
import enum
from turtle import position
from typing import OrderedDict
from astropy.stats import sigma_clipped_stats
import numpy as np
import sep
import os
from astropy.io import fits

import sys

PIXEL_SCALE=-99

def compute_background(raw_img, mask, BACKTYPE, BACKPARAMS, DIR_OUTPUT=None, NICKNAME=None, HEADER=None):
    if BACKTYPE == 'VAR':
        bkg = sep.Background(raw_img.byteswap().newbyteorder(), mask=mask, **BACKPARAMS)
        back = bkg.back()
        print(f'    Removing background with variable background')
    if BACKTYPE == 'GLOBAL':
        bkg = sep.Background(raw_img.byteswap().newbyteorder(), mask=mask, **BACKPARAMS)
        back = bkg.globalback
        print(f'    Removing background with global background')
    elif BACKTYPE == 'MED':
        mean, med, std = sigma_clipped_stats(raw_img[~mask], sigma=3)
        back = med
        print(f'    Removing background with sigma-clipped median: {med:2.5f}')
    elif BACKTYPE == 'NONE':
        back = np.zeros_like(raw_img)

    if DIR_OUTPUT is not None:
        if BACKTYPE == 'VAR':
            HEADER['BACKTYPE'] = BACKTYPE
            for key in BACKPARAMS:
                HEADER[f'BACK_{key}'] = BACKPARAMS[key]
            hdul = fits.HDUList()
            hdul.append(fits.ImageHDU(name='BACKGROUND', data=back.astype(np.float32), header=HEADER))
            hdul.writeto(os.path.join(DIR_OUTPUT, f'{NICKNAME}_DETBACK.fits'), overwrite=True)

        if BACKTYPE != 'NONE':
            hdu = fits.ImageHDU(name='SCIENCE', data=raw_img, header=HEADER)
            hdu.writeto(os.path.join(DIR_OUTPUT, f'{NICKNAME}_DETECTION_BKGSUB.fits'), overwrite=True)


    return back


# Compute empty aperture curve
def fit_apercurve(stats, plotname=None, pixelscale=PIXEL_SCALE, stat_type=['ksnmad',], init=(1, 5)):
    py = []
    py1 = []
    psizes = []
    for size in stats:
        if size in ('Naper', 'positions', 'medwht'):
            continue
        if size == -1:
            py1.append([stats[size][stype] for stype in stat_type])
        else:
            py.append([stats[size][stype] for stype in stat_type])
            psizes.append(size)

    psizes = np.array(psizes)
    py = np.array(py)
    py1 = np.array(py1)
    N = np.sqrt(np.pi*(psizes/pixelscale/2.)**2)
    px = np.arange(psizes[0], psizes[-1], 0.001)
    pN = np.sqrt(np.pi*(px/pixelscale/2.)**2)

    from scipy.optimize import curve_fit

    p, pcov, s = {}, {}, {}
    for i, st in enumerate(stat_type):
        def func(N, a, b):
            s = py[0, i]
            return s * a * N**b
        s[st] = py1[0, i]
        p[st], pcov[st] = curve_fit(func, N, py[:,i], p0=init)

    if plotname is not None:
        import matplotlib.pyplot as plt
        plt.ioff()
        py = np.array(py)
        fig, ax = plt.subplots(figsize=(5, 5))
        for i, st in enumerate(stat_type):
            def func(N, a, b):
                s = py[0, i]
                return s * a * N**b
            ax.scatter(psizes, py[:,i])

            label=f'{st}: {py1[0,i]:2.2f}$x${p[st][0]:2.2f}$N^{{{p[st][1]:2.2f}}}$'

            plt.plot(px, func(pN, *p[st]), label=label)
    #     print(p)

        # ax.plot(px, func(pN, 0.05, 2.1))
            upper = func(pN, p[st][0], 2)
            ax.plot(px[upper<py[-1,i]], upper[upper<py[-1,i]], ls='dashed', c='grey')
            ax.plot(px, func(pN, p[st][0], 1), ls='dashed', c='grey')

        ax.legend(fontsize=15)
        ax.set(xlim=(0, 1.1*px[-1]))
        ax.set(xlabel='Aperture Diam (\")', ylabel='$\sigma_{\\rm NMAD}$')
        fig.tight_layout()
        fig.savefig(plotname)

    return p, pcov, s

def hist_points(x, bins=None, weights=None):
    if bins is not None:
        counts, bin_edges = np.histogram(x, bins, weights=weights)
    else:
        counts, bin_edges = np.histogram(x, weights=weights)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2.
    bin_widths = np.diff(bin_edges)
    if weights is not None:
        idx = np.digitize(x, bin_edges[:-1])
        count_unc = np.array([np.sqrt(np.sum(weights[idx==i]**2)) for i in 1+np.arange(len(bin_edges[:-1]))])
    else:
        count_unc = np.sqrt(counts)

    assert(len(counts)==len(count_unc))
    return bin_centers, bin_widths, counts, count_unc

# Empty Apertures
def empty_apertures(img, wht, segmap, N=1E6, aper=[0.35, 0.7, 2.0], pixscl=PIXEL_SCALE, plotname=None, zpt_factor=1.):
    from alive_progress import alive_bar
    import numpy as np
    import scipy.stats as stats
    import astropy.units as u
    from astropy.stats import mad_std
    from astropy.stats import sigma_clipped_stats, sigma_clip
    from photutils.aperture import CircularAperture, aperture_photometry
    from astropy.modeling import models, fitting

    from config import TARGET_ZP, SCI_APER


    aper = np.array(aper)
    aperrad = aper / 2. / pixscl # diam sky to rad pix
    maxaper = int(aperrad[aper==SCI_APER][0]) + 1

    size = np.shape(img)
    try:
        segsize = np.shape(segmap)
        assert(size == segsize, f'Image size ({size}) != seg size ({segsize})!')
    except:
        return {}

    kept = 0
    positions = np.zeros((N, 2))
    medwht = np.zeros(N)
    checkimg = (segmap == 0) & (~np.isnan(img)) & (wht>0)
    print(f'{np.nansum(checkimg)/np.nansum(checkimg!=-1)*100:2.1f}% of image available for sky measurements...')
    with alive_bar(N) as bar:
        while kept < N:
            px, py = np.random.uniform(0, 1, 2)
            x, y = int(px * size[0]), int(py * size[1])
            xlo, xhi, ylo, yhi = x-maxaper,x+maxaper, y-maxaper,y+maxaper
            if (xlo < 0) | (xhi > size[0]) | (ylo < 0) | (yhi > size[1]):
                continue
            # print(x, y)
            if np.all(checkimg[xlo:xhi, ylo:yhi]):
                # print(x, y)
                # print(xlo, xhi, ylo, yhi) 
                # print(img[xlo:xhi, ylo:yhi])
                # print()
                positions[kept] = y, x
                intx, inty = int(x), int(y)
                boxwht = wht[intx-4:intx+5, inty-4:inty+5]
                medwht[kept] = np.nanmedian(boxwht[boxwht>0])
                kept += 1
                # print(kept, (y, x), np.sum(checkimg[xlo:xhi, ylo:yhi]))
                bar()

    # now do it in parallel
    print(f'Computing photometry in {len(aperrad)} empty apertures...')
    apertures = [CircularAperture(positions, r=r) for r in aperrad]
    output = aperture_photometry(img, apertures)
    print(f'Done!')


    aperstats = OrderedDict()
    aperstats[-1] = {}
    clean_img = img[(wht>0) & (segmap==0) & (~np.isnan(img))].flatten() * zpt_factor

    print('Sigma clipping...')
    klip_clean_img = sigma_clip(clean_img)
    print('Done. Saving statistics...')
    kmean, kmed, kstd = np.mean(klip_clean_img), np.median(klip_clean_img), np.std(klip_clean_img)
    ksnmad = mad_std(klip_clean_img)
    aperstats[-1]['ksnmad'] = ksnmad
    aperstats[-1]['kmean'] = kmean
    aperstats[-1]['kmed'] = kmed
    aperstats[-1]['kstd'] = kstd
    aperstats[-1]['sigma'] = np.nanstd(clean_img)
    aperstats[-1]['snmad'] = mad_std(clean_img)

    aperstats['Naper'] = N
    aperstats['positions'] = positions
    aperstats['medwht'] = medwht

    # Define the Gaussian function
    from scipy.stats import norm

    print('Fitting Gaussians...')
    pc = np.nanpercentile(klip_clean_img, q=(5, 95))
    bins = np.linspace(pc[0], pc[1], 20)
    px = np.linspace(pc[0], pc[1], 1000)
    bin_centers, bin_widths, counts, count_unc = hist_points(klip_clean_img, bins=bins)
    g_init = models.Gaussian1D(amplitude=np.max(counts), mean=kmean, stddev=kstd)
    fit_g = fitting.LevMarLSQFitter()
    gmodel = fit_g(g_init, bin_centers, counts)

    aperstats[-1]['fit_mean'] = gmodel.mean.value
    aperstats[-1]['fit_std'] = gmodel.stddev.value

    if plotname is not None:
        import matplotlib.pyplot as plt
        plt.ioff()
        ncols = 5
        nrows = int((len(aper)+1)/ncols) + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
        # ax.set(xlab)
        axes[0].errorbar(bin_centers, counts, xerr=bin_widths/2., yerr=count_unc, fmt='o', c='grey')
        # # axes[0].hist(clean_img_neg, bins=bins, color='grey', histtype='step', density=True, label=f'1px kstd_neg:{aperstats[-1]["kstd_neg"]:2.2f}')
        # print(gmodel.stddev.value, type(gmodel.stddev.value))
        axes[0].plot(px, gmodel(px), c='k', label=f'1px std: {gmodel.stddev.value:2.2f}  ksnmad: {ksnmad:2.2f}')

        axes[0].legend(loc='upper left')
        # ax.axvline(-aperstats['sigma'], color='k', ls='dashed')
        # ax.axvline(aperstats['sigma'], color='k', ls='dashed')
        colors = plt.get_cmap('rainbow', len(aper))

        fig2, axes2 = plt.subplots(ncols=2, figsize=(10,5))
        axes2[0].set(xlabel='aperture diameter (arcsec)', ylabel='Depth (AB; $\sigma_{\\rm NMAD}$)')
        axes2[1].set(xlabel='aperture diameter (arcsec)', ylabel='Sky (Flux; median)')

        fig3, ax3 = plt.subplots()
        ax3.scatter(positions.T[0], positions.T[1], s=2, alpha=0.3, c='grey')


    # measure moments + percentiles; AD test
    for i, diam in enumerate(aper):
        phot = output[f'aperture_sum_{i}'] * zpt_factor
        aperstats[diam] = {}

        aperstats[diam]['phot'] = phot
        ax = axes[i+1]

        phot = phot[~(np.isnan(phot) | (phot==0))]
        print(f'Running statistics on {len(phot)} good measurements')

        snmad = mad_std(phot)
        med = np.nanmedian(phot)

        print('Sigma clipping...')
        klip_phot = sigma_clip(phot)

        pc = np.nanpercentile(klip_phot, q=(5, 95))
        bins = np.linspace(pc[0], pc[1], 20)
        px = np.linspace(pc[0], pc[1], 1000)

        print('Fitting Gaussians...')
        from scipy.optimize import curve_fit
        kmean, kmed, kstd = sigma_clipped_stats(phot, sigma=3.)
        ksnmad = mad_std(klip_phot)
        bin_centers, bin_widths, counts, count_unc = hist_points(klip_phot, bins=bins)
        g_init = models.Gaussian1D(amplitude=np.max(counts), mean=kmean, stddev=kstd)
        fit_g = fitting.LevMarLSQFitter()
        gmodel = fit_g(g_init, bin_centers, counts)
        aperstats[diam]['fit_mean'] = gmodel.mean.value
        aperstats[diam]['fit_std'] = gmodel.stddev.value

        aperstats[diam]['mean'] = np.mean(phot)
        aperstats[diam]['std'] = np.std(phot)
        aperstats[diam]['snmad'] = snmad
        aperstats[diam]['norm'] = stats.normaltest(phot)
        aperstats[diam]['median'] = med
        
        aperstats[diam]['ksnmad'] = ksnmad
        aperstats[diam]['kmean'] = kmean
        aperstats[diam]['kmed'] = kmed
        aperstats[diam]['kstd'] = kstd
        pc = np.percentile(phot, q=(5, 16, 84, 95))
        aperstats[diam]['pc'] = pc
        aperstats[diam]['interquart_68'] = pc[2] - pc[1] # 68pc

        if plotname is not None:
            ax.errorbar(bin_centers, counts, xerr=bin_widths/2., yerr=count_unc, fmt='o', c='grey')
            # # ax.hist(clean_img_neg, bins=bins, color='grey', histtype='step', density=True, label=f'1px kstd_neg:{aperstats[-1]["kstd_neg"]:2.2f}')
            ax.plot(px, gmodel(px), c=colors(i), label=f'{diam:2.2f}\" std: {gmodel.stddev.value:2.2f} ksnmad: {ksnmad:2.2f}')
            # ax.hist(sigma_clip(phot), bins=bins, color='k' , histtype='step', density=True, )

            # ax.plot(px, norm.pdf(px, *p), c=colors(i), )
            ax.legend(loc='upper left')

            # ax.axvline(-p[1], color=colors(i), ls='dashed')
            # ax.axvline(p[1], color=colors(i), ls='dashed')


        # for key in aperstats[radius]:
        #     print(key, aperstats[radius][key])
        # print()

    if plotname is not None:

        fig.tight_layout()
        fig.savefig(plotname)

        aper_fitstd = [aperstats[diam]['fit_std'] for diam in aper]
        aper_mean = [aperstats[diam]['fit_mean'] for diam in aper]

        # aper_fitstd_neg = [aperstats[diam]['snmad_neg'] for diam in aper]
        # aper_mean_neg = [aperstats[diam]['median_neg'] for diam in aper]
        axes2[0].plot(aper, TARGET_ZP - 2.5*np.log10(aper_fitstd), marker='o', c='k')
        axes2[1].plot(aper, aper_mean, marker='o', c='k')
        # axes2[0].plot(aper, TARGET_ZP - 2.5*np.log10(aper_fitstd_neg), marker='o')
        # axes2[1].plot(aper, aper_mean_neg, marker='o')

        fig2.tight_layout()
        fig2.savefig(plotname.replace('emptyaper', 'depth'))
        fig3.savefig(plotname.replace('emptyaper', 'aperpos'))


    return aperstats
# Star finder


# Make auto mask

def new_workingspace(version, path='.'):
    trypath = os.path.join(path, version)
    if os.path.exists(trypath):
        raise RuntimeError(f'Path already exists! {trypath}')
    os.mkdir(trypath)
    for dirname in ('external', 'intermediate', 'output', 'catalogs'):
        os.mkdir(os.path.join(path, f'{version}/{dirname}'))


# Compute COG for PSF
def psf_cog(psfmodel, nearrad=None):
    x = np.arange(-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2)
    y = x.copy()
    px = np.arange(0, np.shape(psfmodel)[0]/2, 0.2)
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + yv**2)
    cumcurve = np.array([np.sum(psfmodel[radius<i]) for i in px])
    import scipy.interpolate
    modcumcurve = scipy.interpolate.interp1d(px, cumcurve, fill_value = 'extrapolate')

    if nearrad is None:
        return px, cumcurve, modcumcurve

    return modcumcurve(nearrad)

def get_date():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(now)
    return now

# Make and rotate PSF
def get_psf(filt, field='uncover', angle=None, fov=4, og_fov=10, pixscl=PIXEL_SCALE, date=None, output=''):
    # makes the PSF at og_fov and clips down to fov. Works with 0.04 "/px
    import webbpsf
    from astropy.io import fits
    import numpy as np
    from scipy import ndimage
    import os
    from astropy.io import ascii

    from config import SW_FILTERS, LW_FILTERS, PATH_SW_ENERGY, PATH_LW_ENERGY

    # Check if filter is valid and get correction term
    if filt in SW_FILTERS:
        if fov != 4:
            print('WARNING! I will not fetch the correct encircled energy for your requested FOV!')
        # 17 corresponds with 2" radius (i.e. 4" FOV)
        encircled = ascii.read(PATH_SW_ENERGY)[17][filt]
    elif filt in LW_FILTERS:
        if fov != 4:
            print('WARNING! I will not fetch the correct encircled energy for your requested FOV!')
        encircled = ascii.read(PATH_LW_ENERGY)[17][filt]
    else:
        print(f'{filt} is NOT a valid NIRCam filter!')
        return

    # Observed PA_V3 for fields
    angles = {'ceers': 130.7889803307112, 'smacs': 144.6479834976019, 'glass': 251.2973235468314, 'uncover': 40.98680919}
    if angle is None:
        angle = angles[field]
    nc = webbpsf.NIRCam()
    nc.options['parity'] = 'odd'

    outname = os.path.join(output, 'psf_'+field+'_'+filt+'_'+str(fov)+'arcsec') # what to save as?

    # make an oversampled webbpsf
    if date is not None:
        nc.load_wss_opd_by_date(date, plot=False)
    nc.filter = filt
    nc.pixelscale = pixscl
    psf = nc.calc_psf(oversample=1, fov_arcsec=og_fov)[0].data

    # rotate and handle interpolation internally; keep k = 1 to avoid -ve pixels
    rotated = ndimage.rotate(psf, -angle, reshape=False, order=1, mode='constant', cval=0.0)

    clip = int((og_fov - fov) / 2 / nc.pixelscale)
    rotated = rotated[clip:-clip, clip:-clip]
    # print(np.shape(rotated))

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(rotated)
    Y, X = np.ogrid[:h, :w]
    r = fov / 2. / nc.pixelscale
    center = [w/2., h/2.]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    rotated /= np.sum(rotated[dist_from_center < r])
    rotated *= encircled # to get the missing flux accounted for

    # and save
    newhdu = fits.PrimaryHDU(rotated)
    newhdu.writeto(outname+'.fits', overwrite=True)

    return rotated



def make_cutout(ra, dec, size, nickname, filters, dir_images, precomp=None, row=None, plot=True, write=True, include_rgb=False, rgb=['f444w', 'f277w', 'f115w'], redshift=-99, matched_pattern='', dir='.'):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from matplotlib.colors import SymLogNorm
    import matplotlib.pyplot as plt
    import glob, sys

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    print(coord)

    from config import TARGET_ZP

    hdul = fits.HDUList()

    if include_rgb:
        from astropy.visualization import make_lupton_rgb

    if plot:
        if len(filters) > 7:
            fig, axes = plt.subplots(ncols=8, nrows=2, figsize=(3*8, 3*2))
        else:
            fig, axes = plt.subplots(ncols=len(filters), figsize=(3*len(filters), 3))

    for filt, ax in zip(filters, axes.flatten()):

        # print(filt)
        if precomp is None:
            fn = glob.glob(os.path.join(dir_images, f'*{filt}*_sci_skysubvar{matched_pattern}.fits.gz'))[0]
            if not os.path.exists(fn):
                print(f'WARNING -- image for {filt} does not exist at {fn}. Skipping!')
                continue
            # print('foo')
            hdu = fits.open(fn)[0]
            img = hdu.data
            # print('foo2')
            # if plot: # get these from the big mosaic!
            #     mean, median, rms = sigma_clipped_stats(img[img!=0], sigma=3)
            wcs = WCS(hdu.header)
        else:
            hdu, img, head = precomp[filt]
            wcs = WCS(head)

        if not wcs.footprint_contains(coord):
            print(f'CRITICAL -- {coord} is not within the image footprint!')
            sys.exit()
        try:
            cutout = Cutout2D(img, position=coord, size=size*u.arcsec, wcs=wcs)
        except:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            continue

        if plot: # get these from the big mosaic!
            img = cutout.data
            mean, median, rms = sigma_clipped_stats(img[img!=0], sigma=3)
            img -= median
            if np.isnan(rms):
                rms = 0.002

        if write:
            hdu.data = cutout.data
            hdu.header.update(cutout.wcs.to_header())
            hdu.name = filt
            hdul.append(hdu)

        if plot: # nice stamp pdfs scaled optimally to show noise + structure

            if filt == rgb[0]:
                r = img * 1.4
            if filt == rgb[1]:
                g = img * 1
            if filt == rgb[2]:
                b = img * 0.35

            if row is None:
                flux, fluxerr = -99, -99
                mag, magerr = -99, -99
                snr = -99
            if f'f_{filt}' not in row.colnames:
                mag, magerr = -99, -99
                flux, fluxerr = -99, -99
                snr = -99
            else:
                flux, fluxerr = row[f'f_{filt}'], row[f'e_{filt}']
                snr = flux / fluxerr

                mag = TARGET_ZP - 2.5*np.log10(flux)
                magerr = 2.5 / np.log(10) / (flux/fluxerr)

                if flux <=0:
                    mag, magerr = -1, flux/fluxerr

                # print(filt)
                # print(flux, fluxerr)
                # print(mag, magerr)
            scale = np.nanmax(img)
            if scale <= 0:
                scale = 0.02
            elif np.isnan(scale):
                scale = 1
            # print(filt, rms, scale, np.nanmedian(img), np.nanmin(img), np.nanmax(img), np.sum(img))
            ax.imshow(img, cmap='RdGy', norm=SymLogNorm(3*rms, 1, -scale, scale))
            ax.text(0.05, 1.05, f'{filt}\n{flux:2.2f}+/-{fluxerr:2.2f} 10*nJy (S/N:{snr:2.2f})', transform=ax.transAxes)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)


    if (plot & include_rgb):
        from astropy.visualization import make_lupton_rgb
        img = make_lupton_rgb(r, g, b, stretch=0.1, minimum=-0.01)
        fig_rgb, ax_rgb = plt.subplots(figsize=(5,5))
        ax_rgb.imshow(img)
        ax_rgb.text(0.01, 0.01, f'{rgb[0]}+{rgb[1]}+{rgb[2]}', transform=ax_rgb.transAxes, color='w', fontsize=15)
        ax_rgb.text(0.01, 0.90, f'{nickname} $z$ = {redshift:2.2f}', transform=ax_rgb.transAxes, color='w', fontsize=20)
        ax_rgb.axes.xaxis.set_visible(False)
        ax_rgb.axes.yaxis.set_visible(False)
        if write:
            fig_rgb.savefig(os.path.join(dir, f'cutouts/{nickname}_z{redshift:2.1f}_RGB.pdf'), dpi=300)

    if write:
        fig.savefig(os.path.join(dir, f'cutouts/{nickname}_z{redshift:2.1f}.pdf'), dpi=300)
        hdul.writeto(os.path.join(dir, f'cutouts/{nickname}_z{redshift:2.1f}.fits'), overwrite=True)


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def binned_med(X, Y, xrange=None, dbins=1.0, bins=None, use_scott=False):
    if use_scott:
        from astropy.stats import scott_bin_width
        __, bins = scott_bin_width(X, return_bins=True)
    elif bins is not None:
        bins = bins
    else:
        bins = np.arange(xrange[0], xrange[1], dbins)
    # delta = bins[1]-bins[0]

    Y = Y[X<bins[-1]]
    X = X[X<bins[-1]]
    idx  = np.digitize(X,bins[:-1])
    foo = np.array([median_confidence_interval(Y[idx==k]) for k in range(1, len(bins))])
    binned_median, running_std = foo[:,0], np.array((foo[:,1], foo[:,2]))
    Nbins = np.array([np.sum(idx==k) for k in range(1, len(bins))])
    bin_centers = bins[:-1] + np.diff(bins)/2.
    return Nbins, bin_centers, binned_median, running_std

def median_confidence_interval(data, confidence=0.34):
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    m = np.nanmedian(data)
    sdata = np.sort(data)
    hdata = sdata[sdata > m]
    ldata = sdata[sdata <= m]
    n_hdata = len(hdata)
    n_ldata = len(ldata)
    try:
        hmax = hdata[(np.arange(n_hdata) / n_hdata) < confidence][-1]
    except:
        hmax = m
    try:
        hmin = ldata[::-1][(np.arange(n_ldata) / n_ldata) < confidence][-1]
    except:
        hmin = m
    return m, hmin, hmax
