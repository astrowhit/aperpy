# Compute background
from astropy.stats import sigma_clipped_stats
import numpy as np
import sep
import os
from astropy.io import fits

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


# Empty Apertures
def emtpy_apertures(img, segmap, N=1E6, aper=[0.35, 0.7, 2.0], pixscl=0.04):
    from alive_progress import alive_bar
    import numpy as np
    import scipy.stats as stats
    import astropy.units as u
    from astropy.stats import sigma_clipped_stats
    from photutils.aperture import CircularAperture, aperture_photometry

    aperrad = np.array(aper) / 2. / pixscl # diam sky to rad pix
    maxaper = int(np.max(aper)) + 1

    size = np.shape(img)
    try:
        segsize = np.shape(segmap)
        assert(size == segsize, f'Image size ({size}) != seg size ({segsize})!')
    except:
        return {}
    
    kept = 0
    positions = np.zeros((N, 2))
    with alive_bar(N) as bar:
        while kept < N:
            px, py = np.random.uniform(0, 1, 2)
            x, y = int(px * size[0]), int(py * size[1])
            box = slice(x-maxaper,x+maxaper), slice(y-maxaper,y+maxaper)
            subseg, subimg = segmap[box], img[box]
            # print(x, y)
            if np.all(subseg==0) & (not np.any((subimg==0.0) | np.isnan(subimg))):
                positions[kept] = y, x
                kept += 1
                bar()

    # now do it in parallel
    apertures = [CircularAperture(positions, r=r) for r in aperrad]
    output = aperture_photometry(img, apertures)

    aperstats = {}
    aperstats['sigma_1'] = np.nanstd(img[segmap==0])
    # measure moments + percentiles; AD test
    for i, radius in enumerate(aper):
        phot = output[f'aperture_sum_{i}']
        # print(phot)
        aperstats[radius] = {}
        aperstats[radius]['mean'] = np.mean(phot)
        aperstats[radius]['std'] = np.std(phot)
        aperstats[radius]['snmad'] = 1.48 * np.median(np.abs(phot))
        aperstats[radius]['norm'] = stats.normaltest(phot)
        aperstats[radius]['median'] = np.median(phot)
        kmean, kmed, kstd = sigma_clipped_stats(phot)
        aperstats[radius]['kmean'] = kmean
        aperstats[radius]['kmed'] = kmed
        aperstats[radius]['kstd'] = kstd
        pc = np.percentile(phot, q=(5, 16, 84, 95))
        aperstats[radius]['pc'] = pc
        aperstats[radius]['interquart_68'] = pc[2] - pc[1] # 68pc

        # print(radius)
        # for key in aperstats[radius]:
        #     print(key, aperstats[radius][key])
        # print()

    return aperstats

# Star finder


# Make auto mask


# Make and rotate PSF
def get_psf(filt, field='ceers', angle=None, fov=4, og_fov=10, pixscl=0.04):
    # makes the PSF at og_fov and clips down to fov. Works with 0.04 "/px
    import webbpsf
    from astropy.io import fits
    import numpy as np
    from scipy import ndimage
    import os
    from astropy.io import ascii

    # Check if filter is valid and get correction term
    DIR_CORR = '.'
    if filt in ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']:
        # 17 corresponds with 2" radius (i.e. 4" FOV)
        encircled = ascii.read(os.path.join(DIR_CORR, 'Encircled_Energy_SW.txt'))[17][filt]
    elif filt in ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']:
        encircled = ascii.read(os.path.join(DIR_CORR, 'Encircled_Energy_LW.txt'))[17][filt]
    else:
        print(f'{filt} is NOT a valid NIRCam filter!')
        return

    # Observed PA_V3 for fields
    angles = {'ceers': 130.7889803307112, 'smacs': 144.6479834976019, 'glass': 251.2973235468314}
    if angle is None:
        angle = angles[field]
    nc = webbpsf.NIRCam()
    nc.options['parity'] = 'odd'
    
    outname = 'psf_'+field+'_'+filt+'_'+str(fov)+'arcsec' # what to save as?

    # make an oversampled webbpsf
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