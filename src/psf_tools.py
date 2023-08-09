from photutils.centroids import centroid_com
import astropy.units as u

from astropy.nddata import block_reduce

# from photutils.detection import find_peaks
import time
import numpy as np

np.errstate(invalid='ignore')

import matplotlib.pyplot as plt

from photutils.detection import find_peaks
from astropy.nddata import block_reduce
from astropy.stats import mad_std
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Linear1D
import astropy.units as u
import os
import glob

import numpy as np

import matplotlib.pyplot as plt

from math import atan2,degrees
from scipy.stats import loglaplace, chi2

from astropy.visualization import ImageNormalize
from astropy.stats import mad_std

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker


def plot_profile(psf, target):
    shape = psf.shape
    center = (shape[1]//2, shape[0]//2)
    radii_pix = np.arange(1,40,1)
    apertures = [CircularAperture(center, r=r) for r in radii_pix] #r in pixels

    phot_table = aperture_photometry(psf, apertures)
    flux_psf = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

    phot_table = aperture_photometry(target, apertures)
    flux_target = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

    return radii_pix[:-1], (flux_psf)[0:-1], (flux_target)[0:-1]


def powspace(start, stop, num=30, power=0.5, **kwargs):
    """Generate a square-root spaced array with a specified number of points
    between two endpoints.

    Parameters
    ----------
    start : float
        The starting value of the range.
    stop : float
        The ending value of the range.
    pow: power of distribution, defaults to sqrt
    num_points : int, optional
        The number of points to generate in the array. Default is 50.

    Returns
    -------
    numpy.ndarray
        A 1-D array of `num_points` values spaced equally in square-root space
        between `start` and `stop`.
    """
    return np.linspace(start**power, stop**power, num=num, **kwargs)**(1/power)


def measure_curve_of_growth(image, position=None, radii=None, rnorm='auto', nradii=30, verbose=False, showme=False, rbg='auto'):
    """
    Measure a curve of growth from cumulative circular aperture photometry on a list of radii centered on the center of mass of a source in a 2D image.

    Parameters
    ----------
    image : `~numpy.ndarray`
        2D image array.
    position : `~astropy.coordinates.SkyCoord`
        Position of the source.
    radii : `~astropy.units.Quantity` array
        Array of aperture radii.

    Returns
    -------
    `~astropy.table.Table`
        Table of photometry results, with columns 'aperture_radius' and 'aperture_flux'.
    """

    if type(radii) is type(None):
        radii = powspace(0.5,image.shape[1]/2,nradii)

    # Calculate the centroid of the source in the image
    if type(position) is type(None):
#        x0, y0 = centroid_2dg(image)
         position = centroid_com(image)

    if rnorm == 'auto': rnorm = image.shape[1]/2.0
    if rbg == 'auto': rbg = image.shape[1]/2.0

    apertures = [CircularAperture(position, r=r) for r in radii]

    if rbg:
        bg_mask = apertures[-1].to_mask().to_image(image.shape) == 0
        bg = np.nanmedian(image[bg_mask])
        if verbose: print('background',bg)
    else:
        bg = 0.

    # Perform aperture photometry for each aperture
    phot_table = aperture_photometry(image-bg, apertures)
    # Calculate cumulative aperture fluxes
    cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

    if rnorm:
        rnorm_indx = np.searchsorted(radii, rnorm)
        cog /= cog[rnorm_indx]


    area = np.pi*radii**2
    area_cog = np.insert(np.diff(area),0,area[0])
    profile = np.insert(np.diff(cog),0,cog[0])/area_cog
    profile /= profile.max()

    if showme:
        plt.plot(radii, cog, marker='o')
        plt.plot(radii,profile/profile.max())
        plt.xlabel('radius pix')
        plt.ylabel('curve of growth')

    # Create output table of aperture radii and cumulative fluxes
    return radii, cog, profile

def stamp_rms_snr(img, block_size=3, rotate=True):
    if rotate:
        p180 = np.flip(img,axis=(0,1))
        dp = img-p180
    else:
        dp = img.copy()

    s = dp.shape[1]
    buf = 6
    dp[s//buf:(buf-1)*s//buf,s//buf:(buf-1)*s//buf] = np.nan
    dp3 = block_reduce(dp,block_size=3)

    rms = mad_std(dp,ignore_nan=True)/block_size * np.sqrt(img.size)
    if rotate: rms /= np.sqrt(2)

    snr = img.sum()/rms

    return rms, snr

pixscale = (40<<u.mas)/1000


def show_cogs(*args, title='', linear=False, pixscale=0.04, label=None, outname=''):
    npsfs = len(args)
    nfilts = len(args[0])

    xtick = [0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0]
    plt.figure(figsize=(20,4.5))

    if not label:
        label = ['' for p in range(npsfs)]

    for filti in range(nfilts):
        psf_ref = args[0][filti]
        psf_ref2 = args[-1][filti]
        r, cog_ref, prof_ref = measure_curve_of_growth(psf_ref,nradii=50)
        r, cog_ref2, prof_ref2 = measure_curve_of_growth(psf_ref2,nradii=50)
        r = r * pixscale

        plt.subplot(141)
        plt.plot(r,prof_ref,label=label[0])
        plt.title(title+' profile')
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.yscale('log')
        plt.xlim(0,1)
        plt.ylim(1e-5,1)
        plt.xlabel('arcsec')
        plt.axhline(y=0,alpha=0.5,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        ax=plt.gca()
        rms, snr = stamp_rms_snr(psf_ref)
        dx, dy = centroid_com(psf_ref)
        plt.text(0.6,0.8,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C0')

        plt.subplot(142)

        plt.plot(r,cog_ref,label=label[0])
        plt.xlabel('arcsec')
        plt.title('cog')
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
#        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)

        plt.subplot(143)
        plt.plot(r,np.ones_like(r),label=label[0])
        plt.xlabel('arcsec')
        plt.title('cog / cog_'+label[0])
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.xlabel('arcsec')
        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)
        plt.ylim(0.5,1.5)

        plt.subplot(144)
        plt.plot(r,cog_ref/cog_ref2,label=label[0],c='C0')
        plt.xlabel('arcsec')
        plt.title('cog / cog_'+label[-1])
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.xlabel('arcsec')
        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)
        plt.ylim(0.5,1.5)


        cogs = []
        profs = []
        psfs = [psf_ref]
        for psfi in np.arange(1, npsfs):
            psf = args[psfi][filti]
            _, cog, prof = measure_curve_of_growth(psf,nradii=50)
            cogs.append(cog)
            profs.append(prof)
            dx, dy = centroid_com(psf)
            rms, snr = stamp_rms_snr(psf)

            plt.subplot(141)
            plt.plot(r,prof)

            plt.text(0.5,0.8-psfi*0.1,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C'+str(psfi))
            plt.xlim(0.02,1)

            plt.subplot(142)
            plt.plot(r,cog,label=label[psfi],c='C'+str(psfi))
            plt.legend()

            plt.subplot(143)
            plt.plot(r,cog/cog_ref,c='C'+str(psfi))

            plt.subplot(144)
            plt.plot(r,cog/cog_ref2,c='C'+str(psfi))

            psfs.append(psf)

        plt.savefig('_'.join([outname,'psf_cog.pdf']),dpi=300)

        _ = imshow(psfs,cross_hairs=True,nsig=50,title=label)

        plt.savefig('_'.join([outname,'psf_average.pdf']),dpi=300)

def get_filename(imagedir, filt, skyext=''):

    # if filt in ['f090w','f115w','f150w','f200w']:
    #     add = '_block40'
    # else:
    #     add=''
    # if filt in ['f435w','f606w','f814w','f105w','f125w','f140w','f160w']:
    #     field = 'buffalo_v5.4'
    # else:
    #     field = 'uncover_v6.0'


    filename = glob.glob(os.path.join(imagedir, f'*{filt}*sci*{skyext}.fits.gz'))[0]
    # filename = field+'_abell2744clu_'+filt+add+'_bcgs_sci.fits.gz'
    starname = filename.replace('.fits','')+'_star_cat.fits'

    return filename, starname


def find_stars(filename=None, block_size=5, npeaks=1000, size=15, radii=[0.5,1.,2.,4.,7.5], range=[0,4], mag_lim = 24.0,
               threshold_min = -0.5, threshold_mode=[-0.2,0.2], shift_lim=2, zp=28.9, instars=None, showme=True, label='',
               outdir='./', plotdir='./'):

    img, hdr = fits.getdata(filename, header=True)
    wcs = WCS(hdr)

    imgb = block_reduce(img, block_size, func=np.sum)
    sig = mad_std(imgb[imgb>0], ignore_nan=True)/block_size

#    img[~np.isfinite(img)] = 0.0
    peaks = find_peaks(img, threshold=10*sig, npeaks=npeaks)
    # print(peaks)
    peaks.rename_column('x_peak','x')
    peaks.rename_column('y_peak','y')
    ra,dec = wcs.all_pix2world(peaks['x'], peaks['y'], 0)
    peaks['ra'] = ra
    peaks['dec'] = dec
    peaks['x0'] = 0.0
    peaks['y0'] = 0.0
    peaks['minv'] = 0.0
    for ir in np.arange(len(radii)): peaks['r'+str(ir)] = 0.
    for ir in np.arange(len(radii)): peaks['p'+str(ir)] = 0.

    t0 = time.time()
    stars = []
    for ip,p in enumerate(peaks):
        co = Cutout2D(img, (p['x'], p['y']), size, mode='partial')
        # measure offset, feed it to measure cog
        # if offset > 3 pixels -> skip
        position = centroid_com(co.data)
        peaks['x0'][ip] = position[0] - size//2
        peaks['y0'][ip] = position[1] - size//2
        peaks['minv'][ip] = np.nanmin(co.data)
        _ , cog, profile = measure_curve_of_growth(co.data, radii=np.array(radii), position=position, rnorm=None, rbg=None)
        for ir in np.arange(len(radii)): peaks['r'+str(ir)][ip] = cog[ir]
        for ir in np.arange(len(radii)): peaks['p'+str(ir)][ip] = profile[ir]
        co.radii = np.array(radii)
        co.cog = cog
        co.profile = profile
        stars.append(co)

    stars = np.array(stars)

    peaks['mag'] = zp-2.5*np.log10(peaks['r4'])
    r = peaks['r4']/peaks['r2']
    shift_lim_root = np.sqrt(shift_lim)

    ok_mag =  peaks['mag'] < mag_lim
    ok_min =  peaks['minv'] > threshold_min
    ok_phot = np.isfinite(peaks['r'+str(len(radii)-1)]) &  np.isfinite(peaks['r2']) & np.isfinite(peaks['p1'])
    ok_shift = (np.sqrt(peaks['x0']**2 + peaks['y0']**2) < shift_lim) & \
               (np.abs(peaks['x0']) < shift_lim_root) & (np.abs(peaks['y0']) < shift_lim_root)

    # ratio apertures @@@ hardcoded
    h = np.histogram(r[(r>1.2) & ok_mag], bins=np.arange(0, range[1], threshold_mode[1]/2.),range=range)
    ih = np.argmax(h[0])
    rmode = h[1][ih]
    ok_mode =  ((r/rmode-1) > threshold_mode[0]) & ((r/rmode-1) < threshold_mode[1])
    ok = ok_phot & ok_mode & ok_min & ok_shift & ok_mag
        
    # sigma clip around linear relation
    try:
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, sigma=2.8, niter=2)
        lfit, outlier = fitter(Linear1D(),x=zp-2.5*np.log10(peaks['r4'][ok]),y=(peaks['r4']/peaks['r2'])[ok])
        ioutlier = np.where(ok)[0][outlier]
        ok[ioutlier] = False
    except:
        print('linear fit failed')
        ioutlier = 0
        lfit = None

    mags = zp-2.5*np.log10(peaks['r4'])

    peaks['id'] = 1
    peaks['id'][ok] = np.arange(1,len(peaks[ok])+1)

    if showme:
        if not os.path.exists(outdir): os.mkdir(outdir)
        plt.figure(figsize=(14,8))
        plt.subplot(231)
        mags = peaks['mag']
        mlim_plot = np.nanpercentile(mags,[5,95]) + np.array([-2,1])
        # print(mlim_plot)
        plt.scatter(mags,r,10)
        plt.scatter(mags[~ok_shift],r[~ok_shift],10,label='bad shift',c='C2')
        plt.scatter(mags[ok],r[ok],10,label='ok',c='C1')
        plt.scatter(mags[ioutlier],r[ioutlier],10,label='outlier',c='darkred')
        if lfit: plt.plot(np.arange(14,30), lfit(np.arange(14,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
        plt.legend()
        plt.ylim(0,14)
        plt.xlim(mlim_plot[0],mlim_plot[1])
        plt.title(' aper(2) / aper(4) vs mag(aper(4))')

        plt.subplot(232)
        ratio_median = np.nanmedian(r[ok])
        plt.scatter(mags,r,10)
        plt.scatter(mags[~ok_shift],r[~ok_shift],10,label='bad shift',c='C2')
        plt.scatter(mags[ok],r[ok],10,label='ok',c='C1')
        plt.scatter(mags[ioutlier],r[ioutlier],10,label='outlier',c='darkred')
        if lfit: plt.plot(np.arange(15,30), lfit(np.arange(15,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
        plt.legend()
        plt.ylim(ratio_median-1,ratio_median+1)
        plt.xlim(mlim_plot[0],mlim_plot[1])
        plt.title('aper(2) / aper(4) vs mag(aper(4))')

        plt.subplot(233)
        _ = plt.hist(r,bins=range[1]*20,range=range)
        _ = plt.hist(r[ok],bins=range[1]*20,range=range)
        plt.title('aper(2) / aper(4)')

        plt.subplot(234)
        plt.scatter(zp-2.5*np.log10(peaks['r3'][ok]),(peaks['peak_value']/peaks['r3'])[ok])
        plt.scatter(zp-2.5*np.log10(peaks['r3'])[ioutlier],(peaks['peak_value'] /peaks['r3'])[ioutlier],c='darkred')
        plt.ylim(0,1)
        plt.title('peak / aper(3) vs maper(3)')

        plt.subplot(235)
        plt.scatter(peaks['x0'][ok],peaks['y0'][ok],c='C1')
        plt.scatter(peaks['x0'][ioutlier],peaks['y0'][ioutlier],c='darkred')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.title('offset (pix)')

        plt.subplot(236)
        plt.scatter(peaks['x'][ok],peaks['y'][ok],c='C1')
        plt.scatter(peaks['x'][ioutlier],peaks['y'][ioutlier],c='darkred')
        plt.axis('scaled')
        plt.title('position (pix)')
        plt.tight_layout()
        suffix = '.fits' + filename.split('.fits')[-1]
        plt.savefig(outdir+'/'+os.path.basename(filename).replace(suffix,'_diagnostic.pdf'))
        
        dd = [st.data for st in stars[ok]]
        title = ['{} {:.1f} {:.2f} {:.2f} {:.1f} {:.1f}'.format(ii, mm, pp,qq,xx,yy) for ii,mm,pp,qq,xx,yy in zip(peaks['id'][ok],mags[ok],peaks['p1'][ok],peaks['minv'][ok],peaks['x0'][ok],peaks['y0'][ok])]
        imshow(dd,nsig=30,title=title)
        plt.tight_layout()
        plt.savefig(outdir+'/'+os.path.basename(filename).replace(suffix,'_star_stamps.pdf'))
    
    peaks[ok].write(outdir+'/'+os.path.basename(filename).replace(suffix,'_star_cat.fits'),overwrite=True)
                
    return peaks[ok], stars[ok]


import cv2
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from photutils import CircularAperture, aperture_photometry
from astropy.table import hstack
import pickle
from astropy.io import fits

import cv2

from astropy.stats import sigma_clip

from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)

def grow(mask, structure=disk(2), **kwargs):
    return binary_dilation(mask,structure=structure,**kwargs)

# grow mask not along sigma clip axis but in 2d axis
def sigma_clip_3d(data, maxiters=2, axis=0, **kwargs):
    clipped_data = data.copy()
    for i in range(maxiters):
        clipped_data, lo, hi = sigma_clip(clipped_data, maxiters=0, axis=0, masked=True, grow=False, return_bounds=True, **kwargs)
        # grow mask
        for i in range(len(clipped_data.mask)): clipped_data.mask[i,:,:] = grow(clipped_data.mask[i,:,:],iterations=1)

    return np.mean(clipped_data,axis=axis), lo, hi, clipped_data

# interpolation=cv2.INTER_LANCZOS4
def imshift(img, ddx, ddy, interpolation=cv2.INTER_CUBIC):
    # affine matrix
    M = np.float32([[1,0,ddx],[0,1,ddy]])
    # output shape
    wxh = img.shape[::-1]
    return cv2.warpAffine(img, M, wxh, flags=interpolation)


def stamp_rms_snr(img, block_size=3, rotate=True):
    if rotate:
        p180 = np.flip(img,axis=(0,1))
        dp = img-p180
    else:
        dp = img.copy()

    s = dp.shape[1]
    buf = 6
    dp[s//buf:(buf-1)*s//buf,s//buf:(buf-1)*s//buf] = np.nan
    dp3 = block_reduce(dp,block_size=3)

    rms = mad_std(dp,ignore_nan=True)/block_size * np.sqrt(img.size)
    if rotate: rms /= np.sqrt(2)

    snr = img.sum()/rms
    return rms, snr

from astropy.table import Table


class PSF():
    def __init__(self, image=None, x=None, y=None, ids=None, pixsize=101, pixelscale=0.04):
        if type(image) == np.ndarray:
            img = image
            xx=x
            yy=y
            self.filename = None
        else:
            img, hdr = fits.getdata(image, header=True)
            wcs = WCS(hdr)
            xx,yy = wcs.all_world2pix(x, y, 0)
            self.filename  = image

        if type(ids) == type(None):
            ids = np.arange(1,len(x)+1)

        self.nx = pixsize
        self.c0 = self.nx//2
        self.cat = Table([ids,xx,yy,x,y],names=['id','x','y','ra','dec'])

        data = np.array([Cutout2D(img, (xx[i],yy[i]), (pixsize, pixsize),mode='partial').data for i in np.arange(len(x))])
        self.data = np.ma.array(data,mask = ~np.isfinite(data) | (data == 0) )
        self.data_orig = self.data.copy()
        self.ok = np.ones(len(self.cat))

    def phot(self, radius=8):
        pos = np.array([self.cat['x'],self.cat['y']])

        caper = CircularAperture((self.c0,self.c0), r=radius)
        cmask = Cutout2D(caper.to_mask(),(radius,radius), self.nx,mode='partial').data   # check ding galfit for better way
        phot = [aperture_photometry(st, caper)['aperture_sum'][0] for st in self.data]
        return phot

    def measure(self,norm_radius=8):
        peaks = []
        c0 = self.nx//2
        self.norm_radius = norm_radius

        peaks = np.array([st.max()for st in self.data])
        peaks[~np.isfinite(peaks) | (peaks==0)] = 0
        pos = np.array([self.cat['x'],self.cat['y']])

#        data = np.array(self.data)
#        data[self.data.mask] = 0
        caper = CircularAperture((self.c0,self.c0), r=self.norm_radius)
        cmask = Cutout2D(caper.to_mask(),(self.norm_radius,self.norm_radius), self.nx,mode='partial').data

        phot = self.phot(radius=self.norm_radius)
        sat =  [aperture_photometry(st, caper)['aperture_sum'][0] for st in np.array(self.data)] # casting to array removes mask
        cmin = [ np.nanmin(st*cmask) for st in self.data]

        self.cat['frac_mask'] = 0.0

        for i in np.arange(len(self.data)):
            self.data[i].mask |= (self.data[i]*cmask) < 0.0

        self.cat['peak'] =  peaks
        self.cat['cmin'] =  np.array(cmin)
        self.cat['phot'] =  np.array(phot)
        self.cat['saturated'] =  np.int32(~np.isfinite(np.array(sat)))

        flip_ratio = []
        rms_array = []
        for st in self.data:
            rms, snr = stamp_rms_snr(st)
            rms_array.append(rms)
            dp = st - np.flip(st,axis=(0,1))
            flip_ratio.append(np.abs(st).sum()/np.abs(dp).sum())

        self.cat['snr'] = 2*np.array(phot)/np.array(rms_array)
        self.cat['flip_ratio'] = np.array(flip_ratio)
        self.cat['phot_frac_mask'] = 1.0

    def select(self, snr_lim = 800, dshift_lim=3, mask_lim=0.40, phot_frac_mask_lim = 0.85, showme=False, **kwargs):
        self.ok = (self.cat['dshift'] < dshift_lim) & (self.cat['snr'] > snr_lim) & (self.cat['frac_mask'] < mask_lim) & (self.cat['phot_frac_mask'] > phot_frac_mask_lim)

    #(self.cat['cmin'] >= -1.5)  #& (self.cat['cmin'] >= -1.5)
        self.cat['ok'] = np.int32(self.ok)
        self.cat['ok_shift'] = (self.cat['dshift'] < dshift_lim)
        self.cat['ok_snr'] = (self.cat['snr'] > snr_lim)
        self.cat['ok_frac_mask'] = (self.cat['frac_mask'] < mask_lim)
        self.cat['ok_phot_frac_mask'] = (self.cat['phot_frac_mask'] > phot_frac_mask_lim)

        for c in self.cat.colnames:
            if 'id' not in c: self.cat[c].format='.3g'

        if showme:
            title = f"{self.cat['id']}, {self.cat['ok']}"
            fig, ax = imshow(self.data, title=title,**kwargs)
            fig.savefig('test.pdf',dpi=300)
            # self.cat.pprint_all()


    def stack(self,sigma=3,maxiters=2):
        iok = np.where(self.ok)[0]

        norm = self.cat['phot'][iok]
        data = self.data_orig[iok].copy()
        for i in np.arange(len(data)): data[i] = data[i]/norm[i]

        stack, lo, hi, clipped = sigma_clip_3d(data,sigma=sigma,axis=0,maxiters=maxiters)
        self.clipped = clipped
       # self.clipped[~np.isfinite(self.clipped)] = 0

        # print('-',len(self.ok[self.ok]))

        for i in np.arange(len(data)):
            self.ok[iok[i]] = self.ok[iok[i]] and ~self.clipped[i].mask[50,50]
            self.data[iok[i]].mask = self.clipped[i].mask
            mask = self.data[iok[i]].mask
            self.cat['frac_mask'][iok[i]] = np.size(mask[mask]) / np.size(mask)

        self.psf_average = stack
        self.cat['phot_frac_mask'] = self.phot(radius=self.norm_radius)/self.cat['phot']

    def show_by_id(self, ID, **kwargs):
        indx = np.where(self.cat['id']==ID)[0][0]
        imshow([self.data[indx]], **kwargs)
        return self.data[indx]

    def growth_curves(self):
        for i in np.arange(len(self.data)):
            radii, cog, profile = measure_curve_of_growth(a)
            r = radii*self.pixelscale

    def center(self,window=21,interpolation=cv2.INTER_CUBIC):
        if 'x0' in self.cat.colnames: return

        cw = window//2
        c0 = self.c0
        pos = []
        for i in np.arange(len(self.data)):
            p = self.data[i,:,:]
            st = Cutout2D(p,(self.c0,self.c0),window,mode='partial',fill_value=0).data
            st[~np.isfinite(st)] = 0
            x0, y0 = centroid_com(st)

            p = imshift(p, (cw-x0), (cw-y0),interpolation=interpolation)

            # now measure shift on recentered cutout
            # first in small window
            st = Cutout2D(p,(self.c0,self.c0),window,mode='partial',fill_value=0).data
            x1,y1 = centroid_com(st)

            # now in central half of stamp
            st2 = Cutout2D(p,(self.c0,self.c0),int(self.nx*0.5),fill_value=0).data
            # measure moment shift in positive definite in case there are strong ying yang residuals
            x2,y2 = centroid_com(np.maximum(p,0))

            p = np.ma.array(p, mask = ~np.isfinite(p) | (p==0))
            self.data[i,:,:] = p

            # difference in shift between central window and half of stamp is measure of contamination
            # from bright off axis sources
            dsh = np.sqrt(((c0-x2)-(cw-x1))**2 + ((c0-y2)-(cw-y1))**2)
            pos.append([cw-x0,cw-y0,cw-x1,cw-y1,dsh])

        self.cat = hstack([self.cat,Table(np.array(pos),names=['x0','y0','x1','y1','dshift'])])

    def save(self, outname=''):
        # with open('_'.join([outname, 'psf_stamps.fits']), 'wb') as handle:
        #     self.data[self.ok]

        # self.data[self.ok].filled(-99).write('_'.join([outname, 'psf_stamps.fits']),overwrite=True)

        fits.writeto('_'.join([outname, 'psf.fits']), np.array(self.psf_average),overwrite=True)

        self.cat[self.ok].write('_'.join([outname, 'psf_cat.fits']),overwrite=True)

        title = f"{self.cat['id']}, {self.cat['ok']}"
        fig, ax = imshow(self.data, nsig=30, title=title)
        fig.savefig('_'.join([outname, 'psf_stamps.pdf']),dpi=300)


#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)


def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)


import os
from astropy import cosmology
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.coordinates import ICRS
from astropy.wcs import WCS as pywcs
from astropy.stats import mad_std as mad
from scipy.ndimage import median_filter, generic_filter
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import (MinMaxInterval, LinearStretch, SqrtStretch, AsinhStretch,
                                   ImageNormalize)
from astropy.visualization import lupton_rgb

def plot_cross_hairs(n, r, lw=4, color='white'):
    c = n/2
    plt.plot([n/2,n/2],[n/2-2*r,n/2-r],c=color,lw=lw)
    plt.plot([n/2-2*r,n/2-r],[n/2,n/2],c=color,lw=lw)

def cutout_by_list(images, cat, size=2.4, scale=5, diam_aper=1/6.0, zout=None,
                     colnames=None, detection=None, outname='', oid=None,
                     color='black', rgbindex=[8,5,3], lupton_kw={'minimum':-0.3, 'stretch':2, 'Q':8}, ):
    ra = cat['ra']
    dec = cat['dec']
    id = cat['id']

    if outname:
        if not os.path.exists(outname): os.mkdir(outname)

    position = SkyCoord( ICRS(ra=ra*u.deg, dec=dec*u.deg))
    nfilt = len(images)
    nobj = len(ra)

    cutout_grid = []
    for ix,file in enumerate(images):
        img, hdr = fits.getdata(file,header=True)
        wcs = pywcs(hdr)

        cutout_list = []
        for iy,p in enumerate(position):
            cutout = Cutout2D(img, p, (size*u.arcsec, size*u.arcsec), wcs=wcs, copy=True)
            cutout_list.append(cutout)

        cutout_grid.append(cutout_list)

    cutout_array = np.array(cutout_grid,dtype='object')

    nstamps = nfilt + 1 if any(rgbindex) else 0
    w=1.44
    fs = 12
    lw = 1
    # print('width',w*nstamps,nstamps, nobj)
    fig, ax = plt.subplots(figsize=(w*nstamps,w*nobj))

    for ipage,p in enumerate(position):
        obj_stamps = cutout_array[:,ipage]

        if len(oid) > 0:
            textid = str(oid[ipage])
        else:
            textid = str(id[ipage])

        # print(len(obj_stamps),obj_stamps.shape, nstamps)
        for istamp,stamp in enumerate(obj_stamps):
            v = mad(stamp.data)
            npix = stamp.data.shape[0]
            plt.subplot(nobj, nstamps, istamp+1 + ipage*nstamps)

            im = plt.imshow(-stamp.data+4.5*v,origin='lower',vmin=-scale*v,vmax=scale*v, cmap='gray')
#            im.set_in_layout(True)
            plot_cross_hairs(npix, diam_aper/2*npix, lw=lw, color=color)

            if istamp == 0:
                plt.text(npix/20, 4.8*npix/6, textid, color='black',fontsize=fs) # weight='bold',

            if ipage == 0:
                plt.title(colnames[istamp],fontsize=fs)

            plt.xticks([])
            plt.yticks([])

        if  any(rgbindex):
            plt.subplot(nobj, nstamps, nstamps + ipage*nstamps)
            # print(images[rgbindex[0]],images[rgbindex[1]],images[rgbindex[2]])
            rgbscale = 0.3
            rgboff = 0.2
            r = obj_stamps[rgbindex[0]].data/scale*rgbscale/v   + rgboff
            g = obj_stamps[rgbindex[1]].data/scale*rgbscale/v*1.3 + rgboff
            b = obj_stamps[rgbindex[2]].data/scale*rgbscale/v*1.6 + rgboff
            # print(r.min(),r.max())
            norm = ImageNormalize(vmin=-scale*v, vmax=scale*v, stretch=SqrtStretch())
            img = lupton_rgb.make_lupton_rgb(r,g,b, **lupton_kw)
     #       plt.imshow(img, origin='lower',norm=norm)

            plt.imshow(np.array([r, g, b]).transpose(1,2,0), origin='lower', norm=norm)
            plt.xticks([])
            plt.yticks([])

            #            plt.axis('off')


    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)
    plt.savefig('figs/stamps'+outname+'.pdf', bbox_inches='tight',dpi=300)
#    plt.savefig('figs/stamps'+outname+'.svg', bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()


# transfers ax to new fig?
# changes geometry?
def plot_axes(ax, fig=None, geometry=(1,1,1)):
    if fig is None:
        fig = plt.figure()
    if ax.get_geometry() != geometry :
        ax.change_geometry(*geometry)
    ax = fig.axes.append(ax)
    return fig


# note, always expects a list of images, so for single image do [image]
def imshow(args, cross_hairs=False, log=False, **kwargs):
    width = 20
    nargs = len(args)
    if nargs == 0: return
    if not (ncol := kwargs.get('ncol')): ncol = int(np.ceil(np.sqrt(nargs)))+1
    if not (nsig := kwargs.get('nsig')): nsig = 5
    if not (stretch := kwargs.get('stretch')): stretch = LinearStretch()

    nrow = int(np.ceil(nargs/ncol))
    panel_width = width/ncol
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(ncol*panel_width,nrow*panel_width))

    if type(ax) is not np.ndarray: ax = np.array(ax)
    for arg, axi in zip(args, ax.flat):
        sig = mad_std(arg[(arg != 0) & np.isfinite(arg)])
        if sig == 0: sig=1
        norm = ImageNormalize(np.float32(arg), vmin=-nsig*sig, vmax=nsig*sig, stretch=stretch)
#        axi.imshow(arg, norm=norm, origin='lower', cmap='gray',interpolation='nearest')
        axi.imshow(arg, norm=norm, origin='lower', interpolation='nearest')
        axi.set_axis_off()
        if cross_hairs:
            axi.plot(50,50, color='red', marker='+', ms=10, mew=1)
#    c = n/2
#    plt.plot([n/2,n/2],[n/2-2*r,n/2-r],c=color,lw=lw)
#    plt.plot([n/2-2*r,n/2-r],[n/2,n/2],c=color,lw=lw)
#           plot_cross_hairs(arg.shape[0],arg.shape[0]//4,color='red')

    if type(title := kwargs.get('title')) is not type(None):
        for fi,axi in zip(title,ax.flat): axi.set_title(fi)

    return fig, ax

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

mscale.register_scale(SquareRootScale)

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

from collections import defaultdict
import yaml


if 0:
    X = np.linspace(0,1,500)
    A = [1,2,5,10,20]
    funcs = [np.arctan,np.sin,loglaplace(4).pdf,chi2(5).pdf]

    plt.subplot(221)
    for a in A:
        plt.plot(X,np.arctan(a*X),label=str(a))

    labelLines(plt.gca().get_lines(),zorder=2.5)

    plt.subplot(222)
    for a in A:
        plt.plot(X,np.sin(a*X),label=str(a))

    labelLines(plt.gca().get_lines(),align=False,fontsize=14)

    plt.subplot(223)
    for a in A:
        plt.plot(X,loglaplace(4).pdf(a*X),label=str(a))

    xvals = [0.8,0.55,0.22,0.104,0.045]
    labelLines(plt.gca().get_lines(),align=False,xvals=xvals,color='k')

    plt.subplot(224)
    for a in A:
        plt.plot(X,chi2(5).pdf(a*X),label=str(a))

    lines = plt.gca().get_lines()
    l1=lines[-1]
    labelLine(l1,0.6,label=r'$Re=${}'.format(l1.get_label()),ha='left',va='bottom',align = False)
    labelLines(lines[:-1],align=False)

    plt.show()

def renorm_psf(psfmodel, filt, fov=4.04, pixscl=0.04):
    
    filt = filt.upper()

    # Encircled energy for WFC3 IR within 2" radius, ACS Optical, and UVIS from HST docs
    encircled = {}
    encircled['F225W'] = 0.993
    encircled['F275W'] = 0.984
    encircled['F336W'] = 0.9905
    encircled['F435W'] = 0.979
    encircled['F606W'] = 0.975
    encircled['F775W'] = 0.972
    encircled['F814W'] = 0.972
    encircled['F850LP'] = 0.970
    encircled['F098M'] = 0.974
    encircled['F105W'] = 0.973
    encircled['F125W'] = 0.969
    encircled['F140W'] = 0.967
    encircled['F160W'] = 0.966
    encircled['F090W'] = 0.9837
    encircled['F115W'] = 0.9822
    encircled['F150W'] = 0.9804
    encircled['F200W'] = 0.9767
    encircled['F277W'] = 0.9691
    encircled['F356W'] = 0.9618
    encircled['F410M'] = 0.9568
    encircled['F444W'] = 0.9546

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(psfmodel)
    Y, X = np.ogrid[:h, :w]
    r = fov / 2. / pixscl
    center = [w/2., h/2.]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    psfmodel /= np.sum(psfmodel[dist_from_center < r])
    psfmodel *= encircled[filt] # to get the missing flux accounted for

    return psfmodel