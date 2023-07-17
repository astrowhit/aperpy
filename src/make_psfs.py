# This is just a wrapper around the webb_tool
from webb_tools import get_webbpsf, get_date
from astropy.io import fits
import numpy as np
import os, sys
from astropy.nddata import block_reduce
from photutils.psf import create_matching_kernel, SplitCosineBellWindow
from scipy.ndimage import zoom
from astropy.visualization import simple_norm


PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_PSFS, PIXEL_SCALE, DIR_OUTPUT, PSF_FOV, FILTERS, MATCH_BAND, SKYEXT, DIR_KERNELS
from psf_tools import *

method = 'pypher'
pypher_r = 3e-3

oversample = 3
outdir = DIR_PSFS

target_filter = MATCH_BAND
image_dir = DIR_OUTPUT
print('target filter',target_filter)
hdr = fits.getheader(glob.glob(os.path.join(DIR_OUTPUT, f'*{target_filter}*sci*{SKYEXT}.fits.gz'))[0])

alpha = 0.3
beta = 0.15
window = SplitCosineBellWindow(alpha=alpha, beta=beta)

use_filters = [MATCH_BAND] + [f for f in FILTERS if f != MATCH_BAND]
for pfilt in use_filters:
    filename = glob.glob(os.path.join(DIR_OUTPUT, f'*{pfilt}*sci*{SKYEXT}.fits.gz'))[0]
    starname = filename.replace('.fits', '_star_cat.fits').replace(DIR_OUTPUT, DIR_PSFS)
    outname = os.path.join(DIR_PSFS, f'{pfilt}.fits')

    peaks, stars = find_stars(filename, outdir=DIR_PSFS, label=pfilt)

    print(len(peaks))
    # if f != 'f090w': continue

    snr_lim = 1000
    sigma = 2.8 if pfilt in ['f090w'] else 4.0
    #snr_lim = 500 if f in ['f435w','f606w','f814w','f105w','f125w','f140w','f160w'] else 800
    showme=False

    maglim = [18.,24.0]
    ok = (peaks['mag'] > maglim[0]) & ( peaks['mag'] < maglim[1] )
    ra, dec, ids = peaks['ra'][ok], peaks['dec'][ok], peaks['id'][ok]

    print(filename, starname)

    psf = PSF(image=filename, x=ra, y=dec, ids=ids, pixsize=101)
    psf.center()
    psf.measure()
    psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.99, showme=showme, nsig=30)
    psf.stack(sigma=sigma)
    psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.4, showme=True, nsig=30)
    psf.save(outname.replace('.fits',''))
    
    imshow(psf.data[psf.ok],nsig=50,title=psf.cat['id'][psf.ok])
    plt.savefig(outname.replace('.fits','_stamps_used.pdf'),dpi=300)
    show_cogs([psf.psf_average],title=pfilt, label=['oPSF'],outname=DIR_PSFS+pfilt)
    # psf.cat[psf.ok]

    # [show_cogs([psf.psf_average], [i]) for i in psf.data[psf.ok].data]
    filt_psf = np.array(psf.psf_average)
    if pfilt == MATCH_BAND:
        target_psf = filt_psf
    psfname = glob.glob(DIR_PSFS+'*'+pfilt.lower()+'*'+'psf.fits')[0]
    outname = DIR_KERNELS+os.path.basename(psfname).replace('psf','kernel')
    print(pfilt,psfname,outname)
    
#    if pfilt in ['f105w','f125w','f140w','f160w','f410m','f444w']: pypher_r = 3e-3
    
    filt_psf = fits.getdata(psfname)
    if oversample > 1:  filt_psf = zoom(filt_psf,3.0)

    filt_psf /= filt_psf.sum()
    if method == 'pypher':
        fits.writeto(DIR_KERNELS+'psf_a.fits',filt_psf,header=hdr,overwrite=True)
        fits.writeto(DIR_KERNELS+'psf_b.fits',target_psf,header=hdr,overwrite=True)
        os.system(f'pypher {DIR_KERNELS}psf_a.fits {DIR_KERNELS}psf_b.fits {DIR_KERNELS}kernel_a_to_b.fits -r {pypher_r:.3g}')
        kernel = fits.getdata('kernel_a_to_b.fits')
        os.remove(DIR_KERNELS+'psf_a.fits')
        os.remove(DIR_KERNELS+'psf_b.fits')
        os.remove(DIR_KERNELS+'kernel_a_to_b.fits')

    else:
        kernel =  create_matching_kernel(filt_psf, target_psf, window=window)
        
    simple = simple_norm(kernel,stretch='linear',power=1, min_cut=-5e-4/oversample**2, max_cut=5e-4/oversample**2)

    if oversample > 1:
        if 1:
            kernel = block_reduce(kernel,block_size=oversample, func=np.sum) 
            filt_psf = block_reduce(filt_psf,block_size=oversample, func=np.sum) 
            target_psf_bin = block_reduce(target_psf,block_size=oversample, func=np.sum) 
            kernel /= kernel.sum()
            filt_psf /= filt_psf.sum()
            target_psf_bin /= target_psf_bin.sum()
    else:
        target_psf_bin = target_psf

    fits.writeto(outname.lower(), np.float32(np.array(kernel/kernel.sum())),overwrite=True)

nfilt = len(use_filters)
plt.figure(figsize=(30,nfilt*4))
npanel = 7
for pfilt in use_filters:

    psfname = glob.glob(DIR_PSFS+'*'+pfilt.lower()+'*'+'psf.fits')[0]
    outname = DIR_KERNELS+os.path.basename(psfname).replace('psf','kernel')

    filt_psf = fits.getdata(psfname)
    if oversample > 1:  filt_psf = zoom(filt_psf,3.0)
    filt_psf /= filt_psf.sum()
    kernel = fits.getdata(outname.lower())

    simple = simple_norm(kernel,stretch='linear',power=1, min_cut=-5e-4, max_cut=5e-4)
    plt.imshow(kernel, norm=simple, interpolation='antialiased',origin='lower')

    print('centroid psf,target,kernel', centroid_2dg(filt_psf), centroid_2dg(target_psf_bin), centroid_2dg(kernel))
    filt_psf_conv = convolve_fft(filt_psf, kernel)
    print(np.max(filt_psf),filt_psf.shape)

    plt.subplot(nfilt,npanel,4+i*npanel)
    plt.title("convolved "+pfilt)
    plt.imshow(filt_psf_conv, norm=simple, interpolation='antialiased',origin='lower')

    plt.subplot(nfilt,npanel,5+i*npanel)
    plt.title("residual "+pfilt)
    res = filt_psf_conv-target_psf_bin
    plt.imshow(res, norm=simple, interpolation='antialiased',origin='lower')

    plt.subplot(nfilt,npanel,7+i*npanel)
    r,pf,pt = plot_profile(filt_psf_conv,target_psf_bin)
    plt.ylim(0.95,1.05)
    if method == 'pypher': 
        plt.title('pypher r={}'.format(pypher_r))
    else:
        plt.title('alpha={}, beta={}'.format(alpha,beta))
    plt.axvline(x=0.16,ls=':')
    plt.xlabel('radius arcsec')
    plt.ylabel('ee_psf_conv / ee_psf_target')

    plt.subplot(nfilt,npanel,6+i*npanel)
    plt.title('COG / COG_target')
    plt.plot(r*0.04,pf,lw=3)
    plt.plot(r*0.04,pt,'--',alpha=0.7,lw=3)
    plt.xlabel('radius arcsec')
    plt.ylabel('ee')

    plt.tight_layout()
    plt.savefig(DIR_PSFS+'kernels.pdf',dpi=300)

# ----------------

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (imshow_norm, MinMaxInterval, simple_norm,
                                   SqrtStretch, SinhStretch, LogStretch)

from photutils import CircularAperture
from photutils import aperture_photometry
from photutils.psf import create_matching_kernel, TukeyWindow, TopHatWindow, SplitCosineBellWindow
from photutils.centroids import centroid_com, centroid_quadratic
from photutils.centroids import centroid_1dg, centroid_2dg
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
#import webbpsf
#from PIL import Image
from astropy.io import fits

#ht = Table.read('HST_PSF_ee.fits')




# Default behavior generates a 10" FOV PSF and clips down to 4" FOV; 0.04 "/px
# for filt in FILTERS:
#     filt = filt.upper()
#     if filt not in USE_FILTERS: continue
#     print(f'Fetching WebbPSF for {filt} at PA {ANGLE}deg for date {date}')
#     get_webbpsf(filt, FIELD, ANGLE, output=DIR_PSFS, date=date, pixscl=PIXEL_SCALE)


def renorm_psf(filt, field, dir=DIR_PSFS, pixscl=PIXEL_SCALE, fov=PSF_FOV):
    psfmodel = fits.getdata(os.path.join(dir, f'{filt.lower()}_psf_unmatched.fits'))

    # encircled = {} # rounded to nearest 100nm, see hst docs
    # encircled['F105W'] = 0.975
    # encircled['F125W'] = 0.969
    # encircled['F140W'] = 0.967
    # encircled['F160W'] = 0.967
    # encircled['F435W'] = 0.989
    # encircled['F606W'] = 0.980
    # encircled['F814W'] = 0.976

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

    # and save
    newhdu = fits.PrimaryHDU(psfmodel)
    newhdu.writeto(os.path.join(DIR_PSFS, f'psf_{field}_{filt}_{fov}arcsec.fits'), overwrite=True)

# for filt in FILTERS:
#     filt = filt.upper()
#     if filt in USE_FILTERS: continue
#     print(f'Normalizing ePSF for {filt}...')
#     renorm_psf(filt, FIELD)
