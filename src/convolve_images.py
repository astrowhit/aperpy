import os
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
import time

TARGET_BAND = 'f160w'
DIR_IMAGES = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
DIR_KERNELS = f'/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/{TARGET_BAND}_matched_psf_shapelets/'
DIR_WEIGHTS = '/Volumes/External1/Projects/Current/CEERS/data/external/egs-grizli-v4/'
DIR_OUTPUT = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
IMG_EXT = 1
WHT_EXT = 0

for FILENAME in glob.glob(os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-*_sci_skysubvar.fits.gz')):
    # if not (('f410m' in FILENAME) or ('f444w' in FILENAME)): continue
    band = FILENAME.split('ceers-full-grizli-v4.0-')[1][:5]
    # if not ('f444w' in FILENAME): continue
    hdul = fits.open(FILENAME)
    fn_weight = FILENAME.replace(DIR_IMAGES, DIR_WEIGHTS).replace('sci_skysubvar', 'wht')
    hdul_wht = fits.open(fn_weight)

    if band != TARGET_BAND:
        print(f'PSF-matching sci {band} to {TARGET_BAND}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{band}_kernel.fits')
        # fn_kernel = '/Volumes/External1/Projects/Current/CEERS/data/external/ceers_kernels_v2/F444W.fits'
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        weight = hdul_wht[WHT_EXT].data

        hdul[IMG_EXT].data = convolve_fft(hdul[IMG_EXT].data, kernel, allow_huge=True)
        hdul[IMG_EXT].data[weight==0] = 0.

        err = np.where(weight==0, 0, 1/np.sqrt(weight))
        err_conv = convolve_fft(err, kernel, allow_huge=True) #, allow_huge=True)
        hdul_wht[WHT_EXT].data = np.where(err_conv==0, 0, 1./(err**2))
        hdul_wht[WHT_EXT].data[weight==0] = 0.
        print(f'Finished in {time.time()-tstart:2.2f}s')

    hdul.writeto(FILENAME.replace(DIR_IMAGES, DIR_OUTPUT).replace('_skysubvar.fits.gz', f'_skysubvar_{TARGET_BAND}-matched.fits.gz'), overwrite=True)
    hdul_wht.writeto(FILENAME.replace(DIR_IMAGES, DIR_OUTPUT).replace('_sci_skysubvar.fits.gz', f'_wht_{TARGET_BAND}-matched.fits.gz'), overwrite=True)


