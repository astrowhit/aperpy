import os
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
import time

TARGET_BAND = 'f444w'
DIR_IMAGES = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
DIR_KERNELS = f'/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/{TARGET_BAND}_matched_psf_regularization/'
DIR_WEIGHTS = '/Volumes/External1/Projects/Current/CEERS/data/external/egs-grizli-v4/'
DIR_OUTPUT = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
IMG_EXT = 1

for FILENAME in glob.glob(os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-*_sci_skysubvar.fits.gz')):

    band = FILENAME.split('ceers-full-grizli-v4.0-')[1][:5]
    if not (('f150w' in FILENAME) or ('f115w' in FILENAME)or ('f200w' in FILENAME)): continue
    hdul = fits.open(FILENAME)
    if band != TARGET_BAND:
        print(f'PSF-matching {band} to {TARGET_BAND}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{band}_kernel.fits')
        kernel = fits.getdata(fn_kernel)

        fn_weight = FILENAME.replace(DIR_IMAGES, DIR_WEIGHTS).replace('sci_skysubvar', 'wht')
        weight = fits.getdata(fn_weight)

        hdul[IMG_EXT].data = convolve_fft(hdul[IMG_EXT].data, kernel, allow_huge=True)
        hdul[IMG_EXT].data[weight==0] = 0.
        print(f'Finished in {time.time()-tstart:2.2f}s')

    hdul.writeto(FILENAME.replace(DIR_IMAGES, DIR_OUTPUT).replace('_skysubvar.fits.gz', f'_skysubvar_{TARGET_BAND}-matched.fits.gz'), overwrite=True)
    
    


