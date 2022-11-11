import os, sys
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
import time

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_OUTPUT, DIR_IMAGES, DIR_KERNELS, DIR_OUTPUT, FILTERS, USE_FFT_CONV

KERNEL = sys.argv[2]
SCI_FILENAMES = list(glob.glob(DIR_OUTPUT+'/*_sci_skysubvar.fits*'))

if USE_FFT_CONV:
    convolve_func = convolve_fft
    convolve_kwargs = {'allow_huge': True}
else:
    convolve_func = convolve
    convolve_kwargs = {}

for filename in SCI_FILENAMES:
    if os.path.exists(filename.replace('.fits.gz', '_f444w-matched.fits.gz')): continue
    if f'sci_skysubvar' not in filename: continue
    for band in FILTERS:
        if band in filename:
            break
    # if not ('f444w' in filename): continue
    hdul = fits.open(filename)
    fn_weight = filename.replace(DIR_OUTPUT, DIR_IMAGES).replace(f'sci_skysubvar', 'wht')
    hdul_wht = fits.open(fn_weight)

    if band != KERNEL:
        print(f'PSF-matching sci {band} to {KERNEL}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{KERNEL}_matched_psfs/{band}_kernel.fits')
        print(fn_kernel)
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        weight = hdul_wht[0].data

        print(np.shape(hdul[0].data))
        hdul[0].data = convolve_func(hdul[0].data, kernel, **convolve_kwargs).astype(np.float32)
        hdul[0].data[weight==0] = 0.
        hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_skysubvar.fits', f'_skysubvar_{KERNEL}-matched.fits'), overwrite=True)

        err = np.where(weight==0, 0, 1/np.sqrt(weight))
        err_conv = convolve_func(err, kernel, **convolve_kwargs).astype(np.float32)
        hdul_wht[0].data = np.where(err_conv==0, 0, 1./(err**2))
        hdul_wht[0].data[weight==0] = 0.
        hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci_skysubvar.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
        print(f'Finished in {time.time()-tstart:2.2f}s')

    else:
        hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_skysubvar.fits', f'_skysubvar_{KERNEL}-matched.fits'), overwrite=True)
        hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci_skysubvar.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
