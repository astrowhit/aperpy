import os, sys
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
import time

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_OUTPUT, DIR_IMAGES, DIR_KERNELS, DIR_OUTPUT, FILTERS

KERNEL = sys.argv[2]
SCI_FILENAMES = list(glob.glob(DIR_OUTPUT+'/*_sci_skysubvar.fits*'))

for filename in SCI_FILENAMES:
    if f'sci_skysubvar' not in filename: continue
    # if not (('f410m' in FILENAME) or ('f444w' in FILENAME)): continue
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
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        weight = hdul_wht[0].data

        print(np.shape(hdul[0].data))
        hdul[0].data = convolve_fft(hdul[0].data, kernel, allow_huge=True)
        hdul[0].data[weight==0] = 0.

        err = np.where(weight==0, 0, 1/np.sqrt(weight))
        err_conv = convolve_fft(err, kernel, allow_huge=True)
        hdul_wht[0].data = np.where(err_conv==0, 0, 1./(err**2))
        hdul_wht[0].data[weight==0] = 0.
        print(f'Finished in {time.time()-tstart:2.2f}s')

    hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_skysubvar.fits', f'_skysubvar_{KERNEL}-matched.fits'), overwrite=True)
    hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci_skysubvar.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
