import os, sys
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
import time

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_OUTPUT, DIR_IMAGES, DIR_KERNELS, DIR_OUTPUT, FILTERS, USE_FFT_CONV, WHT_REPLACE, SKYEXT

KERNEL = sys.argv[2]
SCI_FILENAMES = list(glob.glob(DIR_OUTPUT+f'/*_sci{SKYEXT}.fits*'))
# KERNEL = 'f444w'
# DIR_KERNELS = '/Volumes/Weaver_2TB/Projects/Current/UNCOVER/notebooks/tests/SW_PSF/'
# SCI_FILENAMES = ['/Volumes/Weaver_2TB/Projects/Current/UNCOVER/notebooks/tests/SW_PSF/uncover_v6.0_abell2744clu_f115w_bcgs_sci.fits.gz']
# SKYEXT = ''

if USE_FFT_CONV:
    convolve_func = convolve_fft
    convolve_kwargs = {'allow_huge': True}
else:
    convolve_func = convolve
    convolve_kwargs = {}

for filename in SCI_FILENAMES:
    if os.path.exists(filename.replace('.fits', f'_{KERNEL}-matched.fits')): continue
    # if f'sci{SKYEXT}' not in filename: continue
    for band in FILTERS:
        if band in filename:
            break
    # if not ('f090w' in filename): continue
    fn_weight = filename.replace(DIR_OUTPUT, DIR_IMAGES).replace(WHT_REPLACE[0], WHT_REPLACE[1])
    print(band)
    print('  science image: ', filename)
    print('  weight image: ', fn_weight)
    # hdul = fits.open(filename)
    hdul_wht = fits.open(fn_weight)

    if band != KERNEL:
        print(f'  PSF-matching sci {band} to {KERNEL}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{KERNEL}_matched_psfs/{band}_kernel.fits')
        print('  using kernel ', fn_kernel.split('/')[-1])
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        weight = hdul_wht[0].data

        # # print(np.shape(hdul[0].data))
        # print('Running convolution...')
        # hdul[0].data = convolve_func(hdul[0].data, kernel, **convolve_kwargs).astype(np.float32)
        # hdul[0].data[weight==0] = 0.
        # outfilename = filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits')
        # hdul.writeto(outfilename, overwrite=True)
        # print('Wrote file to ', outfilename)

        err = np.where(weight==0, 0, 1/np.sqrt(weight))
        err_conv = convolve_func(err, kernel, **convolve_kwargs).astype(np.float32)
        hdul_wht[0].data = np.where(err_conv==0, 0, 1./(err_conv**2))
        hdul_wht[0].data[weight==0] = 0.
        hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
        print(f'Finished in {time.time()-tstart:2.2f}s')

    else:
        # hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits'), overwrite=True)
        hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
