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

if USE_FFT_CONV:
    convolve_func = convolve_fft
    convolve_kwargs = {'allow_huge': True}
else:
    convolve_func = convolve
    convolve_kwargs = {}

for filename in SCI_FILENAMES:
    outsciname = filename.replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits')
    outwhtname=filename.replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits')
    if os.path.exists(outsciname) and os.path.exists(outwhtname):
        print(outsciname, outwhtname)
        print(f'Convolved images exist, I will not overwrite')
        continue

    haveit = False
    while not haveit:
        for band in FILTERS:
            if band in filename:
                haveit = True
                break
    if not haveit:
        print(f'Valid band not found in filename {filename}! Check your config for requested bands!')
        sys.exit()
    fn_weight = filename.replace(DIR_OUTPUT, DIR_IMAGES).replace(WHT_REPLACE[0], WHT_REPLACE[1])
    print(band)
    print('  science image: ', filename)
    print('  weight image: ', fn_weight)
    hdul = fits.open(filename)
    hdul_wht = fits.open(fn_weight)

    if band != KERNEL:
        print(f'  PSF-matching sci {band} to {KERNEL}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{band}_kernel.fits')
        print('  using kernel ', fn_kernel.split('/')[-1])
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        weight = hdul_wht[0].data

        if not os.path.exists(outsciname):
            print('Running science image convolution...')
            hdul[0].data = convolve_func(hdul[0].data, kernel, **convolve_kwargs).astype(np.float32)
            print('convolved...')
            hdul[0].data[weight==0] = 0.
            hdul.writeto(outsciname, overwrite=True)
            print('Wrote file to ', outsciname)
            hdul.close()
        else:
            print(outsciname)
            print(f'{band.upper()} convolved science image exists, I will not overwrite')

        if not os.path.exists(outwhtname):
            print('Running weight image convolution...')
            err = np.where(weight==0, 0, 1/np.sqrt(weight))
            err_conv = convolve_func(err, kernel, **convolve_kwargs).astype(np.float32)
            hdul_wht[0].data = np.where(err_conv==0, 0, 1./(err_conv**2))
            hdul_wht[0].data[weight==0] = 0.
            hdul_wht.writeto(outwhtname, overwrite=True)
            print('Wrote weight file to ', outwhtname)
        else:
            print(outwhtname)
            print(f'{band.upper()} convolved weight image exists, I will not overwrite')


        print(f'Finished in {time.time()-tstart:2.2f}s')

    else:
        hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits'), overwrite=True)
        hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
