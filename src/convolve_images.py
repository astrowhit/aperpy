import os, sys
from astropy.io import fits
import numpy as np
import glob
from astropy.convolution import convolve, convolve_fft
from scipy.signal import fftconvolve
import time

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_OUTPUT, DIR_IMAGES, DIR_KERNELS, DIR_OUTPUT, FILTERS, USE_FFT_CONV, WHT_REPLACE, SKYEXT, IS_COMPRESSED

KERNEL = sys.argv[2]
if SKYEXT == '':
    SCI_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_sci{SKYEXT}.fits*'))
else:
    SCI_FILENAMES = list(glob.glob(DIR_OUTPUT+f'/*_sci{SKYEXT}.fits*'))

if USE_FFT_CONV:
    # convolve_func = convolve_fft
    # convolve_kwargs = {'allow_huge': True}
    convolve_func = fftconvolve
    convolve_kwargs = {'mode': 'same'}
else:
    convolve_func = convolve
    convolve_kwargs = {}

for filename in SCI_FILENAMES:
    outsciname = filename.replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits')
    outwhtname = filename.replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits')
    if SKYEXT == '':
        outsciname = outsciname.replace(DIR_IMAGES, DIR_OUTPUT)
        outwhtname = outwhtname.replace(DIR_IMAGES, DIR_OUTPUT)
    
    if os.path.exists(outsciname) and os.path.exists(outwhtname):
        print(outsciname, outwhtname)
        print(f'Convolved images exist, I will not overwrite')
        continue

    fn_weight = filename.replace(DIR_OUTPUT, DIR_IMAGES).replace(WHT_REPLACE[0], WHT_REPLACE[1])

    haveit = False
    # while not haveit:
    for band in FILTERS:
        if band in filename:
            haveit = True
            break
    if not haveit:
        print(f'Valid band not found in filename {filename}! Check your config for requested bands!')
        sys.exit()

    print(band)
    print('  science image: ', filename)
    print('  weight image: ', fn_weight)
    
    if band != KERNEL:
        print(f'  PSF-matching sci {band} to {KERNEL}')
        tstart = time.time()
        fn_kernel = os.path.join(DIR_KERNELS, f'{band}_kernel.fits')
        print('  using kernel ', fn_kernel.split('/')[-1])
        kernel = fits.getdata(fn_kernel)
        kernel /= np.sum(kernel)

        if not os.path.exists(outsciname):
            hdul = fits.open(filename)
            print('Running science image convolution...')
            hdul[0].data = convolve_func(hdul[0].data, kernel, **convolve_kwargs).astype(np.float32)
            print('convolved...')
            hdul_wht = fits.open(fn_weight)
            hdul[0].data[hdul_wht[0].data==0] = 0.
            hdul_wht.close()
            hdul.writeto(outsciname, overwrite=True)
            print('Wrote file to ', outsciname)
            hdul.close()
        else:
            print(outsciname)
            print(f'{band.upper()} convolved science image exists, I will not overwrite')
        if not os.path.exists(outwhtname):
            hdul_wht = fits.open(fn_weight)
            weight = hdul_wht[0].data
            print('Running weight image convolution...')
            err = np.where(weight==0, 0, 1/np.sqrt(weight))
            del weight
            err_conv = convolve_func(err, kernel, **convolve_kwargs).astype(np.float32)
            hdul_wht[0].data = np.where(err_conv==0, 0, 1./(err_conv**2))
            hdul_wht[0].data[err==0] = 0.
            hdul_wht.writeto(outwhtname, overwrite=True)
            print('Wrote weight file to ', outwhtname)
            del err
            del err_conv
            hdul_wht.close()
        else:
            print(outwhtname)
            print(f'{band.upper()} convolved weight image exists, I will not overwrite')

        print(f'Finished in {time.time()-tstart:2.2f}s')

    else:
        if not os.path.exists(outsciname):
            hdul = fits.open(filename)
            hdul.writeto(outsciname, overwrite=True)
            # hdul.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'{SKYEXT}.fits', f'{SKYEXT}_{KERNEL}-matched.fits'), overwrite=True)
            hdul.close()
        if not os.path.exists(outwhtname):
            hdul_wht = fits.open(fn_weight)
            hdul_wht.writeto(outwhtname, overwrite=True)
            # hdul_wht.writeto(filename.replace(DIR_IMAGES, DIR_OUTPUT).replace(f'_sci{SKYEXT}.fits', f'_wht_{KERNEL}-matched.fits'), overwrite=True)
            hdul_wht.close()
