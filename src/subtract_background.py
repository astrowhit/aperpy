from typing import OrderedDict
import os
from astropy.io import fits
import numpy as np
from scipy.stats import chi
from astropy.stats import sigma_clipped_stats
import glob
import time
from webb_tools import compute_background
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy import signal
from astropy.nddata import block_reduce, block_replicate
import numpy as np
from scipy.ndimage import zoom

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_IMAGES, DIR_OUTPUT, BACKTYPE, BACKPARAMS, \
            FILTERS, MED_CENTERS, MED_SIZE, FILTER_SIZE, PIXEL_SCALE, IS_CLUSTER, BLOCK_SIZE

SCI_FILENAMES = list(glob.glob(DIR_IMAGES+'/*_sci.fits*'))
WHT_FILENAMES = list(glob.glob(DIR_IMAGES+'/*_wht.fits*'))

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

# 150, 200, 356, 444
for filename in SCI_FILENAMES:
    if np.sum([(filt in filename) for filt in FILTERS]) == 0: continue
    fn_sci = filename
    fn_wht = filename.replace('sci', 'wht')
    print(fn_sci.split('/')[-1])

    head = fits.getheader(fn_sci)
    img = fits.getdata(fn_sci)
    wht = fits.getdata(fn_wht)
    mask = None

    back = compute_background(img, mask, BACKTYPE.upper(), BACKPARAMS)
    back[wht<=0.] = 0.

    if IS_CLUSTER:
        # replace with median filter on central window
        for MED_CENTER in MED_CENTERS:
            print(f'    Cutting out cluster region ({MED_CENTER}, {MED_SIZE})')
            subimg = Cutout2D(img, MED_CENTER, MED_SIZE, wcs=WCS(head))
            subwht = Cutout2D(wht, MED_CENTER, MED_SIZE, wcs=WCS(head))
            print(f'    Block summing cluster cutout ({BLOCK_SIZE} x {BLOCK_SIZE})')
            subwht_blocked = block_reduce(subwht.data, BLOCK_SIZE, func=np.sum) / (BLOCK_SIZE**2)**2
            subimg_blocked = block_reduce(subimg.data*subwht.data, BLOCK_SIZE, func=np.sum) / subwht_blocked / (BLOCK_SIZE**2)
            print(f'    Building median filtered cluster image ({FILTER_SIZE}\"; {round_up_to_odd(FILTER_SIZE / (PIXEL_SCALE * BLOCK_SIZE))}px)')
            start = time.time()
            subimg_filt = signal.medfilt2d(subimg_blocked, kernel_size=round_up_to_odd(FILTER_SIZE / (PIXEL_SCALE * BLOCK_SIZE)))
            subimg_filt[subwht_blocked<=0.] = 0.
            print('    Upsampling back to native pixel scale')
            # subimg_upscl = block_replicate(subimg_filt, BLOCK_SIZE, conserve_sum=True)
            subimg_upscl = zoom(subimg_filt, BLOCK_SIZE, order=1) / (BLOCK_SIZE**2)
            print(np.shape(subimg_upscl), np.shape(subimg_filt), np.shape(subwht.data))
            subimg_upscl[subwht.data<=0.] = 0.
            subimg_upscl[np.isnan(subimg_upscl)] = 0.
            back[subimg.slices_original[0], subimg.slices_original[1]] = subimg_upscl
            print(f'    Subtracted median filter over cluster window ({time.time() - start:2.1f}s)')

    if BACKTYPE == 'var':
        for key in BACKPARAMS:
            head[f'BACK_{key}'] = BACKPARAMS[key]
        outbfname = filename.replace('.fits', f'_sky{BACKTYPE}.fits').replace(DIR_IMAGES, DIR_OUTPUT)
        fits.PrimaryHDU(data=back.astype(np.float32), header=head).writeto(outbfname, overwrite=True)

    outfname = filename.replace('.fits', f'_skysub{BACKTYPE}.fits').replace(DIR_IMAGES, DIR_OUTPUT)
    fits.PrimaryHDU(data=(img-back).astype(np.float32), header=head).writeto(outfname, overwrite=True)
    print(f'Written to {outfname}')


        #     print(f'    Cutting out cluster region ({MED_CENTER}, {MED_SIZE})')
        # subimg = Cutout2D(img, MED_CENTER, MED_SIZE, wcs=WCS(head))
        # subwht = Cutout2D(wht, MED_CENTER, MED_SIZE, wcs=WCS(head))
        # print(f'    Block averaging cluster cutout ({BLOCK_SIZE}x)')
        # subwht_blocked = block_reduce(subwht.data, BLOCK_SIZE, func=np.sum) / (BLOCK_SIZE**2)**2
        # subimg_blocked = block_reduce(subimg.data*subwht.data, BLOCK_SIZE, func=np.sum) / subwht_blocked / (BLOCK_SIZE**2)
        # print(f'    Building median filtered cluster image ({FILTER_SIZE}\"; {round_up_to_odd(FILTER_SIZE / (PIXEL_SCALE * BLOCK_SIZE))}px)')
        # start = time.time()
        # subimg_filt = signal.medfilt2d(subimg_blocked, kernel_size=round_up_to_odd(FILTER_SIZE / (PIXEL_SCALE * BLOCK_SIZE)))
        # subimg_filt[subwht_blocked<=0.] = 0.
        # print('    Upsampling back to native pixel scale')
        # subimg_upscl = block_replicate(subimg_filt.data*subwht_blocked, BLOCK_SIZE, conserve_sum=False) * subwht.data * (BLOCK_SIZE**2)
        # back[subimg.slices_original[0], subimg.slices_original[1]] = subimg_upscl
        # print(f'    Subtracted median filter over cluster window ({time.time() - start:2.1f}s)')
