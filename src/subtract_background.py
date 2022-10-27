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

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_IMAGES, DIR_OUTPUT, BACKTYPE, BACKPARAMS, \
            FILTERS, MED_CENTER, MED_SIZE, FILTER_SIZE, PIXEL_SCALE, IS_CLUSTER

SCI_FILENAMES = list(glob.glob(DIR_IMAGES+'/*_sci.fits*'))
WHT_FILENAMES = list(glob.glob(DIR_IMAGES+'/*_wht.fits*'))

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
        print(f'    Cutting out cluster region ({MED_CENTER}, {MED_SIZE})')
        subimg = Cutout2D(img, MED_CENTER, MED_SIZE, wcs=WCS(head))
        subwht = Cutout2D(wht, MED_CENTER, MED_SIZE, wcs=WCS(head))
        print('    Building median filtered cluster image for subtraction...')
        start = time.time()
        subimg_filt = signal.medfilt2d(subimg.data, kernel_size=int(FILTER_SIZE / PIXEL_SCALE))
        subimg_filt[subwht.data<=0.] = 0.
        back[subimg.slices_original[0], subimg.slices_original[1]] = subimg_filt
        print(f'    Subtracted median filter over cluster window ({time.time() - start:2.1f}s)')

    if BACKTYPE == 'var':
        for key in BACKPARAMS:
            head[f'BACK_{key}'] = BACKPARAMS[key]
        outbfname = filename.replace('.fits', f'_sky{BACKTYPE}.fits').replace(DIR_IMAGES, DIR_OUTPUT)
        fits.PrimaryHDU(data=back.astype(np.float32), header=head).writeto(outbfname, overwrite=True)

    outfname = filename.replace('.fits', f'_skysub{BACKTYPE}.fits').replace(DIR_IMAGES, DIR_OUTPUT)
    fits.PrimaryHDU(data=(img-back).astype(np.float32), header=head).writeto(outfname, overwrite=True)
    print(f'Written to {outfname}')
