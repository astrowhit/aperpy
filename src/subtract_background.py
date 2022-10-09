from typing import OrderedDict
import os
from astropy.io import fits
import numpy as np
from scipy.stats import chi
from astropy.stats import sigma_clipped_stats
import glob
from webb_tools import compute_background

DIR_IMAGES = '/Volumes/External1/Projects/Current/CEERS/data/external/egs-grizli-v4/'
DIR_OUTPUT = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'

BACKPARAMS = dict(bw=32, bh=32, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'var'

# 150, 200, 356, 444
for FILENAME in glob.glob(os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-*_sci.fits.gz')):
    fn_sci = os.path.join(DIR_IMAGES, FILENAME)
    fn_wht = os.path.join(DIR_IMAGES, FILENAME.replace('sci', 'wht'))
    print(fn_sci, fn_wht)

    head = fits.getheader(fn_sci)
    img = fits.getdata(fn_sci)
    wht = fits.getdata(fn_wht)
    mask = None

    back = compute_background(img, mask, BACKTYPE.upper(), BACKPARAMS)
    back[wht<=0.] = 0.

    if BACKTYPE == 'var':
        for key in BACKPARAMS:
            head[f'BACK_{key}'] = BACKPARAMS[key]
        outbfname = FILENAME.replace('.fits.gz', f'_sky{BACKTYPE}.fits.gz').replace(DIR_IMAGES, DIR_OUTPUT)
        fits.ImageHDU(data=back.astype(np.float32), header=head).writeto(os.path.join(DIR_OUTPUT, outbfname), overwrite=True)

    outfname = FILENAME.replace('.fits.gz', f'_skysub{BACKTYPE}.fits.gz').replace(DIR_IMAGES, DIR_OUTPUT)
    fits.ImageHDU(data=(img-back).astype(np.float32), header=head).writeto(os.path.join(DIR_OUTPUT, outfname), overwrite=True)
    print(f'Written to {outfname}')