

import sys, os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
import numpy as np

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_IMAGES, SW_FILTERS, WEBB_FILTERS, WHT_REPLACE, BORROW_HEADER_FILE

SCI_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_sci.fits*'))

BORROW_HEADER = fits.getheader(BORROW_HEADER_FILE)

for filename in SCI_FILENAMES:

    is_ok = False
    for band in SW_FILTERS:
        if band.lower() in filename:
            is_ok = True
    if not is_ok:
        continue

    print(filename.split('/')[-1])
    sci, header = fits.getdata(filename), fits.getheader(filename)
    wht = fits.getdata(filename.replace(WHT_REPLACE[0], WHT_REPLACE[1]))

    wcs = WCS(header)

    block_wht = block_reduce(wht, 2, func=np.sum) / 4**2
    block_sci = block_reduce(sci*wht, 2, func=np.sum) / block_wht / 4

    cols = ['CRPIX1',
    'CRPIX2',
    'CD1_1',
    'CD2_2',
    'CDELT1',
    'CDELT2',
    'CUNIT1',
    'CUNIT2',
    'CTYPE1',
    'CTYPE2',
    'CRVAL1',
    'CRVAL2',
    'LONPOLE',
    'LATPOLE']

    for coln in cols:
        header[coln] = BORROW_HEADER[coln]

    if 'bcgs' in filename: # HACK to keep things working smoothly for uncover...
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_bcgs_sci', '_block40_wht'))
        print(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
    else:
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_sci', '_block40_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_sci', '_block40_wht'))
        print(filename.replace('_sci', '_block40_sci'))

