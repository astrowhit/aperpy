

import sys, os
import glob
from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy.nddata import block_reduce, block_replicate
import numpy as np

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_IMAGES, SW_FILTERS, BLOCK_WHT_REPLACE,\
                   BORROW_HEADER_FILE, PIXEL_SCALE, USE_EXPTIME

SCI_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_sci.fits*'))
BORROW_HEADER = fits.getheader(BORROW_HEADER_FILE)

if not os.path.exists(f'{DIR_IMAGES}raw/'):
    os.mkdir(f'{DIR_IMAGES}raw/')

for filename in SCI_FILENAMES:

    print(filename)
    is_ok = False
    for band in SW_FILTERS:
        if band.lower() in filename and 'block40' not in filename:
            is_ok = True
    if not is_ok:
        continue

    header = fits.getheader(filename)
    wcs = WCS(header)
    pcs = np.round(utils.proj_plane_pixel_scales(wcs)[0] * 3600, 2)
    if pcs == PIXEL_SCALE: continue

    # print(filename.split('/')[-1])
    sci = fits.getdata(filename)
    wht_filename = filename.replace(*BLOCK_WHT_REPLACE)
    wht = fits.getdata(wht_filename)

    print('Reading from...')
    print(filename)
    print(wht_filename)

    block_wht = block_reduce(wht, int(PIXEL_SCALE/pcs), func=np.sum) / 4**2
    block_sci = block_reduce(sci*wht, int(PIXEL_SCALE/pcs), func=np.sum) / block_wht / 4

    block_sci[~np.isfinite(block_sci)]=0.
    block_wht[~np.isfinite(block_wht)]=0.

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
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_bcgs_sci', '_block40_bcgs_sci'), overwrite=True)
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_bcgs_sci', '_block40_wht'), overwrite=True)
        print(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
    else:
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_sci', '_block40_sci'), overwrite=True)
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_sci', '_block40_wht'), overwrite=True)
        print(filename.replace('_sci', '_block40_sci'))

    os.rename(filename,filename.replace(DIR_IMAGES,DIR_IMAGES+'raw/'))
    os.rename(wht_filename,wht_filename.replace(DIR_IMAGES,DIR_IMAGES+'raw/'))


if USE_EXPTIME: 
    EXP_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_exp.fits*'))

    for filename in EXP_FILENAMES:
        if 'block40' in filename: continue

        header = fits.getheader(filename)
        wcs = WCS(header)
        pcs = np.round(utils.proj_plane_pixel_scales(wcs)[0] * 3600, 2)
        print(pcs)
        if pcs == PIXEL_SCALE: continue

        # print(filename.split('/')[-1])
        exp = fits.getdata(filename)

        print('Reading from...')
        print(filename)

        block_exp = block_replicate(exp, int(pcs/PIXEL_SCALE), conserve_sum=False)
        block_exp[~np.isfinite(block_exp)]=0.

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
            fits.PrimaryHDU(block_exp, header=header).writeto(filename.replace('_bcgs_exp', '_block40_bcgs_exp'), overwrite=True)
            print(filename.replace('_bcgs_exp', '_block40_bcgs_exp'))
        else:
            fits.PrimaryHDU(block_exp, header=header).writeto(filename.replace('_exp', '_block40_exp'), overwrite=True)
            print(filename.replace('_exp', '_block40_exp'))

        os.rename(filename,filename.replace(DIR_IMAGES,DIR_IMAGES+'raw/'))
