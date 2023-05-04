

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

# print(DIR_IMAGES)
SCI_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_sci.fits*'))
# print(SCI_FILENAMES)

BORROW_HEADER = fits.getheader(BORROW_HEADER_FILE)

for filename in SCI_FILENAMES:
    # if 'f090w' not in filename: continue
    # print(filename)

    is_ok = False
    for band in SW_FILTERS:
        if band.lower() in filename:
            is_ok = True
    if not is_ok:
        continue


    # if 'f444w-matched' not in filename: continue

    print(filename.split('/')[-1])
    # for region in ('ne', 'sw'):
    #     # get file
    sci, header = fits.getdata(filename), fits.getheader(filename)
    wht = fits.getdata(filename.replace(WHT_REPLACE[0], WHT_REPLACE[1]))

    wcs = WCS(header)
    # blocked_wht = block_reduce(wht, 2, func=np.sum) / 4**2
    # blocked_sci = block_reduce(sci*wht, 2, func=np.sum) / blocked_wht / 4

    # if filt in SW_FILTERS:
        # block reduce from 20mas to 40mas
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
        # print(header[coln], BORROW_HEADER[coln])
        header[coln] = BORROW_HEADER[coln]
        # print(header[coln])

    if 'bcgs' in filename: # HACK to keep things working smoothly for uncover...
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_bcgs_sci', '_block40_wht'))
        print(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
    else:
        # print(filename.replace('_sci', '_block40_sci'))
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_sci', '_block40_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_sci', '_block40_wht'))
        print(filename.replace('_sci', '_block40_sci'))

    # # stitch together
    # for itype in ('sci', 'wht'):
    #     sizes = np.shape(images[itype+'_ne']), np.shape(images[itype+'_sw'])
    #     target_size = (sizes[0][0], sizes[0][1] + sizes[1][1])
    #     print(sizes, target_size)

    #     newimg = np.zeros(target_size)
    #     newimg[0:target_size[0], 0:sizes[0][1]] = images[itype+'_ne']
    #     newimg[0:target_size[0], sizes[0][1]:target_size[1]] = images[itype+'_sw']

    #     newimg = newimg.astype(np.float32)

    #     # update CD matrix for 40 mas + CRPIX
    #     header['CRPIX1'] = 12288
    #     header['CRPIX2'] = 4096
    #     header['CRVAL1'] = 214.9140403
    #     header['CRVAL2'] = 52.9036667
    #     header['CD1_1']   = -7.1420845520725E-06 #/ Coordinate transformation matrix element
    #     header['CD1_2']   = -8.5116049235439E-06 #/ Coordinate transformation matrix element
    #     header['CD2_1']   = -8.5116049235439E-06 #/ Coordinate transformation matrix element
    #     header['CD2_2']   =  7.1420845520726E-06

    #     # write out
    #     outpath = save_fn.replace('sw', 'full')
    #     if itype not in outpath:
    #         if itype == 'wht':
    #             outpath = outpath.replace('sci', 'wht')

    #         elif itype == 'sci':
    #             outpath = outpath.replace('wht', 'sci')
    #     hdul = fits.PrimaryHDU(data=newimg, header=header)
    #     print(outpath)
    #     print()
    #     hdul.writeto(outpath, overwrite=True)
