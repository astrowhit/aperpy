

import sys, os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
import numpy as np

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_IMAGES, SW_FILTERS, WEBB_FILTERS, WHT_REPLACE

SW_HACK = '_f444w-matched'
print(DIR_IMAGES)
SCI_FILENAMES = list(glob.glob(DIR_IMAGES+f'/*_sci{SW_HACK}.fits*'))
print(SCI_FILENAMES)

for filename in SCI_FILENAMES:
    # if 'f090w' not in filename: continue
    print(filename)

    for band in WEBB_FILTERS:
        if band.lower() in filename:
            break
    if band.upper() not in SW_FILTERS:
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


    # update CD matrix for 40 mas + CRPIX
    header['CRPIX1']  =               6529.0 # / Pixel coordinate of reference point
    header['CRPIX2']  =               4570.0 # / Pixel coordinate of reference point
    header['CD1_1']   = -1.1111111111111E-05 # / Coordinate transformation matrix element
    header['CD2_2']   =  1.1111111111111E-05 # / Coordinate transformation matrix element
    header['CDELT1']  =                  1.0 # / [deg] Coordinate increment at reference point
    header['CDELT2']  =                  1.0 # / [deg] Coordinate increment at reference point
    header['CUNIT1']  = 'deg'                # / Units of coordinate increment and value
    header['CUNIT2']  = 'deg'                # / Units of coordinate increment and value
    header['CTYPE1']  = 'RA---TAN'           # / Right ascension, gnomonic projection
    header['CTYPE2']  = 'DEC--TAN'           # / Declination, gnomonic projection
    header['CRVAL1']  =               3.5875 # / [deg] Coordinate value at reference point
    header['CRVAL2']  =          -30.3966667 # / [deg] Coordinate value at reference point
    header['LONPOLE'] =                180.0 # / [deg] Native longitude of celestial pole
    header['LATPOLE'] =          -30.3966667 # / [deg] Native latitude of celestial pole

    if 'bcgs' in filename: # HACK to keep things working smoothly for uncover...
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_bcgs_sci', '_block40_bcgs_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_bcgs_sci', '_block40_wht'))
    else:
        # print(filename.replace('_sci', '_block40_sci'))
        fits.PrimaryHDU(block_sci, header=header).writeto(filename.replace('_sci', '_block40_sci'))
        fits.PrimaryHDU(block_wht, header=header).writeto(filename.replace('_sci', '_block40_wht'))

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
