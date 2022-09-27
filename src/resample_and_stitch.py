

import sys, os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
import numpy as np

FILT = None #sys.argv[1]

WORKING_DIR = '/Volumes/External1/Projects/Current/CEERS/data/external/egs-grizli-v4'

FILENAMES = list(glob.glob(WORKING_DIR+'/*'))

SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']

FILTERS = ['F115W', 'F150W',
         'F200W', 'F277W', 'F356W', 'F410M', 'F444W']
FILTERS += ['F606W', 'F606WU', 'F814W', 'F105W', 'F125W', 'F140W', 'F160W',
            'F435W']

FILTERS = ['F150W'] #'F115W', 'F200W']

for FILT in FILTERS:
    images = {}
    for region in ('ne', 'sw'):
        for itype in ('sci', 'wht'):
            # get file
            pattern = f'ceers-{region}-grizli-v4.0-{FILT.lower()}'
            print(pattern)
            tryout = os.path.join(WORKING_DIR, pattern)
            for fn in FILENAMES:
                if fn.startswith(tryout) & fn.endswith(f'{itype}.fits.gz'):
                    save_fn = fn
                    break
            hdul = fits.open(fn)
            header = hdul[0].header
            wcs = WCS(header)
            img = hdul[0].data

            print(pattern, itype, np.shape(img))
            print(save_fn)
            # print(wcs)

            if FILT in SW_FILTERS:
                # block reduce from 20mas to 40mas
                if itype == 'sci': newsub = block_reduce(img, 2)
                # # if itype == 'sci':
                # #     newsub /= 2**2
                if itype == 'wht':
                    newsub = 1. / block_reduce(1./img, 2)
            else:
                newsub = img

            images[itype+'_'+region] = newsub

    # stitch together
    for itype in ('sci', 'wht'):
        sizes = np.shape(images[itype+'_ne']), np.shape(images[itype+'_sw'])
        target_size = (sizes[0][0], sizes[0][1] + sizes[1][1])
        print(sizes, target_size)

        newimg = np.zeros(target_size)
        newimg[0:target_size[0], 0:sizes[0][1]] = images[itype+'_ne']
        newimg[0:target_size[0], sizes[0][1]:target_size[1]] = images[itype+'_sw']

        newimg = newimg.astype(np.float32)

        # update CD matrix for 40 mas + CRPIX
        header['CRPIX1'] = 12288
        header['CRPIX2'] = 4096
        header['CRVAL1'] = 214.9140403
        header['CRVAL2'] = 52.9036667
        header['CD1_1']   = -7.1420845520725E-06 #/ Coordinate transformation matrix element
        header['CD1_2']   = -8.5116049235439E-06 #/ Coordinate transformation matrix element
        header['CD2_1']   = -8.5116049235439E-06 #/ Coordinate transformation matrix element
        header['CD2_2']   =  7.1420845520726E-06

        # write out
        outpath = save_fn.replace('sw', 'full').replace('.fits.gz', '.fits.gz')
        if itype not in outpath:
            if itype == 'wht':
                outpath = outpath.replace('sci', 'wht')
            elif itype == 'sci':
                outpath = outpath.replace('wht', 'sci')
        hdul = fits.PrimaryHDU(data=newimg, header=header)
        print(outpath)
        print()
        hdul.writeto(outpath, overwrite=True)
 