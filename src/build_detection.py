import os
from astropy.io import fits
import numpy as np
from scipy.stats import chi

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_CATALOGS, DETECTION_GROUPS, DETECTION_IMAGES, DET_TYPE, DIRWHT_REPLACE, IS_COMPRESSED, WHT_REPLACE

# chi-mean, noise-equalized, stack
def chi_mean(bands, outname, science_fnames, weight_fnames, is_compressed=True):
    # sum (S / N) -- but why add sigma linearly
    print(f'Building chi-mean image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            raw_img = fits.getdata(fn_sci)
            raw_img = (raw_img)**2 * fits.getdata(fn_wht)
            img = raw_img
            n = (raw_img != 0).astype(int)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            raw_img = (raw_img)**2 * fits.getdata(fn_wht)
            img += raw_img
            n += (raw_img != 0).astype(int)
            del raw_img

    mu = np.zeros_like(img)
    for ni in np.unique(n):
        mu[n == ni] = chi.stats(df=ni, moments='m')
    img = ( np.sqrt(img) - mu ) / np.sqrt( n - mu**2 )

    chiout = f'{outname}_chimean.fits'
    if is_compressed:
        chiout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(chiout, overwrite=True)

# optimum average, so "noise equalized"
def noise_equalized(bands, outname, science_fnames, weight_fnames, is_compressed=True):
    # SUM( X * WHT) / SUM(WHT)
    print(f'Building noise equalized image from {bands}')
    if np.isscalar(bands):
        bands = [bands,]
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(fn_sci)
        print(fn_wht)
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            raw_img = fits.getdata(fn_sci)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img) * wht
            top = raw_img
            bot = wht
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img) * wht
            top += raw_img
            bot += wht
            del raw_img

    optavg = np.where(bot==0., 0., top / bot)
    opterr = np.sqrt(np.where(bot<=0, 0., 1. / bot))
    comb = optavg / opterr # signal / noise

    avgout = f'{outname}_optavg.fits'
    errout = f'{outname}_opterr.fits'
    neqout = f'{outname}_noise-equal.fits'
    if is_compressed:
        avgout += '.gz'
        errout += '.gz'
        neqout += '.gz'
    fits.PrimaryHDU(data=optavg.astype(np.float32), header=head).writeto(avgout, overwrite=True)
    fits.PrimaryHDU(data=opterr.astype(np.float32), header=head).writeto(errout, overwrite=True)
    fits.PrimaryHDU(data=comb.astype(np.float32), header=head).writeto(neqout, overwrite=True)


def sumstack(bands, outname, science_fnames, weight_fnames, is_compressed=True):
    print(f'Building simple stack image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            wht = fits.getdata(fn_wht)
            raw_img = fits.getdata(fn_sci)
            img = raw_img
            wht = fits.getdata(fn_wht)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            img += raw_img
            wht += fits.getdata(fn_wht)
            del raw_img

    sciout = f'{outname}_sumstack_sci.fits'
    whtout = f'{outname}_sumstack_wht.fits'
    if is_compressed:
        sciout += '.gz'
        whtout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(sciout, overwrite=True)
    fits.PrimaryHDU(data=wht.astype(np.float32), header=head).writeto(whtout, overwrite=True)


if __name__ == "__main__":
    DET_NICKNAME = sys.argv[2]
    outpath = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    bands = DETECTION_GROUPS[DET_NICKNAME.split('_')[0]]
    print(bands)
    science_fnames = DETECTION_IMAGES
    weight_fnames = {}
    for band in bands:
        weight_fnames[band] = DETECTION_IMAGES[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])

    if DET_TYPE == 'noise-equal':
        noise_equalized(bands, os.path.join(outpath, f'{DET_NICKNAME}'),
                        science_fnames= science_fnames,
                        weight_fnames= weight_fnames, 
                        is_compressed=IS_COMPRESSED)
    else:
        sys.exit('Other detecton choices are deprecated! Edit code at your own risk...')