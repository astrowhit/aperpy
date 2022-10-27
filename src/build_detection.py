import os
from astropy.io import fits
import numpy as np
from scipy.stats import chi

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_CATALOGS, DETECTION_GROUPS, DETECTION_IMAGES, DET_TYPE, DIRWHT_REPLACE

WHT_REPLACE = ('sci_skysubvar', 'wht')

# chi-mean, noise-equalized, stack
def chi_mean(bands, outname):
    # sum (S / N) -- but why add sigma linearly
    print(f'Building chi-mean image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = DETECTION_IMAGES[band]
        fn_wht = DETECTION_IMAGES[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
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
    if '.gz' in fn_sci:
        chiout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(chiout, overwrite=True)

# optimum average, so "noise equalized"
def noise_equalized(bands, outname):
    # SUM( X * WHT) / SUM(WHT)
    print(f'Building noise equalized image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = DETECTION_IMAGES[band]
        fn_wht = DETECTION_IMAGES[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
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
    if '.gz' in fn_sci:
        avgout += '.gz'
        errout += '.gz'
        neqout += '.gz'
    fits.PrimaryHDU(data=optavg.astype(np.float32), header=head).writeto(avgout, overwrite=True)
    fits.PrimaryHDU(data=opterr.astype(np.float32), header=head).writeto(errout, overwrite=True)
    fits.PrimaryHDU(data=comb.astype(np.float32), header=head).writeto(neqout, overwrite=True)


def sumstack(bands, outname):
    print(f'Building simple stack image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = DETECTION_IMAGES[band]
        fn_wht = DETECTION_IMAGES[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
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
    if '.gz' in fn_sci:
        sciout += '.gz'
        whtout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(sciout, overwrite=True)
    fits.PrimaryHDU(data=wht.astype(np.float32), header=head).writeto(whtout, overwrite=True)

DET_NICKNAME = sys.argv[2]
outpath = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}')
if not os.path.exists(outpath):
    os.mkdir(outpath)
bands = DETECTION_GROUPS[DET_NICKNAME.split('_')[0]]
if DET_TYPE == 'noise-equal':
    noise_equalized(bands, os.path.join(outpath, f'{DET_NICKNAME}'))
