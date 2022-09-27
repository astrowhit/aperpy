from typing import OrderedDict
import os
from astropy.io import fits
import numpy as np
from scipy.stats import chi
from astropy.stats import sigma_clipped_stats
import sep
from webb_tools import compute_background

DIR_IMAGES = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
DIR_WEIGHTS = '/Volumes/External1/Projects/Current/CEERS/data/external/egs-grizli-v4/'
DIR_OUTPUT = '/Volumes/External1/Projects/Current/CEERS/data/intermediate/v4/'
images = OrderedDict()
# images['f115w'] = os.path.join(DIR_IMAGES, 'ceers-grizli-v4-f115w-clear_drc_sci_skysubvar.fits.gz')
images['f150w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f150w-clear_drc_sci_skysubvar.fits.gz')
images['f200w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f200w-clear_drc_sci_skysubvar.fits.gz')
# images['f277w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f277w-clear_drc_sci_skysubvar.fits.gz')
images['f356w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f356w-clear_drc_sci_skysubvar.fits.gz')
images['f444w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f444w-clear_drc_sci_skysubvar.fits.gz')

WHT_REPLACE = ('sci_skysubvar', 'wht')
DIRWHT_REPLACE = (DIR_IMAGES, DIR_WEIGHTS)
BACKPARAMS = dict(bw=64, bh=64, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'NONE'

HEADEXT = 1

blue = ('f150w', 'f200w')
red = ('f356w', 'f444w')

# chi-mean, noise-equalized, stack
def chi_mean(bands, outname):
    # sum (S / N) -- but why add sigma linearly
    print(f'Building chi-mean image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = images[band]
        fn_wht = images[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, HEADEXT)
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            raw_img = (raw_img - back)**2 * fits.getdata(fn_wht)
            img = raw_img
            n = (raw_img != 0).astype(int)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            raw_img = (raw_img - back)**2 * fits.getdata(fn_wht)
            img += raw_img
            n += (raw_img != 0).astype(int)
            del raw_img

    mu = np.zeros_like(img)
    for ni in np.unique(n):
        mu[n == ni] = chi.stats(df=ni, moments='m')
    img = ( np.sqrt(img) - mu ) / np.sqrt( n - mu**2 )
    
    fits.ImageHDU(data=img.astype(np.float32), header=head).writeto(f'{outname}_chimean.fits.gz', overwrite=True)
    
# optimum average, so "noise equalized"
def opt_avg(bands, outname):
    # SUM( X * WHT) / SUM(WHT)
    print(f'Building optimal-average image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = images[band]
        fn_wht = images[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, HEADEXT)
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img - back) * wht
            top = raw_img
            bot = wht
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img - back) * wht
            top += raw_img
            bot += wht
            del raw_img

    optavg = top / bot
    opterr = np.sqrt(np.where(bot<=0, 0., 1. / bot))
    comb = optavg / opterr # signal / noise
    fits.ImageHDU(data=optavg.astype(np.float32), header=head).writeto(f'{outname}_optavg.fits.gz', overwrite=True)
    fits.ImageHDU(data=opterr.astype(np.float32), header=head).writeto(f'{outname}_opterr.fits.gz', overwrite=True)
    fits.ImageHDU(data=comb.astype(np.float32), header=head).writeto(f'{outname}_noise-equal.fits.gz', overwrite=True)


def sumstack(bands, outname):
    print(f'Building simple stack image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = images[band]
        fn_wht = images[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, HEADEXT)
            wht = fits.getdata(fn_wht)
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            img = raw_img - back
            wht = fits.getdata(fn_wht)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            back = compute_background(raw_img, mask=None, BACKTYPE=BACKTYPE, BACKPARAMS=BACKPARAMS)
            img += raw_img - back
            wht += fits.getdata(fn_wht)
            del raw_img
    
    fits.ImageHDU(data=img.astype(np.float32), header=head).writeto(f'{outname}_sumstack_sci.fits.gz', overwrite=True)
    fits.ImageHDU(data=wht.astype(np.float32), header=head).writeto(f'{outname}_sumstack_wht.fits.gz', overwrite=True)


for func in (opt_avg,):
    func(blue, os.path.join(DIR_OUTPUT, 'SW_'+'-'.join(blue)))
    # func(red, os.path.join(DIR_OUTPUT,'LW_'+'-'.join(red)))
