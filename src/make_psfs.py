# This is just a wrapper around the webb_tool
from webb_tools import get_psf, get_date
from astropy.io import fits
import numpy as np
import os, sys

PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import WEBB_FILTERS, FIELD, ANGLE, DIR_PSFS, PIXEL_SCALE, PSF_FOV, HST_FILTERS, USE_NEAREST_DATE, FILTERS


if USE_NEAREST_DATE:
    date = get_date()
else:
    date = None

# Default behavior generates a 10" FOV PSF and clips down to 4" FOV; 0.04 "/px
for filt in FILTERS:
    filt = filt.upper()
    if filt not in WEBB_FILTERS: continue
    print(f'Fetching WebbPSF for {filt} at PA {ANGLE}deg...')
    get_psf(filt, FIELD, ANGLE, output=DIR_PSFS, date=date)


def renorm_hst_psf(filt, field, dir=DIR_PSFS, pixscl=PIXEL_SCALE, fov=PSF_FOV):
    psfmodel = fits.getdata(os.path.join(dir, f'{filt.lower()}_psf_unmatched.fits'))

    encircled = {} # rounded to nearest 100nm, see hst docs, 2"
    encircled['F105W'] = 0.975
    encircled['F125W'] = 0.969
    encircled['F140W'] = 0.967
    encircled['F160W'] = 0.967
    encircled['F435W'] = 0.989
    encircled['F606W'] = 0.980
    encircled['F814W'] = 0.976

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(psfmodel)
    Y, X = np.ogrid[:h, :w]
    r = fov / 2. / pixscl
    center = [w/2., h/2.]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    psfmodel /= np.sum(psfmodel[dist_from_center < r])
    psfmodel *= encircled[filt] # to get the missing flux accounted for

    # and save
    newhdu = fits.PrimaryHDU(psfmodel)
    newhdu.writeto(os.path.join(DIR_PSFS, f'psf_{field}_{filt}_{fov}arcsec.fits'), overwrite=True)

for filt in HST_FILTERS:
    filt = filt.upper()
    if filt not in HST_FILTERS: continue
    print(f'Normalizing HST PSF for {filt}...')
    renorm_hst_psf(filt, FIELD)
