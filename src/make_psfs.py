# This is just a wrapper around the webb_tool
from webb_tools import get_psf, get_date
from astropy.io import fits
import numpy as np
import os, sys

PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import WEBB_FILTERS, FIELD, ANGLE, DIR_PSFS, PIXEL_SCALE, PSF_FOV, HST_FILTERS, USE_DATE, USE_NEAREST_DATE, FILTERS, USE_FILTERS


if USE_NEAREST_DATE:
    date = get_date()
elif USE_DATE is not None:
    date = USE_DATE
else:
    date = None

# Default behavior generates a 10" FOV PSF and clips down to 4" FOV; 0.04 "/px
for filt in FILTERS:
    filt = filt.upper()
    if filt not in USE_FILTERS: continue
    print(f'Fetching WebbPSF for {filt} at PA {ANGLE}deg for date {date}')
    get_psf(filt, FIELD, ANGLE, output=DIR_PSFS, date=date, pixscl=PIXEL_SCALE)


def renorm_psf(filt, field, dir=DIR_PSFS, pixscl=PIXEL_SCALE, fov=PSF_FOV):
    psfmodel = fits.getdata(os.path.join(dir, f'{filt.lower()}_psf_unmatched.fits'))

    # encircled = {} # rounded to nearest 100nm, see hst docs
    # encircled['F105W'] = 0.975
    # encircled['F125W'] = 0.969
    # encircled['F140W'] = 0.967
    # encircled['F160W'] = 0.967
    # encircled['F435W'] = 0.989
    # encircled['F606W'] = 0.980
    # encircled['F814W'] = 0.976

    # Encircled energy for WFC3 IR within 2" radius, ACS Optical, and UVIS from HST docs
    encircled = {}
    encircled['F225W'] = 0.993
    encircled['F275W'] = 0.984
    encircled['F336W'] = 0.9905
    encircled['F435W'] = 0.979
    encircled['F606W'] = 0.975
    encircled['F775W'] = 0.972
    encircled['F814W'] = 0.972
    encircled['F850LP'] = 0.970
    encircled['F098M'] = 0.974
    encircled['F105W'] = 0.973
    encircled['F125W'] = 0.969
    encircled['F140W'] = 0.967
    encircled['F160W'] = 0.966
    encircled['F090W'] = 0.9837
    encircled['F115W'] = 0.9822
    encircled['F150W'] = 0.9804
    encircled['F200W'] = 0.9767
    encircled['F277W'] = 0.9691
    encircled['F356W'] = 0.9618
    encircled['F410M'] = 0.9568
    encircled['F444W'] = 0.9546

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

for filt in FILTERS:
    filt = filt.upper()
    if filt in USE_FILTERS: continue
    print(f'Normalizing ePSF for {filt}...')
    renorm_psf(filt, FIELD)
