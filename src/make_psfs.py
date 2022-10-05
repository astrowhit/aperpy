# This is just a wrapper around the webb_tool
from FOOwebb_tools import get_psf
from astropy.io import fits
import numpy as np
import os

WEBB_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
WEBB_FILTERS += ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']
HST_FILTERS = ['F105W', 'F125W', 'F140W', 'F160W', 'F435W', 'F606W', 'F814W']

FIELD = 'ceers'
ANGLE = 0. # use to override quick 'field' PAs:
# {'ceers': 130.7889803307112, 'smacs': 144.6479834976019, 'glass': 251.2973235468314}

# Default behavior generates a 10" FOV PSF and clips down to 4" FOV; 0.04 "/px
# for filt in WEBB_FILTERS:
#     get_psf(filt, FIELD, ANGLE)


def renorm_hst_psf(filt, field, dir='/Volumes/External1/Projects/Current/CEERS/data/external/psf_jrw_v4/hst_psfs_v4', pixscl=0.04, fov=4):
    psfmodel = fits.getdata(os.path.join(dir, f'{filt.lower()}_psf.fits'))

    encircled = {} # rounded to nearest 100nm, see hst docs
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
    newhdu.writeto(f'psf_{field}_{filt}_{fov}arcsec.fits', overwrite=True)

for filt in HST_FILTERS:
    renorm_hst_psf(filt, FIELD)
