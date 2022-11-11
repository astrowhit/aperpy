import os
from typing import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord

### GENERAL
KERNELS = {}
KERNELS['f444w'] = 'regularization'
# KERNELS['f160w'] = 'shapelets'
USE_FFT_CONV = True # use FFT convolution

DETECTION_PARAMS = dict(
    thresh =  1.5,
    minarea = 9,
    kernelfwhm = 1.00170,
    deblend_nthresh = 16,
    deblend_cont = 0.00315,
    clean = True,
    clean_param = 1.66776,
    )

PHOT_APER = [0.16, 0.32, 0.48, 0.7 , 1.  , 2.] # diameter in arcsec
PHOT_AUTOPARAMS = 1.5, 2.5 # Kron-scaling radius, mimumum circular diameter (not used!)
PHOT_FLUXFRAC = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux
PHOT_KRONPARAM = 6.0 # SE hardcodes this as 6.0

PIXEL_SCALE = 0.04 # arcsec / px

REF_BAND = 'f444w'
APPLY_MWDUST = 'MEDIAN'
USE_FFT_CONV = True

SCI_APER = 0.7 # science aperture for maximum empty aperture size
MAKE_SCIREADY_ALL = True # make aperture corrected catalogs for all apertures

### DIRECTORIES
IS_COMPRESSED = True # outputs files as .gz
WORKING_DIR = '/Volumes/External1/Projects/Current/UNCOVER/data/vTest'
DIR_IMAGES = os.path.join(WORKING_DIR, 'external/grizli-v5/')
DIR_OUTPUT = os.path.join(WORKING_DIR, 'output/')
DIR_PSFS = os.path.join(WORKING_DIR, 'intermediate/PSF/')
DIR_KERNELS = os.path.join(WORKING_DIR, 'intermediate/kernels/') # Generally
DIR_CATALOGS = os.path.join(WORKING_DIR, 'catalogs/')

DIRWHT_REPLACE = (DIR_OUTPUT, DIR_IMAGES)
DIR_SFD = '~/Projects/Common/py_tools/sfddata-master'
ZSPEC = '/Volumes/External1/Projects/Current/UNCOVER/data/vTest/external/zspec_abell2744_all.fits'
ZCOL = 'z'
ZRA = 'RA'
ZDEC = 'DEC'
ZCONF = 'zconf', (3, 4)
MAX_SEP = 0.3 * u.arcsec

### MEDIAN FILTERING
IS_CLUSTER = True
FILTER_SIZE = 5 # arcsec
MED_CENTER = SkyCoord(3.587*u.deg, -30.40*u.deg)
MED_SIZE = 1.3*u.arcmin

### WEBBPSF GENERATION
PSF_FOV = 4 # arcsec
FIELD = 'uncover'
ANGLE = None # takes the default uncover PA
USE_NEAREST_DATE = False

### BACKGROUNDS
BACKPARAMS = dict(bw=32, bh=32, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'var'

### DETECTION COADD
DETECTION_GROUPS = {}
DETECTION_GROUPS['SW'] = ('f150w', 'f200w')
DETECTION_GROUPS['LW'] = ('f277w', 'f356w', 'f444w')

DET_TYPE = 'noise-equal'
DETECTION_NICKNAMES = []
for nickname in DETECTION_GROUPS:
    joined = '-'.join(DETECTION_GROUPS[nickname])
    DETECTION_NICKNAMES.append(f'{nickname}_{joined}')

import glob
DETECTION_IMAGES = OrderedDict()
for group in DETECTION_GROUPS:
    for filt in DETECTION_GROUPS[group]:
        for path in glob.glob(DIR_OUTPUT+'*'):
            if ('sci_skysubvar.fits.gz' in path) & (filt in path):
                DETECTION_IMAGES[filt] = path

### ZEROPOINTS
PHOT_ZP = OrderedDict()
## PHOT_ZP['f435w'] = 28.9
## PHOT_ZP['f606w'] = 28.9
## PHOT_ZP['f814w'] = 28.9
PHOT_ZP['f105w'] = 28.9
PHOT_ZP['f125w'] = 28.9
PHOT_ZP['f140w'] = 28.9
PHOT_ZP['f160w'] = 28.9
PHOT_ZP['f150w'] = 28.9
PHOT_ZP['f200w'] = 28.9
PHOT_ZP['f277w'] = 28.9
PHOT_ZP['f356w'] = 28.9
PHOT_ZP['f444w'] = 28.9
FILTERS = list(PHOT_ZP.keys()) # FILTERS = 'None' # detection only!
TARGET_ZP = 25.0

FILTERS = [x for x in list(PHOT_ZP.keys())]


### PHOTOZ
TRANSLATE_FNAME = 'abell2744_uncover.translate'
ITERATE_ZP = False

### AREA CALCULATIONS
RA_RANGE = (3.487, 3.687)
DEC_RANGE = (-30.5, -30.2)

# ----------------

HST_FILTERS = ['F105W', 'F125W', 'F140W', 'F160W', 'F435W', 'F606W', 'F814W']

SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']

WEBB_FILTERS = SW_FILTERS + LW_FILTERS

PIVOT = OrderedDict([('f435w', 4318.828102108889),
             ('f606w', 5920.818879556311),
             ('f814w', 8056.879509287926),
             ('f105w', 10543.523234897353),
             ('f125w', 12470.520322831206),
             ('f140w', 13924.163916315556),
             ('f160w', 15396.616154585481),
             ('f115w', 11540.),
             ('f150w', 15007.454908178013),
             ('f200w', 19886.478139793544),
             ('f277w', 27577.958764384803),
             ('f356w', 35682.27763839694),
             ('f410m', 40820.),
             ('f444w', 44036.71097714713)])
