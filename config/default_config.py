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
    clean_param = 1.66776,
    )

PHOT_APER = [0.32, 0.48, 0.7 , 1.  , 2.  , 3.] # diameter in arcsec
PHOT_AUTOPARAMS = 2.5, 3.5 # for MAG_AUTO
PHOT_FLUXFRAC = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux

PIXEL_SCALE = 0.04 # arcsec / px

REF_BAND = 'f444w'
APPLY_MWDUST = True

SCI_APER = 0.7 # science aperture
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
ZRA = 'RA'
ZDEC = 'DEC'
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
