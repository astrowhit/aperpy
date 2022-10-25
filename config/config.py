import os
from typing import OrderedDict

### GENERAL
KERNELS = {}
KERNELS['f444w'] = 'regularization'
KERNELS['f160w'] = 'shapelets'

DETECTION_PARAMS = dict(
    thresh =  2,
    minarea = 10,
    kernelfwhm = 1.00170,
    deblend_nthresh = 16,
    deblend_cont = 0.00315,
    clean_param = 1.66776,
    )

PHOT_APER = [0.32, 0.48, 0.7 , 1.  , 2.  , 3.] # diameter in arcsec
PHOT_AUTOPARAMS = 2.5, 3.5 # for MAG_AUTO
PHOT_FLUXFRAC = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux

REF_BAND = 'f444w'
FNAME_REF_PSF = f'./data/external/psf_jrw_v4/psf_ceers_F444W_4arcsec.fits'
APPLY_MWDUST = True

### DIRECTORIES
WORKING_DIR = '/Volumes/External1/Projects/Current/UNCOVER/data/vTest'
DIR_IMAGES = os.path.join(WORKING_DIR, 'external/grizli-v5')
DIR_OUTPUT = os.path.join(WORKING_DIR, 'output/')
DIR_PSFS = os.path.join(WORKING_DIR, 'intermediate/PSF/')
DIR_KERNELS = os.path.join(WORKING_DIR, 'intermediate/kernels/')
DIR_CATALOGS = os.path.join(WORKING_DIR, 'catalogs/')

DIR_SFD = '~/Projects/Common/py_tools/sfddata-master'


### MEDIAN FILTERING
FILTER_SIZE = 5 # arcsec
PIXEL_SCALE = 0.04 # arcsec / px

### WEBBPSF GENERATION
PSF_FOV = 4 # arcsec
FIELD = 'uncover'
ANGLE = None # takes the default uncover PA

### BACKGROUNDS
BACKPARAMS = dict(bw=32, bh=32, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'var'

### DETECTION COADD
DETECTION_GROUPS = {}
DETECTION_GROUPS['SW'] = ('f150w', 'f200w')
DETECTION_GROUPS['LW'] = ('f277w', 'f356w', 'f444w')

DETECTION_NICKNAMES = []
for nickname in DETECTION_GROUPS:
    joined = '-'.join(DETECTION_GROUPS[nickname])
    DETECTION_NICKNAMES.append(f'{nickname}_{joined}')

DETECTION_IMAGES = OrderedDict()
DETECTION_IMAGES['f150w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f150w-clear_drc_sci_skysubvar.fits.gz')
DETECTION_IMAGES['f200w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f200w-clear_drc_sci_skysubvar.fits.gz')
DETECTION_IMAGES['f277w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f277w-clear_drc_sci_skysubvar.fits.gz')
DETECTION_IMAGES['f356w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f356w-clear_drc_sci_skysubvar.fits.gz')
DETECTION_IMAGES['f444w'] = os.path.join(DIR_IMAGES, 'ceers-full-grizli-v4.0-f444w-clear_drc_sci_skysubvar.fits.gz')

### ZEROPOINTS
PHOT_ZP = OrderedDict()
PHOT_ZP['f435w'] = 28.9
PHOT_ZP['f606w'] = 28.9
PHOT_ZP['f814w'] = 28.9
## PHOT_ZP['f098m'] = 28.9
PHOT_ZP['f105w'] = 28.9
PHOT_ZP['f125w'] = 28.9
PHOT_ZP['f140w'] = 28.9
PHOT_ZP['f160w'] = 28.9
PHOT_ZP['f115w'] = 28.9
PHOT_ZP['f150w'] = 28.9
PHOT_ZP['f200w'] = 28.9
PHOT_ZP['f277w'] = 28.9
PHOT_ZP['f410m'] = 28.9
PHOT_ZP['f356w'] = 28.9
PHOT_ZP['f444w'] = 28.9
PHOT_NICKNAMES = list(PHOT_ZP.keys()) # PHOT_NICKNAMES = 'None' # detection only!
TARGET_ZPT = 25.0

FILTERS = [x.upper for x in list(PHOT_ZP.keys())]




# ----------------

HST_FILTERS = ['F105W', 'F125W', 'F140W', 'F160W', 'F435W', 'F606W', 'F814W']

SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']

WEBB_FILTERS = SW_FILTERS + LW_FILTERS