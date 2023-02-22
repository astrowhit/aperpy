import os
from typing import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord

### GENERAL
KERNELS = {}
KERNELS['f444w'] = 'regularization'
# KERNELS['f160w'] = 'shapelets'

DETECTION_PARAMS = dict(
    thresh =  1.2,
    minarea = 3,
    kernelfwhm = 3.5,
    deblend_nthresh = 32,
    deblend_cont = 0.0001,
    clean_param = 1.0,
    clean = False,
    )

PHOT_APER = [0.32, 0.48, 0.7 , 1.5] # diameter in arcsec
PHOT_AUTOPARAMS = 2.0, 1.0 # Kron-scaling radius, mimumum kron factor
PHOT_FLUXRADIUS = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux (always keep 0.5!)
PHOT_KRONPARAM = 6.0 # SE hardcodes this as 6.0
PHOT_USEMASK = True # masks out neighbors when measuring kron, auto fluxes, and flux radius (not circ apers)
REF_BAND = 'f444w'
USE_COMBINED_KRON_IMAGE = True   # uses a REF_BAND PSF-matched NE image for kron radius/flux + flux radius
KRON_REF_BAND = ['f277w', 'f356w', 'f444w']


PIXEL_SCALE = 0.04 # arcsec / px
APPLY_MWDUST = 'MEDIAN'
USE_FFT_CONV = True

SCI_APER = 0.32 # science aperture
MAKE_SCIREADY_ALL = True # make aperture corrected catalogs for all apertures


### DIRECTORIES
PROJECT = 'UNCOVER'
VERSION = '0.4.1a'
IS_COMPRESSED = True # outputs files as .gz
WORKING_DIR = '/Volumes/Weaver_2TB/Projects/Current/UNCOVER/data/v0.4a'
DIR_IMAGES = os.path.join(WORKING_DIR, '../external/grizli-v5.4-bcgsub/')
DIR_OUTPUT = os.path.join(WORKING_DIR, 'output/')
DIR_PSFS = os.path.join(WORKING_DIR, 'intermediate/PSF/')
DIR_KERNELS = os.path.join(WORKING_DIR, 'intermediate/kernels/') # Generally
DIR_CATALOGS = os.path.join(WORKING_DIR, 'catalogs/')

PATH_SW_ENERGY = '/Users/jweaver/Projects/Software/aperpy/config/Encircled_Energy_SW.txt'
PATH_LW_ENERGY = '/Users/jweaver/Projects/Software/aperpy/config/Encircled_Energy_LW.txt'

SKYEXT = ''
WHT_REPLACE = ('bcgs_sci', 'wht') # easy as it comes.
DIRWHT_REPLACE = (DIR_OUTPUT, DIR_IMAGES) #i.e. no change
DIR_SFD = '~/Projects/Common/py_tools/sfddata-master'
ZSPEC = '/Volumes/Weaver_2TB/Projects/Current/UNCOVER/data/external/zspec_abell2744_all.fits'
ZCOL = 'z'
ZRA = 'RA'
ZDEC = 'DEC'
ZCONF = 'zconf', (3, 4)
MAX_SEP = 0.3 * u.arcsec

### MEDIAN FILTERING
IS_CLUSTER = True
FILTER_SIZE = 8.3 # arcsec
MED_CENTERS = [SkyCoord(3.587*u.deg, -30.40*u.deg), SkyCoord(3.577*u.deg, -30.35*u.deg), SkyCoord(3.548*u.deg, -30.37*u.deg)]
MED_SIZE = 1.3*u.arcmin
BLOCK_SIZE = 10 # pixels

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
# DETECTION_GROUPS['SW'] = ('f150w', 'f200w')
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
            if ('sci.fits.gz' in path) & (filt in path):
                DETECTION_IMAGES[filt] = path

### ZEROPOINTS
PHOT_ZP = OrderedDict()
PHOT_ZP['f435w'] = 28.9
PHOT_ZP['f606w'] = 28.9
PHOT_ZP['f814w'] = 28.9
PHOT_ZP['f090w'] = 28.9
PHOT_ZP['f105w'] = 28.9
PHOT_ZP['f115w'] = 28.9
PHOT_ZP['f125w'] = 28.9
PHOT_ZP['f140w'] = 28.9
PHOT_ZP['f150w'] = 28.9
PHOT_ZP['f160w'] = 28.9
PHOT_ZP['f200w'] = 28.9
PHOT_ZP['f277w'] = 28.9
PHOT_ZP['f356w'] = 28.9
PHOT_ZP['f410m'] = 28.9
PHOT_ZP['f444w'] = 28.9
FILTERS = [x for x in list(PHOT_ZP.keys())]
TARGET_ZP = 28.9
FLUX_UNIT = '10*nJy'


### PHOTOZ
TRANSLATE_FNAME = 'abell2744_uncover.translate'
ITERATE_ZP = False
TEMPLATE_SETS = ('fsps_full', 'sfhz')

### AREA CALCULATIONS
RA_RANGE = (3.487, 3.687)
DEC_RANGE = (-30.5, -30.2)

### STARS AND BAD PIXELS -- currrently set for f444w-matched images only!
# POINT-LIKE FLAG - WEBB
PS_WEBB_FLUXRATIO = (0.7, 0.32)
PS_WEBB_FLUXRATIO_RANGE = (1.2, 1.5)
PS_WEBB_FILT = 'f200w'
PS_WEBB_MAGLIMIT = 25.0
PS_WEBB_APERSIZE = 0.7

# POINT-LIKE FLAG - HST
PS_HST_FLUXRATIO = (0.7, 0.32)
PS_HST_FLUXRATIO_RANGE = (1.2, 1.5)
PS_HST_FILT = 'f160w'
PS_HST_MAGLIMIT = 23.8
PS_HST_APERSIZE = 0.7

# GAIA
GAIA_ROW_LIMIT = 10000
GAIA_XMATCH_RADIUS = 0.7*u.arcsec

# BADWHT
FN_BADWHT = os.path.join(os.path.join(WORKING_DIR, DIR_IMAGES), 'uncover_v5.4_abell2744clu_f200w_block40_wht.fits.gz')
SATURATEDSTAR_MAGLIMIT = 21
SATURATEDSTAR_FILT = 'f200w'
SATURATEDSTAR_APERSIZE = 0.7

# EXTRABAD (e.g. bCGs)
FN_EXTRABAD = '/Volumes/Weaver_2TB/Projects/Current/UNCOVER/data/external/uncover_v5.4_f444w_bcgs_out.fits'
EXTRABAD_XMATCH_RADIUS = 3*u.arcsec
EXTRABAD_LABEL = 'bCG residuals'

### BAD PIXELS
BP_FLUXRATIO = (0.7, 0.32)
BP_FLUXRATIO_RANGE = (0, 1.1)
BP_FILT = {'LW':'f444w'}
BP_MAGLIMIT = 26.
BP_APERSIZE = 0.7

### BAD KRON RADII
BK_MINSIZE = 3.5 # arcsec
BK_SLOPE = 250

### USER SUPPLED BAD IDs
PATH_BADOBJECT = None

### HACK for GLASS (to turn off, set GLASS_MASK = None)
from astropy.io import fits
GLASS_MASK = fits.getdata('/Volumes/Weaver_2TB/Projects/Current/UNCOVER/data/external/GLASS_MASK.fits.gz')


# ----------------

HST_FILTERS = ['F105W', 'F125W', 'F140W', 'F160W', 'F435W', 'F475W', 'F606W', 'F775W', 'F814W']

SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']

WEBB_FILTERS = SW_FILTERS + LW_FILTERS

PIVOT = OrderedDict([('f435w', 4318.828102108889),
             ('f606w', 5920.818879556311),
             ('f814w', 8056.879509287926),
             ('f090w', 8989.),
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