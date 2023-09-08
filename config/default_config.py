import os
from typing import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

### GENERAL
KERNELS = {}
KERNELS['f444w'] = 'regularization'

DETECTION_PARAMS = dict(
    thresh =  1.2,
    minarea = 3,
    kernelfwhm = 3.5,
    deblend_nthresh = 32,
    deblend_cont = 0.0001,
    clean_param = 1.0,
    clean = False,
    )

PHOT_APER = [0.32, 0.48, 0.7, 1.0, 1.4] # diameter in arcsec
PHOT_AUTOPARAMS = 2.5, 1.0 # Kron-scaling radius, mimumum kron factor
PHOT_FLUXRADIUS = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux (always keep 0.5!)
PHOT_KRONPARAM = 6.0 # SE hardcodes this as 6.0
PHOT_USEMASK = True # masks out neighbors when measuring kron, auto fluxes, and flux radius (not circ apers)
MATCH_BAND = 'f444w' # indicates band used to match PSFs
PSF_REF_NAME = None

PIXEL_SCALE = 0.04 # arcsec / px
APPLY_MWDUST = 'MEDIAN'
USE_FFT_CONV = True

PHOT_EMPTYAPER_DIAMS = np.linspace(0.16, 1.4, 30)

BORROW_HEADER_FILE = 'path/to/image/file/'

BLEND_SHRINK_FACTOR = 1.2 # factor by which the isophotal areas are shrunk for assigning apertures to blends

SCI_APER = 0.32 # set to the aperture size you expect to use most
MAKE_SCIREADY_ALL = True # make aperture corrected catalogs for all apertures

### DIRECTORIES
PROJECT = 'PROJECT'
VERSION = '0.0.1'
WORKING_DIR = 'path/to/working/directory'
DIR_IMAGES = os.path.join(WORKING_DIR, 'images/')

DIR_OUTPUT = os.path.join(WORKING_DIR, 'output/')
DIR_PSFS = os.path.join(WORKING_DIR, 'intermediate/PSF/')
DIR_KERNELS = os.path.join(WORKING_DIR, 'intermediate/kernels/')
DIR_CATALOGS = os.path.join(WORKING_DIR, 'catalogs/')
IS_COMPRESSED = True # outputs files as .gz

PATH_SW_ENERGY = '/path/to/config/Encircled_Energy_SW_ETCv2.txt'
PATH_LW_ENERGY = '/path/to/config/Encircled_Energy_LW_ETCv2.txt'

SKYEXT = ''
WHT_REPLACE = ('sci', 'wht') # easy as it comes.
DIRWHT_REPLACE = (DIR_OUTPUT, DIR_IMAGES) #i.e. no change
DIR_SFD = 'path/to/sfddata-master' # you need to install SFDMap! # pip install sfdmap + download maps
ZSPEC = 'path/to/spec_z.fits'
ZCOL = 'z'
ZRA = 'RA'
ZDEC = 'DEC'
ZCONF = 'zconf', (3, 4) # confidence flag
MAX_SEP = 0.3 * u.arcsec

### MEDIAN FILTERING
IS_CLUSTER = True  # if True, use median filtering
FILTER_SIZE = 8.3 # arcsec
MED_CENTERS = [SkyCoord(3.587*u.deg, -30.40*u.deg)] # where to center the median filter regions
MED_SIZE = 1.3*u.arcmin
BLOCK_SIZE = 10 # pixels

### PSF GENERATION
OVERSAMPLE = 3
ALPHA = 0.3
BETA = 0.15
PYPHER_R = 3e-3
MAGLIM = (18.0, 24.0)
PSF_FOV = 4 # arcsec

### BACKGROUNDS
BACKPARAMS = dict(bw=32, bh=32, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'var' # var, global, med, or none

### DETECTION COADD # use '-' in nicknames, NOT '_'
DETECTION_GROUPS = {}
DETECTION_GROUPS['LW'] = ('f277w', 'f356w', 'f444w')

USE_COMBINED_KRON_IMAGE = True   # uses a REF_BAND PSF-matched NE image for kron radius/flux + flux radius
KRON_COMBINED_BANDS = {}
KRON_COMBINED_BANDS['LW'] = ('f277w', 'f356w', 'f444w')
KRON_ZPT = 28.9 # I hope it's the same as all of your combined mosaics!

DET_TYPE = 'noise-equal'
DETECTION_NICKNAMES = []
for nickname in DETECTION_GROUPS:
    if len(nickname) > 1:
        joined = '-'.join(DETECTION_GROUPS[nickname])
    else:
        joined = nickname
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
TEMPLATE_SETS = ('fsps_full', 'sfhz') #, 'sfhz_blue')

### AREA CALCULATIONS
RA_RANGE = (3.487, 3.687)
DEC_RANGE = (-30.5, -30.2)

### STARS AND BAD PIXELS -- currrently set for f444w-matched images only!
# POINT-LIKE FLAG - WEBB
PS_WEBB_USE = True
PS_WEBB_FLUXRATIO = (0.7, 0.32)
PS_WEBB_FLUXRATIO_RANGE = (1.1, 1.2)
PS_WEBB_FILT = 'f200w'
PS_WEBB_MAGLIMIT = 25.0
PS_WEBB_APERSIZE = 0.7

# POINT-LIKE FLAG - HST
PS_HST_USE = False
PS_HST_FLUXRATIO = (0.7, 0.32)
PS_HST_FLUXRATIO_RANGE = (1.5, 1.65)
PS_HST_FILT = 'f160w'
PS_HST_MAGLIMIT = 23
PS_HST_APERSIZE = 0.7

# AUTOSTAR -- flag stars found in PSF star catalogs
AUTOSTAR_USE = True
AUTOSTAR_BANDS = FILTERS
AUTOSTAR_XMATCH_RADIUS = 0.3*u.arcsec
AUTOSTAR_NFILT = 1

# GAIA
GAIA_USE = False
GAIA_ROW_LIMIT = 10000
GAIA_XMATCH_RADIUS = 0.6*u.arcsec

# EXTERNAL STARS (useful for high proper motion stars)
EXTERNALSTARS_USE = True
FN_EXTERNALSTARS = 'path/to/external/files/UNCOVER_F160W_stars.fits' # includes ra and dec at minimum
EXTERNALSTARS_XMATCH_RADIUS = 0.7*u.arcsec

# BADWHT (useful for bad regions of the images)
BADWHT_USE = False 
FN_BADWHT = os.path.join(os.path.join(WORKING_DIR, DIR_IMAGES), 'uncover_v7.0_abell2744clu_f200w_block40_wht.fits.gz')
SATURATEDSTAR_MAGLIMIT = 21
SATURATEDSTAR_FILT = 'f200w'
SATURATEDSTAR_APERSIZE = 0.7

# EXTRABAD (e.g. bCGs)
EXTRABAD_USE = True
FN_EXTRABAD = 'path/to/external/files/uncover_v7.0_f444w_bcgs_out.fits'
EXTRABAD_XMATCH_RADIUS = 3*u.arcsec
EXTRABAD_LABEL = 'bCG residuals'

# REGMASK (mask region file of your choice)
REGMASK_USE = True
FN_REGMASK = 'path/to/external/files/UNCOVER_v2.2.0_SUPERCATALOG_starspike_mask.reg'

### BAD PIXELS
BP_USE = True
BP_FLUXRATIO = (0.7, 0.32)
BP_FLUXRATIO_RANGE = (0, 1.1)
BP_FILT = {'LW':'f444w'}
BP_MAGLIMIT = 26.
BP_APERSIZE = 0.7

### ARTIFACTS NEAR BAD PIXELS, EDGES
ANBP_USE = True
ANBP_XMATCH_RADIUS = 3*u.arcsec
ANBP_MIN_NPIX = 10 
ANBP_MAX_NPIX = 1000

### BAD KRON RADII
BK_MINSIZE = 3.5 # arcsec
BK_SLOPE = 250

### USER SUPPLED BAD IDs
BADOBJECT_USE = False
PATH_BADOBJECT = None

### CROSSMATCH (otherwise set to None)
XCAT_FILENAME = None
XCAT_NAME = 'id', 'DR1' # column to include, name to use
XCAT_RAD = 0.08

### CROSSMATCH (otherwise set to None)
XCAT2_FILENAME = None
XCAT2_NAME = 'id', 'INT_v2'
XCAT2_RAD = 0.08

### CROSSMATCH (otherwise set to None)
XCAT3_FILENAME = None
XCAT3_NAME = 'id_msa', 'msa'
XCAT3_RAD = 0.24

# ---------------- generally don't touch this! ------

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