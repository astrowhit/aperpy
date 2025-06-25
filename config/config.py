import os
import glob
from typing import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.io import fits
APERPY = '/path/to/aperpy/'

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

PIXEL_SCALE = 0.04 # arcsec / px
APPLY_MWDUST = 'MEDIAN'
USE_FFT_CONV = True

PHOT_EMPTYAPER_DIAMS = np.linspace(0.16, 1.4, 30)

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
DIR_CONFIG = 'path/to/config/'
IS_COMPRESSED = True # outputs files as .gz

USE_EXPTIME = False # use exposure time maps to get median exposure time for each source

BORROW_HEADER_FILE = 'path/to/image/file/'

PATH_SW_ENERGY = os.path.join(APERPY,'config/Encircled_Energy_SW_ETCv2.txt')
PATH_LW_ENERGY = os.path.join(APERPY,'config/Encircled_Energy_LW_ETCv2.txt')
PATH_HST_ENERGY = os.path.join(APERPY,'config/Encircled_Energy_HST_ETCv2.txt')

SKYEXT = '_skysubvar'
BLOCK_WHT_REPLACE = ('sci', 'wht') # for resampling images
WHT_REPLACE = (f'sci{SKYEXT}', 'wht') # for pipeline steps after subtraction
DIRWHT_REPLACE = (DIR_OUTPUT, DIR_IMAGES) #i.e. no change
DIR_SFD = 'path/to/sfddata-master' # you need to install SFDMap! # pip install sfdmap + download maps
ZSPEC = 'path/to/spec_z.fits'
ZCOL = 'z'
ZRA = 'RA'
ZDEC = 'DEC'
ZCONF = 'zconf', (3, 4) # confidence flag
MAX_SEP = 0.3 * u.arcsec

### MEDIAN FILTERING
IS_CLUSTER = False  # if True, use median filtering
MED_CENTERS = [SkyCoord(3.587*u.deg, -30.40*u.deg)] # where to center the median filter regions
MED_SIZE = 1.3*u.arcmin
BLOCK_SIZE = 10 # pixels
FILTER_SIZE = 8.3 # arcsec

### BACKGROUNDS
BACKPARAMS = dict(bw=32, bh=32, fw=8, fh=8, maskthresh=1, fthresh=0.)
BACKTYPE = 'var' # var, global, med, or none

FILTERS_ACS = ['F435W','F606W','F814W']
FILTERS_WFC = ['F105W','F125W','F140W','F160W']
HST_FILTERS = FILTERS_ACS + FILTERS_WFC

SW_FILTERS = ['F090W','F115W','F150W','F200W']
LW_FILTERS = ['F277W','F356W','F410M','F444W']
WEBB_FILTERS = SW_FILTERS + LW_FILTERS

FILTERS = HST_FILTERS + WEBB_FILTERS
FILTERS = [filt.lower() for filt in FILTERS]


### DETECTION COADD # use '-' in nicknames, NOT '_'
DETECTION_GROUPS = {'LW':{}}
DETECTION_GROUPS['LW']['filters'] = ('f277w', 'f356w', 'f444w')
DETECTION_GROUPS['LW']['method'] = 'noise-equal'

USE_COMBINED_KRON_IMAGE = True   # uses a REF_BAND PSF-matched NE image for kron radius/flux + flux radius
KRON_COMBINED_BANDS = {}
KRON_COMBINED_BANDS['LW'] = ('f277w', 'f356w', 'f444w')
KRON_ZPT = 28.9 # I hope it's the same as all of your combined mosaics!

DETECTION_NICKNAMES = []
for nickname in DETECTION_GROUPS:
    if len(DETECTION_GROUPS[nickname]['filters'])<=3:
        joined = '-'.join(DETECTION_GROUPS[nickname]['filters'])
        DETECTION_NICKNAMES.append(f'{nickname}_{joined}')
    else:
        DETECTION_NICKNAMES.append(nickname)

DETECTION_IMAGES = OrderedDict()
for group in DETECTION_GROUPS:
    for filt in DETECTION_GROUPS[group]:
        for path in glob.glob(DIR_OUTPUT+'*'):
            if (f'sci{SKYEXT}.fits.gz' in path) & (filt in path):
                DETECTION_IMAGES[filt] = path


### ZEROPOINTS
PHOT_ZP = OrderedDict()
TARGET_ZP = 28.9
FLUX_UNIT = '10*nJy'

### PSF parameters
MATCH_BAND = 'f444w' # indicates band used to match PSFs
PSF_REF_NAME = f'{MATCH_BAND.lower()}_psf.fits'
MAGLIM = (14,26)
PSF_FOV = 4 # arcsec
PSF_DICT = {
    # oPSF generation
    'range':{}, # range of flux ratios for determining point-source locus
    'threshold_max':{}, # point source detection threshold
    'mag_lim':{}, # magnitude limit for point source detection
    'snr_lim':{}, # minimum S/N for a point-source to be included in oPSF
    'sigma':{}, # standard deviation for sigma-clipping
    'npeaks':{}, # number of peaks to retain in star finding step
    'aper_scale':{}, # multiplicative factor for scaling default aperture size

    # PSF homogenization
    'method':{}, # method used to homogenize PSF ('pypher' or 'phoutils')
    'pypher_r':{}, # pypher regularization parameter for PSF homogenization
    'oversample':{}, # oversampling factor for PSF homogenization
    'alpha':{}, # alpha parameter for photutils SplitCosineBellWindow
    'beta':{}, # beta parameter for photutils SplitCosineBellWindow
}

# can edit these for filters of your choosing
for filt in FILTERS:
    PHOT_ZP[filt] = 28.9

    PSF_DICT['range'][filt] = [1.2,3]
    PSF_DICT['threshold_max'][filt] = 10
    PSF_DICT['mag_lim'][filt] = 24.0
    PSF_DICT['snr_lim'][filt] = 1000
    PSF_DICT['sigma'][filt] = 2.8
    PSF_DICT['npeaks'][filt] = 1000
    PSF_DICT['aper_scale'][filt] = 0.04/PIXEL_SCALE

    PSF_DICT['method'][filt] = 'pypher'
    PSF_DICT['pypher_r'][filt] = 3e-3
    PSF_DICT['oversample'][filt] = 3
    PSF_DICT['alpha'][filt] = 0.1
    PSF_DICT['beta'][filt] = 0.15


### PHOTOZ
TRANSLATE_FNAME = '/path/to/eazy.translate'
ITERATE_ZP = False
EAZY_FLOOR = True
TEMPLATE_SETS = ('fsps_full', 'sfhz')

### AREA CALCULATIONS
FNAME = glob.glob(f'{DIR_IMAGES}*{LW_FILTERS[-1].lower()}*sci.fits*')[0]
hdr = fits.getheader(FNAME)
L1,L2 = hdr['NAXIS1'],hdr['NAXIS2']
crval1,crval2 = hdr['CRVAL1'],hdr['CRVAL2']
crpix1,crpix2 = hdr['CRPIX1'],hdr['CRPIX2']
ra_min = crval1-(crpix1*PIXEL_SCALE/3600)
dec_min = crval2-(crpix2*PIXEL_SCALE/3600)
ra_max = crval1+((L1-crpix1)*PIXEL_SCALE/3600)
dec_max = crval2+((L2-crpix2)*PIXEL_SCALE/3600)
RA_RANGE = (ra_min, ra_max)
DEC_RANGE = (dec_min, dec_max)

### STARS AND BAD PIXELS -- currrently set for f444w-matched images only!
# POINT-LIKE FLAG - WEBB
PS_WEBB_USE = True
PS_WEBB_FLUXRATIO = (0.7, 0.32)
PS_WEBB_FLUXRATIO_RANGE = (1.1, 1.2)
PS_WEBB_FILT = 'f200w'
PS_WEBB_MAGLIMIT = 25.0
PS_WEBB_APERSIZE = 0.7

# POINT-LIKE FLAG - HST
PS_HST_USE = True
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
GAIA_USE = True
GAIA_ROW_LIMIT = 10000
GAIA_XMATCH_RADIUS = 0.6*u.arcsec

# EXTERNAL STARS (useful for high proper motion stars)
EXTERNALSTARS_USE = True
FN_EXTERNALSTARS = 'path/to/external/files/star_catalog.fits' # includes ra and dec at minimum
EXTERNALSTARS_XMATCH_RADIUS = 0.7*u.arcsec

# COVERAGE FLAGS (flag sources with no coverage in certain bands)
COVERAGE_USE = False
COV_FILTS = ['f435w','f606w']
COV_APERSIZE = 0.7
COV_NAME = 'acs'

# NUMBER OF BANDS (can choose to use specific bands; e.g. medium bands, etc.)
NBANDS_USE = True
NBANDS_APERSIZE = 0.7
NBANDS_FILTS = SW_FILTERS # Set to None to use all bands
NBANDS_NAME = 'NIRCAM_SW'

# BADWHT (useful for bad regions of the images)
BADWHT_USE = True 
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