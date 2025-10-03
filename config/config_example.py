import os, glob
from typing import OrderedDict
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np
APERPY = '/Users/secutler/Documents/aperpy/'
OVERWRITE = False

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

PHOT_APER = [0.2, 0.32, 0.48, 0.7, 1.0, 1.4] # diameter in arcsec
PHOT_AUTOPARAMS = 2.5, 1.0 # Kron-scaling radius, minimum kron factor
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
PROJECT = 'MINERVA-UDS'
SURVEY = PROJECT.split('-')[0]
FIELD = PROJECT.split('-')[1]
REDUCTION = 'grizli'
VERSION = 'n2.2_m2.0_v1.0'
DRIVE = f'/Volumes/SanDisk3/{SURVEY}/'
WORKING_DIR = f'{DRIVE}{FIELD}/{VERSION}'
DIR_IMAGES = os.path.join(WORKING_DIR, 'external/')
DIR_OUTPUT = os.path.join(WORKING_DIR, 'output/')
DIR_PSFS = os.path.join(WORKING_DIR, 'intermediate/PSF/')
DIR_KERNELS = os.path.join(WORKING_DIR, 'intermediate/kernels/')
DIR_CATALOGS = os.path.join(WORKING_DIR, 'catalogs/')
DIR_CONFIG = f'/Users/secutler/Documents/{SURVEY}/'
IS_COMPRESSED = True # outputs files as .gz

USE_EXPTIME = True # use exposure time maps to get median exposure time for each source

BORROW_HEADER_FILE = glob.glob(DIR_IMAGES+'*f444w*sci.fits*')[0]

PATH_SW_ENERGY = APERPY+'config/Encircled_Energy_SW_ETCv2.txt'
PATH_LW_ENERGY = APERPY+'config/Encircled_Energy_LW_ETCv2.txt'
PATH_HST_ENERGY = APERPY+'config/Encircled_Energy_HST_ETCv2.txt'

SKYEXT = ''
BLOCK_WHT_REPLACE = ('sci', 'wht')
WHT_REPLACE = ('sci', 'wht')
DIRWHT_REPLACE = (DIR_OUTPUT, DIR_IMAGES)
DIR_SFD = '~/sfddata-master'
ZSPEC = f'/Users/secutler/Documents/{SURVEY}/{FIELD.lower()}/{FIELD.lower()}_zspec.fits'
ZCOL= 'z'
ZRA = 'ra'
ZDEC = 'dec'
ZCONF = 'use', 1
MAX_SEP = 0.3 * u.arcsec

### MEDIAN FILTERING
IS_CLUSTER = False
FILTER_SIZE = 5 # arcsec
MED_CENTERS = SkyCoord(3.587*u.deg, -30.40*u.deg)
MED_SIZE = 1.3*u.arcmin
BLOCK_SIZE = 4 # pixels

### BACKGROUNDS
BACKPARAMS = dict(bw=128, bh=128, fw=3, fh=3, maskthresh=1, fthresh=0.)
BACKTYPE = 'var'

FILTERS_ACS = ['F435W','F606W','F775W','F814W','F850LP']
FILTERS_WFC = ['F098M','F105W','F125W','F140W','F160W']
HST_FILTERS = FILTERS_ACS + FILTERS_WFC

SW_FILTERS = ['F090W','F115W','F140M','F150W','F162M','F182M','F200W','F210M']
LW_FILTERS = ['F250M','F277W','F300M','F335M','F356W','F360M','F410M','F430M',
              'F444W','F460M','F480M']

USE_FILTERS = []#LW_FILTERS*1 # Filters to use with WebbPSF

WEBB_FILTERS = SW_FILTERS + LW_FILTERS

FILTERS = HST_FILTERS + WEBB_FILTERS
FILTERS = [filt.lower() for filt in FILTERS]

### DETECTION COADD # use '-' in nicknames, NOT '_'
DETECTION_GROUPS = {'ACS+WEBB':{},'LW':{}}

CHI_MEAN_FILTS = [filt for filt in FILTERS if filt.upper() not in FILTERS_WFC]

DETECTION_GROUPS['ACS+WEBB']['filters'] = tuple(CHI_MEAN_FILTS)
DETECTION_GROUPS['ACS+WEBB']['method'] = 'chi-mean'
DETECTION_GROUPS['ACS+WEBB']['n_regions'] = 3

DETECTION_GROUPS['LW']['filters'] = ('f277w', 'f356w', 'f444w')
DETECTION_GROUPS['LW']['method'] = 'noise-equal'

USE_COMBINED_KRON_IMAGE = True   # uses a REF_BAND PSF-matched NE image for kron radius/flux + flux radius
KRON_COMBINED_BANDS = {}
KRON_COMBINED_BANDS['ACS+WEBB'] = tuple(CHI_MEAN_FILTS)
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
    for filt in DETECTION_GROUPS[group]['filters']:
        path = glob.glob(f'{DIR_IMAGES}*{filt}*sci.fits*')[0]
        DETECTION_IMAGES[filt] = path

ID_FLOOR = 1000000 # value to add to all IDs
# zero if you want to leave them untouched, will not work with XCAT below
### CROSSMATCH to old ID versions
XCAT_FILENAMES_MAIN = {'ACS+WEBB': f'{DRIVE}/{FIELD}/n2.1_m2.0_v1.0/catalogs/ACS+WEBB_chi-mean/f444w/MINERVA-UDS_n2.1_m2.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.fits',
                       'LW': f'{DRIVE}/{FIELD}/n2.1_m2.0_v1.0/catalogs/LW_f277w-f356w-f444w_noise-equal/f444w/MINERVA-UDS_n2.1_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits'}
XCAT_NAME_MAIN = 'id' # column to include, name to use
XCAT_RAD_MAIN = 0.08

### ZEROPOINTS
PHOT_ZP = OrderedDict()
TARGET_ZP = 28.9
FLUX_UNIT = '10*nJy'

### PSF parameters
MATCH_BAND = 'f444w' # indicates band used to match PSFs
PSF_REF_NAME = f'{MATCH_BAND.lower()}_psf.fits'
ANGLE = None
MAGLIM = (14,26)
PSF_DICT = {
    # oPSF generation
    'cutout_size':{}, # diameter of PSF cutout in arcsec
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

for filt in FILTERS:
    PHOT_ZP[filt] = 28.9

    PSF_DICT['cutout_size'][filt] = 4
    PSF_DICT['range'][filt] = [1.2,3]
    PSF_DICT['threshold_max'][filt] = 10
    PSF_DICT['mag_lim'][filt] = 24.0
    PSF_DICT['snr_lim'][filt] = 1000
    PSF_DICT['sigma'][filt] = 2.8
    PSF_DICT['npeaks'][filt] = 1000
    PSF_DICT['aper_scale'][filt] = 0.04/PIXEL_SCALE

    if filt == 'f160w':
        PSF_DICT['range'][filt] = [1.2,3.5]
    
    if filt == 'f140w':
        PSF_DICT['range'][filt] = [2.5,4]
    
    if filt == 'f606w':
        PSF_DICT['npeaks'][filt] = 2000

    PSF_DICT['method'][filt] = 'pypher'
    PSF_DICT['pypher_r'][filt] = 3e-3
    PSF_DICT['oversample'][filt] = 3
    PSF_DICT['alpha'][filt] = 0.1
    PSF_DICT['beta'][filt] = 0.15

### PHOTOZ
TRANSLATE_FNAME = f'/Users/secutler/Documents/{SURVEY}/{FIELD.lower()}/{VERSION}/{SURVEY.lower()}_{FIELD.lower()}_{VERSION[:4]}.translate'
ITERATE_ZP = True
EAZY_FLOOR = False
TEMPLATE_SETS = ['larson','sfhz_blue_agn','sfhz']
EAZY_APERS = ['SUPER',0.32,0.20]


### AREA CALCULATIONS
FNAME = glob.glob(f'{DIR_IMAGES}*{MATCH_BAND.lower()}*sci.fits*')[0]
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
PS_WEBB_FLUXRATIO_RANGE = (1.0, 1.2)
PS_WEBB_FILT = 'f200w'
PS_WEBB_MAGLIMIT = 25.0
PS_WEBB_APERSIZE = 0.7

# POINT-LIKE FLAG - HST
PS_HST_USE = True
PS_HST_FLUXRATIO = (1.4, 0.7)
PS_HST_FLUXRATIO_RANGE = (1.0, 1.2)
PS_HST_FILT = 'f160w'
PS_HST_MAGLIMIT = 23.8
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
EXTERNALSTARS_USE = False
FN_EXTERNALSTARS = f'/Users/secutler/Documents/{SURVEY}/{FIELD.lower()}/{FIELD.lower()}_3dhst_stars.fits'
EXTERNALSTARS_XMATCH_RADIUS = 1*u.arcsec

# COVERAGE FLAGS (flag sources with no coverage in certain bands)
COVERAGE_USE = True
COV_FILTS = ['f435w','f606w']
COV_APERSIZE = 0.7
COV_NAME = 'acs'
COV_SEL_EAZY = True

# NUMBER OF BANDS (can choose to use specific bands; e.g. medium bands, etc.)
NBANDS_USE = True
NBANDS_APERSIZE = 0.7
# Set to None to use all bands
NBANDS_FILTS = ['F140M','F162M','F182M','F210M',
                'F250M','F300M','F335M','F360M','F430M','F460M','F480M'] 
NBANDS_NAME = 'MB'

# BADWHT
BADWHT_USE = True
FN_BADWHT = glob.glob(DIR_IMAGES+'*f200w*wht.fits*')[0]
SATURATEDSTAR_MAGLIMIT = 21
SATURATEDSTAR_FILT = 'f200w'
SATURATEDSTAR_APERSIZE = 0.7

# EXTRABAD (e.g. bCGs)
EXTRABAD_USE = False
FN_EXTRABAD = ''
EXTRABAD_XMATCH_RADIUS = 0.5*u.arcsec
EXTRABAD_LABEL = ''

# REGMASK (mask region file of your choice)
REGMASK_USE = True
FN_REGMASK = f'/Users/secutler/Documents/{SURVEY}/{FIELD.lower()}/{VERSION}/{PROJECT}_{VERSION}_starspike_mask.reg'

### BAD PIXELS
BP_USE = True
BP_FLUXRATIO = (0.7, 0.32)
BP_FLUXRATIO_RANGE = (0, 1.1)
BP_FILT = {'LW':'f444w','ACS+WEBB':'f444w'}
BP_MAGLIMIT = 26.
BP_APERSIZE = 0.7

### ARTIFACTS NEAR BAD PIXELS, EDGES (e.g. saturated star segments)
ANBP_USE = True
ANBP_XMATCH_RADIUS = 3*u.arcsec
ANBP_MIN_NPIX = 10
ANBP_MAX_NPIX = 10000

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
