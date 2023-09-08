# Pipeline example

# Load config and hand to aper-py
import os, sys
DIR_CONFIG = '/path/to/here/scripts/'
PATH_APERPY = '/path/to/software/aperpy/src'

sys.path.insert(0, DIR_CONFIG)
from config import DETECTION_NICKNAMES, KERNELS, PHOT_APER, TEMPLATE_SETS

# Re-sample SW to 40mas
os.system(f'python {PATH_APERPY}/resample.py {DIR_CONFIG}')

# Subtract Sky or subtract cluster (+ sky)
os.system(f'python {PATH_APERPY}/subtract_background.py {DIR_CONFIG}')

# # Make detection
for det_nickname in DETECTION_NICKNAMES:
    os.system(f'python {PATH_APERPY}/build_detection.py {DIR_CONFIG} {det_nickname}')

# Make PSFs
os.system(f'python {PATH_APERPY}/make_psfs.py {DIR_CONFIG}')

# Convolve images
for kern in KERNELS:
    os.system(f'python {PATH_APERPY}/convolve_images.py {DIR_CONFIG} {kern}') 

# Extract on raw images + make catalogs + run photo-z (a)
for det_nickname in DETECTION_NICKNAMES:
    os.system(f'python {PATH_APERPY}/source_extract.py {DIR_CONFIG} {det_nickname} {kern}')
    os.system(f'python {PATH_APERPY}/combine_catalogs_kronlike.py {DIR_CONFIG} {det_nickname} {kern}')
    os.system(f'python {PATH_APERPY}/make_supercatalog.py {DIR_CONFIG} {det_nickname} {kern}')

    # run photoz (each aper + super...)
    for APER in ['SUPER'] + PHOT_APER:
        if APER != 'SUPER':
            APER = str(APER).replace('.', '_')
        for template in TEMPLATE_SETS:
            os.system(f'python {PATH_APERPY}/eazy_photoz.py {DIR_CONFIG} {det_nickname} {kern} {APER} {template}')

