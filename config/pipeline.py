# Pipeline for UNCOVER

# Load config and hand to aper-py
import os, sys
DIR_CONFIG = '/Volumes/External1/Projects/Current/CEERS/scripts/'
PATH_APERPY = '/Users/jweaver/Projects/Software/aperpy/src'

sys.path.insert(0, DIR_CONFIG)
from config import DETECTION_NICKNAMES, KERNELS, PHOT_APER, TEMPLATE_SETS

# Make PSFs
# os.system(f'python {PATH_APERPY}/make_psfs.py {DIR_CONFIG}')

# # Re-sample SW to 40mas
# os.system(f'python {PATH_APERPY}/resample.py {DIR_CONFIG}')

# # Subtract Sky or subtract cluster (+ sky)
# os.system(f'python {PATH_APERPY}/subtract_background.py {DIR_CONFIG}')

# # Make detection
for det_nickname in DETECTION_NICKNAMES:
    os.system(f'python {PATH_APERPY}/build_detection.py {DIR_CONFIG} {det_nickname}')

# # # # Convolve images
# for kern in KERNELS:
#     os.system(f'python {PATH_APERPY}/convolve_images.py {DIR_CONFIG} {kern}') 

# # Extract on raw images + make catalogs + run photo-z (a)
for det_nickname in DETECTION_NICKNAMES:
    os.system(f'python {PATH_APERPY}/source_extract.py {DIR_CONFIG} {det_nickname} None')
    os.system(f'python {PATH_APERPY}/combine_catalogs_psf.py {DIR_CONFIG} {det_nickname} None')
    for apersize in PHOT_APER:
        APER = str(apersize).replace('.', '_')
        os.system(f'python {PATH_APERPY}/diagnostics.py {DIR_CONFIG} {det_nickname} None {APER}')
        for template in TEMPLATE_SETS:
            os.system(f'python {PATH_APERPY}/eazy_photoz.py {DIR_CONFIG} {det_nickname} None {APER} {template}')

# run on convolved image sets
for det_nickname in DETECTION_NICKNAMES:
    for kern in KERNELS:
        os.system(f'python {PATH_APERPY}/source_extract.py {DIR_CONFIG} {det_nickname} {kern}')
        os.system(f'python {PATH_APERPY}/combine_catalogs_kronlike.py {DIR_CONFIG} {det_nickname} {kern}')
        for apersize in PHOT_APER:
            APER = str(apersize).replace('.', '_')
            os.system(f'python {PATH_APERPY}/diagnostics.py {DIR_CONFIG} {det_nickname} {kern} {APER}')
            for template in TEMPLATE_SETS:
                os.system(f'python {PATH_APERPY}/eazy_photoz.py {DIR_CONFIG} {det_nickname} {kern} {APER} {template}')
        
