# Pipeline example

# Load config and hand to aper-py
import os, sys
import subprocess

def run(process):
    subprocess.run(process,shell=True,check=True)

DIR_CONFIG = '/path/to/here/scripts/'
PATH_APERPY = '/path/to/software/aperpy/src'

sys.path.insert(0, DIR_CONFIG)
from config import DETECTION_NICKNAMES, KERNELS, PHOT_APER, TEMPLATE_SETS, ITERATE_ZP

resample = False
subtract = False
make_psf = False
convolve = False
make_detection = False
make_catalogs = False
run_eazy = False

# Re-sample SW to 40mas
if resample:
    run(f'python {PATH_APERPY}/resample.py {DIR_CONFIG}')

# Subtract Sky or subtract cluster (+ sky)
if subtract:
    run(f'python {PATH_APERPY}/subtract_background.py {DIR_CONFIG}')

# Make PSFs
if make_psf:
    run(f'python {PATH_APERPY}/make_psfs.py {DIR_CONFIG}')

# Convolve images
if convolve:
    for kern in KERNELS:
        run(f'python {PATH_APERPY}/convolve_images.py {DIR_CONFIG} {kern}') 

for det_nickname in DETECTION_NICKNAMES:
    # # Make detection
    if make_detection:
        run(f'python {PATH_APERPY}/build_detection.py {DIR_CONFIG} {det_nickname}')
   
for det_nickname in DETECTION_NICKNAMES:
    # Make and combine catalogs
    if make_catalogs:
        for kern in KERNELS:
            run(f'python {PATH_APERPY}/source_extract.py {DIR_CONFIG} {det_nickname} {kern}')
            run(f'python {PATH_APERPY}/combine_catalogs_kronlike.py {DIR_CONFIG} {det_nickname} {kern}')
            run(f'python {PATH_APERPY}/make_supercatalog.py {DIR_CONFIG} {det_nickname} {kern}')

if run_eazy:
    # Run eazy-photoz 
    for det_nickname in DETECTION_NICKNAMES:
        for kern in KERNELS:
            for APER in ['SUPER'] + PHOT_APER:
                for template in TEMPLATE_SETS:
                    ITERATE = [False,]
                    if ITERATE_ZP: ITERATE += [True]
                    for ITER in ITERATE:
                        run(f'python {PATH_APERPY}/eazy_photoz.py {DIR_CONFIG} {det_nickname} {kern} {APER} {template} {ITER}')

