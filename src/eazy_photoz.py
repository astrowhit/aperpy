import os, sys
import numpy as np
import matplotlib.pyplot as plt

import eazy
print(eazy.__version__)

# Symlink templates & filters from the eazy-code repository
print('EAZYCODE = '+os.getenv('EAZYCODE'))

eazy.symlink_eazy_inputs() 

# quiet numpy/astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

DIR_CONFIG = sys.argv[1]
sys.path.insert(0, DIR_CONFIG)

DET_NICKNAME =  sys.argv[2] #'LW_f277w-f356w-f444w'  
KERNEL = sys.argv[3] #'f444w'
APERSIZE = str(sys.argv[4]).replace('.', '_')

from config import DIR_CATALOGS, DET_TYPE, TRANSLATE_FNAME, TARGET_ZPT

FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

translate_file = os.path.join(DIR_CONFIG, TRANSLATE_FNAME)

params = {}

params['CATALOG_FILE'] = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_CATALOG.fits')
params['MAIN_OUTPUT_FILE'] = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_CATALOG.eazypy')

params['APPLY_PRIOR'] = 'n'
params['PRIOR_ABZP'] = TARGET_ZPT
params['MW_EBV'] = 0.0
params['CAT_HAS_EXTCORR'] = 'y'

params['Z_MAX'] = 20
params['Z_STEP'] = 0.1

params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'

params['VERBOSITY'] = 1

ez = eazy.photoz.PhotoZ(param_file=None,
                              translate_file=translate_file,
                              zeropoint_file=None, params=params,
                              load_prior=True, load_products=False)


NITER = 10
NBIN = np.minimum(ez.NOBJ//100, 180)

ez.param.params['VERBOSITY'] = 1.
for iter in range(NITER):
    print('Iteration: ', iter)
    
    sn = ez.fnu/ez.efnu
    clip = (sn > 5).sum(axis=1) > 5 # Generally make this higher to ensure reasonable fits
    clip &= ez.cat['use_phot'] == 1
    ez.iterate_zp_templates(idx=ez.idx[clip], update_templates=False, 
                              update_zeropoints=True, iter=iter, n_proc=8, 
                              save_templates=False, error_residuals=(iter > 0), 
                              NBIN=NBIN, get_spatial_offset=False)



# Turn off error corrections derived above
ez.efnu = ez.efnu_orig

# Full catalog
sample = ez.idx # all

ez.fit_parallel(sample, n_proc=4, prior=False)


ez.zphot_zspec(include_errors=False)
fig = plt.gcf()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}_K{KERNEL}_SCIREADY_{APERSIZE}_photoz-specz_.pdf'))


zout, hdu = ez.standard_output(rf_pad_width=0.5, rf_max_err=2, 
                                 prior=False, beta_prior=True)

ez.fit_phoenix_stars()
star_coln = ['star_chi2', 'star_min_ix', 'star_min_chi2', 'star_min_chinu']

for coln in star_coln:
    print(coln, len(ez.__dict__[coln]))
    zout[coln] = ez.__dict__[coln]
#     print(coln)
    
zout.write(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}_K{KERNEL}_SCIREADY_{APERSIZE}.zout.fits'), overwrite=True)


