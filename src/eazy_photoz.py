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

from config import DIR_CATALOGS, DET_TYPE, TRANSLATE_FNAME, TARGET_ZP, ITERATE_ZP, FILTERS

FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

translate_file = os.path.join(DIR_CONFIG, TRANSLATE_FNAME)

params = {}

params['CATALOG_FILE'] = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_CATALOG.fits')
params['MAIN_OUTPUT_FILE'] = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_CATALOG.eazypy')

params['APPLY_PRIOR'] = 'n'
params['PRIOR_ABZP'] = TARGET_ZP
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

if ITERATE_ZP:
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
print(ez.lnp)

ez.zphot_zspec(include_errors=False)
fig = plt.gcf()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_photoz-specz_.pdf'))


zout, hdu = ez.standard_output(rf_pad_width=0.5, rf_max_err=2, 
                                 prior=False, beta_prior=True)

ez.fit_phoenix_stars()
star_coln = ['star_chi2', 'star_min_ix', 'star_min_chi2', 'star_min_chinu']

for coln in star_coln:
    print(coln, len(ez.__dict__[coln]))
    zout[coln] = ez.__dict__[coln]
#     print(coln)
    
zout.write(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}.zout.fits'), overwrite=True)

# Diagnostics

# 1. mod - obs vs. lam; whisker plot
diff = ez.fnu - ez.fmodel
rel_diff = diff / ez.fmodel

sanity = ez.cat['use_phot'] == 1

fig, ax = plt.subplots(figsize=(10,5))
ax.axhline(0, ls='solid', c='grey')
ax.axhline(0.1, ls=(0, (1, 10)), c='grey')
ax.axhline(-0.1, ls=(0, (1, 10)), c='grey')
axt = ax.twiny()
axt.boxplot(rel_diff[sanity], vert=True, positions=ez.pivot*1e-4, widths=0.1, labels=FILTERS, flierprops={'marker':'.', 'markersize':2, 'alpha':0.1})
axt.set(xlim=(0.05, 5))
ax.set(ylim=(-0.5, 0.5), xlim=(0.1, 5), xlabel='Observed Wavelength ($\mu$m)', ylabel='Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$')

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_wavdiffmodel.pdf'))

#
diff = ez.fnu - ez.fmodel
ztest = diff / ez.efnu

sanity = ez.cat['use_phot'] == 1

fig, ax = plt.subplots(figsize=(10,5))
ax.axhline(0, ls='solid', c='grey')
ax.axhline(1, ls=(0, (1, 10)), c='grey')
ax.axhline(-1, ls=(0, (1, 10)), c='grey')
axt = ax.twiny()
axt.boxplot(ztest[sanity], vert=True, positions=ez.pivot*1e-4, widths=0.1, labels=FILTERS, flierprops={'marker':'.', 'markersize':2, 'alpha':0.1})
axt.set(xlim=(0.05, 5))
ax.set(ylim=(-3, 3), xlim=(0.1, 5), xlabel='Observed Wavelength ($\mu$m)', ylabel='Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm uncertainty}}$')

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_wavdiffmodel_ztest.pdf'))


# 2. mod - obs vs. z, per band
fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
[ax.set_xlabel('$z_{\\rm phot}$') for ax in axes[-1]]
axes = axes.flatten()
axes[-1].set(xlim=(0, 6), ylim=(-0.3, 0.3))
fig.suptitle('Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$', y=0.99, fontsize=20)


diff = ez.fnu - ez.fmodel
rel_diff = diff / ez.fmodel

from aperpy.src.webb_tools import histedges_equalN, binned_med

for i, (filt, fname, ax) in enumerate(zip(FILTERS, ez.filters, axes)):

    ax.axhline(0, ls='solid', c='grey')
    ax.axhline(0.1, ls=(0, (1, 10)), c='grey')
    ax.axhline(-0.1, ls=(0, (1, 10)), c='grey')

    ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)

    mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
    snr = (ez.fnu/ez.efnu)[:,i]

    depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

    sanity = ez.cat['use_phot'] == 1
    sanity &= mag <= depth

    ebins = histedges_equalN(ez.zbest[sanity], 10)
    nbins, bin_centers, bmed, bstd = binned_med(ez.zbest[sanity], rel_diff[sanity, i], bins=ebins)

    ax.scatter(ez.zbest[sanity], rel_diff[sanity, i], c='grey', s=1, alpha=0.1)
    ax.plot(bin_centers, bmed, c='royalblue')
    ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_zdiffmodel.pdf'))


#

fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
[ax.set_xlabel('$z_{\\rm phot}$') for ax in axes[-1]]
axes = axes.flatten()
axes[-1].set(xlim=(0, 6), ylim=(-3, 3))
fig.suptitle('Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{uncertainty}}$', y=0.99, fontsize=20)


diff = ez.fnu - ez.fmodel
ztest = diff / ez.efnu

from aperpy.src.webb_tools import histedges_equalN, binned_med

for i, (filt, fname, ax) in enumerate(zip(FILTERS, ez.filters, axes)):

    ax.axhline(0, ls='solid', c='grey')
    ax.axhline(1, ls=(0, (1, 10)), c='grey')
    ax.axhline(-1, ls=(0, (1, 10)), c='grey')

    ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)

    mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
    snr = (ez.fnu/ez.efnu)[:,i]

    depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

    sanity = ez.cat['use_phot'] == 1
    sanity &= mag <= depth

    ebins = histedges_equalN(ez.zbest[sanity], 10)
    nbins, bin_centers, bmed, bstd = binned_med(ez.zbest[sanity], ztest[sanity, i], bins=ebins)

    ax.scatter(ez.zbest[sanity], ztest[sanity, i], c='grey', s=1, alpha=0.1)
    ax.plot(bin_centers, bmed, c='royalblue')
    ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_zdiffmodel_ztest.pdf'))



# 3. Same as 2 but with respect to mag

fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
[ax.set_xlabel('Mag (AB)') for ax in axes[-1]]
axes = axes.flatten()
axes[-1].set(xlim=(20, 28.5), ylim=(-0.3, 0.3))
fig.suptitle('Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$', y=0.99, fontsize=20)

from aperpy.src.webb_tools import histedges_equalN, binned_med

diff = ez.fnu - ez.fmodel
rel_diff = diff / ez.fmodel

for i, (filt, fname, ax) in enumerate(zip(FILTERS, ez.filters, axes)):

    ax.axhline(0, ls='solid', c='grey')
    ax.axhline(0.1, ls=(0, (1, 10)), c='grey')
    ax.axhline(-0.1, ls=(0, (1, 10)), c='grey')

    ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)
    

    mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
    snr = (ez.fnu/ez.efnu)[:,i]

    depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )
    finite = np.isfinite(mag) & np.isfinite(rel_diff[:,i])

    sanity = ez.cat['use_phot'] == 1
    ax.scatter(mag[sanity], rel_diff[sanity, i], c='grey', s=1, alpha=0.1)
    ax.axvline(depth, c='royalblue',ls='dashed')

    sanity &= mag <= depth
    sanity &= finite

    ebins = np.linspace(20, 28, 10)
    ebins = histedges_equalN(mag[sanity], 10)
    nbins, bin_centers, bmed, bstd = binned_med(mag[sanity], rel_diff[sanity, i], bins=ebins)
    
    
    ax.plot(bin_centers, bmed, c='royalblue')
    ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_magdiffmodel.pdf'))

#

fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
[ax.set_xlabel('Mag (AB)') for ax in axes[-1]]
axes = axes.flatten()
axes[-1].set(xlim=(20, 28.5), ylim=(-3, 3))
fig.suptitle('Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{uncertainty}}$', y=0.99, fontsize=20)

from aperpy.src.webb_tools import histedges_equalN, binned_med

diff = ez.fnu - ez.fmodel
ztest = diff / ez.efnu

for i, (filt, fname, ax) in enumerate(zip(FILTERS, ez.filters, axes)):

    ax.axhline(0, ls='solid', c='grey')
    ax.axhline(1, ls=(0, (1, 10)), c='grey')
    ax.axhline(-1, ls=(0, (1, 10)), c='grey')

    ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)
    

    mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
    snr = (ez.fnu/ez.efnu)[:,i]

    depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )
    finite = np.isfinite(mag) & np.isfinite(ztest[:,i])

    sanity = ez.cat['use_phot'] == 1
    ax.scatter(mag[sanity], ztest[sanity, i], c='grey', s=1, alpha=0.1)
    ax.axvline(depth, c='royalblue',ls='dashed')

    sanity &= mag <= depth
    sanity &= finite

    ebins = np.linspace(20, 28, 10)
    ebins = histedges_equalN(mag[sanity], 10)
    nbins, bin_centers, bmed, bstd = binned_med(mag[sanity], ztest[sanity, i], bins=ebins)
    
    
    ax.plot(bin_centers, bmed, c='royalblue')
    ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_magdiffmodel_ztest.pdf'))