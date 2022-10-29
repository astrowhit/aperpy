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

params['Z_MAX'] = 30
params['Z_STEP'] = 0.1

params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'

params['VERBOSITY'] = 1

ez = eazy.photoz.PhotoZ(param_file=None,
                              translate_file=translate_file,
                              zeropoint_file=None, params=params,
                              load_prior=True, load_products=False)


NITER = 10
NBIN = np.minimum(ez.NOBJ//100, 180)

ez.cat = ez.cat.filled(-99)

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

ez.zphot_zspec(include_errors=True)
fig = plt.gcf()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_photoz-specz.pdf'))


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
ztest = diff / ez.efnu
dmag = -2.5*np.log10(ez.fnu/ez.fmodel)

for test, ylabel, fname in zip((rel_diff, ztest, dmag), 
                        ('Relative Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$', '$f$-test $\\frac{\\rm{observed}-\\rm{model}}{\\rm{uncertainty}}$', '$\Delta$Mag observed - model (AB)'),
                        ('reldiff_wav', 'ztest_wav', 'dmag_wav')
                        ):

    fig, ax = plt.subplots(figsize=(10,5))
    ax.axhline(0, ls='solid', c='grey')
    
    if test is ztest:
        ax.set_ylim(-3, 3)
        hline = 1
        ax.set_yticks((-2, 0, 2), ('$-2\sigma$', '$0\sigma$', '$2\sigma$'))
    else:
        ax.set_ylim(-0.35, 0.35)
        hline = 0.1
    ax.axhline(hline, ls=(0, (1, 10)), c='grey')
    ax.axhline(-hline, ls=(0, (1, 10)), c='grey')
    axt = ax.twiny()

    test_ls = []
    for i, filt in enumerate(FILTERS):

        mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
        snr = (ez.fnu/ez.efnu)[:,i]

        depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

        sanity = ez.cat['use_phot'] == 1
        sanity &= mag <= depth
        sanity &= ~np.isnan(mag)

        test_ls.append(test[sanity, i])

    axt.boxplot(test_ls, vert=True, positions=ez.pivot*1e-4, widths=0.1, labels=FILTERS, flierprops={'marker':'.', 'markersize':2, 'alpha':0.1})
    axt.set(xlim=(0.05, 5))
    ax.set(xlim=(0.1, 5), xlabel='Observed Wavelength ($\mu$m)', ylabel=ylabel)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_{fname}.pdf'))


# 2. mod - obs vs. z, per band
diff = ez.fnu - ez.fmodel
rel_diff = diff / ez.fmodel
ztest = diff / ez.efnu
dmag = -2.5*np.log10(ez.fnu/ez.fmodel)

for test, ylabel, fname in zip((rel_diff, ztest, dmag), 
                        ('Relative Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$', '$f$-test $\\frac{\\rm{observed}-\\rm{model}}{\\rm{uncertainty}}$', '$\Delta$Mag observed - model (AB)'),
                        ('reldiff_z', 'ztest_z', 'dmag_z')
                        ):

    fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
    [ax.set_xlabel('$z_{\\rm phot}$') for ax in axes[-1]]
    axes = axes.flatten()
    axes[-1].set(xlim=(0, 6.3))
    fig.suptitle(ylabel, y=0.99, fontsize=20)

    from aperpy.src.webb_tools import histedges_equalN, binned_med

    for i, (filt, ax) in enumerate(zip(FILTERS, axes)):

        ax.axhline(0, ls='solid', c='grey')
        if test is ztest:
            ax.set_ylim(-3, 3)
            hline = 1
            ax.set_yticks((-2, 0, 2), ('$-2\sigma$', '$0\sigma$', '$+2\sigma$'))
        else:
            ax.set_ylim(-0.35, 0.35)
            hline = 0.1
        ax.axhline(hline, ls=(0, (1, 10)), c='grey')
        ax.axhline(-hline, ls=(0, (1, 10)), c='grey')

        ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)

        mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
        snr = (ez.fnu/ez.efnu)[:,i]

        depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

        sanity = ez.cat['use_phot'] == 1
        sanity &= mag <= depth
        sanity &= ~np.isnan(mag)

        ebins = histedges_equalN(ez.zbest[sanity], 15)
        nbins, bin_centers, bmed, bstd = binned_med(ez.zbest[sanity], test[sanity, i], bins=ebins)

        ax.scatter(ez.zbest[sanity], test[sanity, i], c='grey', s=1, alpha=0.1)
        ax.plot(bin_centers, bmed, c='royalblue')
        ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

        delta = np.nanmedian(test[sanity,i])
        ax.text(0.65, 0.8, f'$\Delta={delta:2.3f}$', transform=ax.transAxes, fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_{fname}.pdf'))



# 3. Same as 2 but with respect to mag
diff = ez.fnu - ez.fmodel
rel_diff = diff / ez.fmodel
ztest = diff / ez.efnu
dmag = -2.5*np.log10(ez.fnu/ez.fmodel)

for test, ylabel, fname in zip((rel_diff, ztest, dmag), 
                        ('Relative Flux $\\frac{\\rm{observed}-\\rm{model}}{\\rm{model}}$', '$f$-test $\\frac{\\rm{observed}-\\rm{model}}{\\rm{uncertainty}}$', '$\Delta$Mag observed - model (AB)'),
                        ('reldiff_mag', 'ztest_mag', 'dmag_mag')
                        ):

    fig, axes = plt.subplots(nrows=int(ez.NFILT/2.), ncols=2, figsize=(10, 2*int(ez.NFILT/2.+1)), sharey=True, sharex=True)
    [ax.set_xlabel('Mag (AB)') for ax in axes[-1]]
    axes = axes.flatten()
    axes[-1].set(xlim=(19, 28.25))
    fig.suptitle(ylabel, y=0.99, fontsize=20)

    from aperpy.src.webb_tools import histedges_equalN, binned_med

    for i, (filt, ax) in enumerate(zip(FILTERS, axes)):

        ax.axhline(0, ls='solid', c='grey')
        if test is ztest:
            ax.set_ylim(-3, 3)
            hline = 1
            ax.set_yticks((-2, 0, 2), ('$-2\sigma$', '$0\sigma$', '$+2\sigma$'))
        else:
            ax.set_ylim(-0.35, 0.35)
            hline = 0.1
        ax.axhline(hline, ls=(0, (1, 10)), c='grey')
        ax.axhline(-hline, ls=(0, (1, 10)), c='grey')

        ax.text(0.05, 0.8, filt, transform=ax.transAxes, fontsize=15)

        mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
        snr = (ez.fnu/ez.efnu)[:,i]

        depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

        ax.axvline(depth, ls='dashed', c='royalblue')

        sanity = ez.cat['use_phot'] == 1
        sanity &= mag <= depth
        sanity &= ~np.isnan(mag)

        ebins = histedges_equalN(mag[sanity], 15)
        nbins, bin_centers, bmed, bstd = binned_med(mag[sanity], test[sanity, i], bins=ebins)

        ax.scatter(mag[sanity], test[sanity, i], c='grey', s=1, alpha=0.1)
        ax.plot(bin_centers, bmed, c='royalblue')
        ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

        delta = np.nanmedian(test[sanity,i])
        ax.text(0.65, 0.8, f'$\Delta={delta:2.3f}$', transform=ax.transAxes, fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_SCIREADY_{APERSIZE}_{fname}.pdf'))

#