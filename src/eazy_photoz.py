import os, sys
import numpy as np
import matplotlib.pyplot as plt

import eazy
print(eazy.__version__)

# Symlink templates & filters from the eazy-code repository
envpath = os.getenv('EAZYCODE')
if envpath is None:
    envpath = os.path.join(eazy.utils.path_to_eazy_data(), 'eazy-photoz')
print('EAZYCODE = '+envpath)

try:
    eazy.symlink_eazy_inputs()
except:
    Warning('Could not add symlinks...might be OK.')

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
TEMPLATES = sys.argv[5]

from config import DIR_CATALOGS, DET_TYPE, TRANSLATE_FNAME, TARGET_ZP, ITERATE_ZP, FILTERS, MATCH_BAND, PROJECT, VERSION

FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

translate_file = os.path.join(DIR_CONFIG, TRANSLATE_FNAME)

params = {}

str_aper = APERSIZE.replace('_', '')
if len(str_aper) == 2:
    str_aper += '0' # 07 -> 070

params['CATALOG_FILE'] = os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_D{str_aper}_CATALOG.fits")
params['MAIN_OUTPUT_FILE'] = os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_D{str_aper}_CATALOG.{TEMPLATES}.eazypy")

params['APPLY_PRIOR'] = 'n'
params['PRIOR_ABZP'] = TARGET_ZP
params['MW_EBV'] = 0.0
params['CAT_HAS_EXTCORR'] = 'y'
params['N_MIN_COLORS'] = 2
params['Z_COLUMN'] = 'z_phot'
params['USE_ZSPEC_FOR_REST'] = 'n'
params['SYS_ERR'] = 0.05


params['Z_MAX'] = 15  # 30.
params['Z_STEP'] = 0.01 # 0.005

if TEMPLATES == 'fsps_full':
    params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'
elif TEMPLATES == 'sfhz':
    params['TEMPLATES_FILE'] = 'templates/sfhz/carnall_sfhz_13.param'
elif TEMPLATES == 'sfhz_blue':
    params['TEMPLATES_FILE'] = 'templates/sfhz/blue_sfhz_13.param'

params['VERBOSITY'] = 1

# from astropy.cosmology import WMAP9

ez = eazy.photoz.PhotoZ(param_file=None,  #cosmology=WMAP9,
                              translate_file=translate_file,
                              zeropoint_file=None, params=params,
                              load_prior=False, load_products=False)

print(ez.cosmology)

NITER = 5
NBIN = np.minimum(ez.NOBJ//100, 180)

ez.cat = ez.cat.filled(-99)

ez.param.params['VERBOSITY'] = 1.

if ITERATE_ZP:
    for iter in range(NITER):
        print('Iteration: ', iter)

        sn = ez.fnu/ez.efnu
        clip = (sn > 10).sum(axis=1) > 6 # Generally make this higher to ensure reasonable fits
        clip &= ez.cat['use_phot'] == 1
        ez.iterate_zp_templates(idx=ez.idx[clip], update_templates=False,
                                update_zeropoints=True, iter=iter, n_proc=8,
                                save_templates=False, error_residuals=(iter > 0),
                                NBIN=NBIN, get_spatial_offset=False)


# Turn off error corrections derived above
# ez.efnu = ez.efnu_orig
ez.set_sys_err(positive=True)

# Full catalog
sample = ez.idx # all
# sample = np.isfinite(ez.ZSPEC)

ez.fit_parallel(sample, n_proc=8, prior=False, beta_prior=False)

ez.zphot_zspec(include_errors=True, zmax=6.5, selection=ez.cat['use_phot']==1)
fig = plt.gcf()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}.photoz-specz.pdf'))


zout, hdu = ez.standard_output(rf_pad_width=0.5, rf_max_err=2, n_proc=2,
                                 prior=False, beta_prior=False)

ez.fit_phoenix_stars()
star_coln = ['star_chi2', 'star_min_ix', 'star_min_chi2', 'star_min_chinu']

for coln in star_coln:
    zout[coln] = ez.__dict__[coln]
    print(coln)

u_v = -2.5*np.log10(zout['restU'] / zout['restV'])
v_j = -2.5*np.log10(zout['restV'] / zout['restJ'])

def uvj_sel(v_j):
    ret = 0.72*v_j + 0.75
    ret[v_j >= 1.6] = 1E20
    ret[v_j < 0.9] = 1.4
    return ret

clas = np.nan * np.ones(len(zout), dtype=int)
out = uvj_sel(v_j)
clas[u_v >= out] = 1 # QG
clas[u_v < out] = 0 # SF
clas[(u_v > 2.4) | (u_v < 0.0) | (v_j < -0.6)  | (v_j > 2.0)] = -1
clas[np.isnan(u_v) | np.isnan(v_j)] = -1

zout['u_v'] = u_v
zout['v_j'] = v_j
zout['uvj_class'] = clas

zout['flag_eazy'] = np.where(np.isnan(zout['mass']) | (zout['z_phot_chi2'] > 300), 1, 0)

zout.write(os.path.join(FULLDIR_CATALOGS, f'{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}.zout.fits'), overwrite=True)

import eazy.hdf5
eazy.hdf5.write_hdf5(ez, h5file=ez.param['MAIN_OUTPUT_FILE'] + '.h5')

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
        sanity &= snr > 3
        sanity &= ~np.isnan(mag)

        test_ls.append(test[sanity, i])

    axt.boxplot(test_ls, vert=True, positions=ez.pivot*1e-4, widths=0.1, labels=FILTERS, flierprops={'marker':'.', 'markersize':2, 'alpha':0.1})
    axt.set(xlim=(0.05, 5))
    ax.set(xlim=(0.1, 5), xlabel='Observed Wavelength ($\mu$m)', ylabel=ylabel)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}_{fname}.pdf'))


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

    from webb_tools import histedges_equalN, binned_med

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

        # mag = TARGET_ZP - 2.5*np.log10(ez.fnu[:,i])
        snr = (ez.fnu/ez.efnu)[:,i]

        # depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

        sanity = ez.cat['use_phot'] == 1
        sanity &= snr > 5
        # sanity &= ~np.isnan(mag)

        ebins = histedges_equalN(ez.zbest[sanity], 15)
        nbins, bin_centers, bmed, bstd = binned_med(ez.zbest[sanity], test[sanity, i], bins=ebins)

        ax.scatter(ez.zbest[sanity], test[sanity, i], c='grey', s=1, alpha=0.1)
        ax.plot(bin_centers, bmed, c='royalblue')
        ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

        delta = np.nanmedian(test[sanity,i])
        ax.text(0.65, 0.8, f'$\Delta={delta:2.3f}$', transform=ax.transAxes, fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}_{fname}.pdf'))



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
    axes[-1].set(xlim=(19, 31))
    fig.suptitle(ylabel, y=0.99, fontsize=20)

    from webb_tools import histedges_equalN, binned_med

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

        # depth = TARGET_ZP - 2.5*np.log10( np.median(ez.fnu[:,i][(snr > 2.9) & (snr < 3.1)]) )

        ax.axvline(depth, ls='dashed', c='royalblue')

        sanity = ez.cat['use_phot'] == 1
        sanity &= snr > 5
        # sanity &= ~np.isnan(mag)

        ebins = histedges_equalN(mag[sanity], 15)
        nbins, bin_centers, bmed, bstd = binned_med(mag[sanity], test[sanity, i], bins=ebins)

        ax.scatter(mag[sanity], test[sanity, i], c='grey', s=1, alpha=0.1)
        ax.plot(bin_centers, bmed, c='royalblue')
        ax.fill_between(bin_centers, bstd[0], bstd[1], color='royalblue', alpha=0.2)

        delta = np.nanmedian(test[sanity,i])
        ax.text(0.65, 0.8, f'$\Delta={delta:2.3f}$', transform=ax.transAxes, fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}_{fname}.pdf'))

# Basic properties

import numpy as np

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(4*5, 5))

qg = zout['uvj_class'] == 1
sanity = ez.cat['use_phot'] == 1

# z vs. M
axes[0].scatter(zout['z_phot'][sanity], np.log10(zout['mass'][sanity]), s=3, c='grey', alpha=0.3)
axes[0].scatter(zout['z_phot'][qg & sanity], np.log10(zout['mass'][qg & sanity]), s=3, c='orange')
axes[0].set(xlabel='$z_{\\rm phot}$', ylabel='Log$_{10}\,\mathcal{M}\,(\mathcal{M}_\odot)$')

# M vs. SFR
axes[1].scatter(np.log10(zout['mass'][sanity]), np.log10(zout['sfr']/zout['mass'])[sanity], s=3, c='grey', alpha=0.3)
axes[1].scatter(np.log10(zout['mass'][qg & sanity]), np.log10(zout['sfr'][qg & sanity]/zout['mass'][qg & sanity]), s=3, c='orange')
axes[1].set(xlabel='Log$_{10}\,\mathcal{M}\,(\mathcal{M}_\odot)$', ylabel='Log$_{10}\,{\\rm sSFR}}\,(\mathcal{M}_\odot\,{\\rm yr}^{-1})$')

# UVJ
axes[2].scatter(zout['v_j'][sanity], zout['u_v'][sanity], s=3, c='grey', alpha=0.3)
axes[2].scatter(zout['v_j'][qg & sanity], zout['u_v'][qg & sanity], s=3, c='orange')
axes[2].set(xlabel='$V-J$', ylabel='$U-V$', xlim=(-0.6, 2.0), ylim=(0, 2.4))

# MATCH_BAND
bins = np.arange(17, 30, 0.5)
axes[3].hist(TARGET_ZP - 2.5*np.log10(ez.cat['f_'+MATCH_BAND])[sanity], bins=bins, color='grey')
axes[3].hist(TARGET_ZP - 2.5*np.log10(ez.cat['f_'+MATCH_BAND])[qg & sanity], bins=bins, color='orange')
axes[3].set(xlabel=MATCH_BAND.upper())
axes[3].semilogy()

fig.tight_layout()
fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{PROJECT}_v{VERSION}_{DET_NICKNAME.split("_")[0]}_K{KERNEL}_D{str_aper}_CATALOG_{TEMPLATES}_properties_scatter.pdf'))
