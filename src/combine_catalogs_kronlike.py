from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table, hstack, Column, vstack, MaskedColumn
import numpy as np
import os, sys, glob
import sfdmap
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.svo_fps import SvoFps
import extinction
import matplotlib.pyplot as plt
from webb_tools import psf_cog, fit_apercurve
from astropy.convolution import convolve

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import FILTERS, DIR_SFD, APPLY_MWDUST, DIR_CATALOGS, DIR_OUTPUT, \
    MATCH_BAND, PIXEL_SCALE, PHOT_APER, DIR_KERNELS, DIR_PSFS, FIELD, ZSPEC, \
    MAX_SEP, SCI_APER, MAKE_SCIREADY_ALL, TARGET_ZP, ZCONF, ZRA, ZDEC, ZCOL, FLUX_UNIT, \
    PS_WEBB_FLUXRATIO, PS_WEBB_FLUXRATIO_RANGE, PS_WEBB_FILT, PS_WEBB_MAGLIMIT, PS_WEBB_APERSIZE, \
    PS_HST_FLUXRATIO, PS_HST_FLUXRATIO_RANGE, PS_HST_FILT, PS_HST_MAGLIMIT, PS_HST_APERSIZE, \
    BP_FLUXRATIO, BP_FLUXRATIO_RANGE, BP_FILT, BP_MAGLIMIT, BP_APERSIZE, RA_RANGE, DEC_RANGE, \
    GAIA_ROW_LIMIT, GAIA_XMATCH_RADIUS, FN_BADWHT, SATURATEDSTAR_MAGLIMIT, SATURATEDSTAR_FILT, \
    FN_EXTRABAD, EXTRABAD_XMATCH_RADIUS, EXTRABAD_LABEL, BK_MINSIZE, BK_SLOPE, PATH_BADOBJECT, \
    GLASS_MASK, SATURATEDSTAR_APERSIZE, PS_WEBB_USE, PS_HST_USE, GAIA_USE, BADWHT_USE, EXTRABAD_USE, \
    BP_USE, BADOBJECT_USE, PHOT_USEMASK, PROJECT, VERSION, PSF_FOV, USE_COMBINED_KRON_IMAGE, KRON_COMBINED_BANDS, \
    XCAT_FILENAME, XCAT_NAME


DET_NICKNAME =  sys.argv[2] #'LW_f277w-f356w-f444w'
KERNEL = sys.argv[3] #'f444w'

DET_TYPE = 'noise-equal'
FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

def DIR_KERNEL(band):
    return glob.glob(os.path.join(DIR_KERNELS, f'{KERNEL}*/{band.lower()}_kernel.fits'))[0]
stats = np.load(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_emptyaper_stats.npy'), allow_pickle=True).item()

FNAME_REF_PSF = f'{DIR_PSFS}/{MATCH_BAND.lower()}_psf_unmatched.fits'

def sigma_aper(filter, weight, apersize=0.7):
    # Equation 5
    # apersize = str(apersize).replace('.', '_') + 'arcsec'
    sigma_nmad_filt = stats[filter.lower()][apersize]['fit_std']
    # sigma_nmad_filt = ERROR_TABLE[f'e{apersize}'][ERROR_TABLE['filter']==filter.lower()][0]
    # g_i = 1.*2834.508 # here's to hoping.  EFFECTIVE GAIN!
    fluxerr = sigma_nmad_filt / np.sqrt(weight)  #+ (flux_aper / g_i)
    fluxerr[weight<=0] = np.inf
    return fluxerr

def sigma_total(sigma_aper, tot_cor):
    # equation 6
    return sigma_aper * tot_cor

def sigma_ref_total(sigma1, alpha, beta, kronrad_circ, wht_ref):
    # equation 7
    term1 = (sigma1 * alpha * (np.pi * kronrad_circ**2)**(beta/2.)) / np.sqrt(wht_ref)
    term1[wht_ref<=0] = np.inf
    # g_ref = 1.
    # term2 = flux_refauto / g_ref
    return term1 # + term2)

# def sigma_full(sigma_total, sigma_ref_total, sigma_total_ref):
#     # equation 8
#     sig_full = sigma_total**2 + sigma_ref_total**2 - sigma_total_ref**2
#     # print(np.sum(sig_full < 0) / len(sig_full))
#     return np.sqrt(sig_full)

def flux_ref_total(flux_ref_auto, frac):
    # equation 9
    return flux_ref_auto  / frac

def flux_total(flux_aper, tot_cor):
    # equation 10
    return flux_aper * tot_cor


# loop over filters
KRON_MATCH_BAND = None
USE_FILTERS = FILTERS
if (KERNEL != 'None') & (USE_COMBINED_KRON_IMAGE):
    KRON_MATCH_BAND = '+'.join(KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]])
    if '+' not in KRON_MATCH_BAND:
        KRON_MATCH_BAND = 'sb-' + KRON_MATCH_BAND
    USE_FILTERS = [KRON_MATCH_BAND, ] + list(FILTERS)

for filter in USE_FILTERS:
    filename = os.path.join(FULLDIR_CATALOGS, f'{filter}_{DET_NICKNAME}_K{KERNEL}_PHOT_CATALOG.fits')
    if not os.path.exists(filename):
        print(f'ERROR :: {filename} not found!')
        sys.exit()

    cat = Table.read(filename)
    # print(filter)

    # rename columns if needed:
    for coln in cat.colnames:
        if 'RADIUS' in coln or 'APER' in coln or 'FLAG' in coln or 'AUTO' in coln or 'WHT' in coln or 'ISO' in coln:

            newcol = f'{filter}_{coln}'.replace('.', '_')
            # print(f'   {cat[coln].name} --> {newcol}')
            cat[coln].name = newcol

            try:
                cat[newcol] = cat[newcol].filled(np.nan)
                # print('Filled with NaN!')
            except:
                pass

    if filter == USE_FILTERS[0]:
        maincat = cat
    else:
        newcols = [coln for coln in cat.colnames if coln not in maincat.colnames]
        maincat = hstack([maincat, cat[newcols]])

outfilename = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_COMBINED_CATALOG.fits')

for filter in USE_FILTERS:
    filename = os.path.join(FULLDIR_CATALOGS, f'{filter}_{DET_NICKNAME}_KNone_PHOT_CATALOG.fits')
    if not os.path.exists(filename):
        print(f'ERROR :: {filename} not found!')
        sys.exit()

    cat = Table.read(filename)
    # print(filter)

    # rename columns if needed:
    for coln in cat.colnames:
        if 'RADIUS' in coln or 'APER' in coln or 'FLAG' in coln or 'AUTO' in coln or 'WHT' in coln or 'ISO' in coln:

            newcol = f'{filter}_{coln}'.replace('.', '_')
            # print(f'   {cat[coln].name} --> {newcol}')
            cat[coln].name = newcol

            try:
                cat[newcol] = cat[newcol].filled(np.nan)
                # print('Filled with NaN!')
            except:
                pass

    if filter == USE_FILTERS[0]:
        maincat_unmatched = cat
    else:
        newcols = [coln for coln in cat.colnames if coln not in maincat_unmatched.colnames]
        maincat_unmatched = hstack([maincat_unmatched, cat[newcols]])

outfilename = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_COMBINED_CATALOG.fits')

# print(maincat.colnames)

if USE_COMBINED_KRON_IMAGE:
    KRON_MATCH_BAND = '+'.join(KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]])
    if '+' not in KRON_MATCH_BAND:
        KRON_MATCH_BAND = 'sb-' + KRON_MATCH_BAND
else:
    KRON_MATCH_BAND = MATCH_BAND # behaves as usual with a single ref band

# grab MATCH_BAND PSF convolved to kernel
psfmodel = fits.getdata(FNAME_REF_PSF)
if KERNEL == MATCH_BAND:
    conv_psfmodel = psfmodel.copy()
else:
    kernel = fits.getdata(DIR_KERNEL(MATCH_BAND))
    conv_psfmodel = convolve(psfmodel, kernel)

mask = ''
if PHOT_USEMASK:
    mask = '_masked'


# Get some static refband stuff
plotname = os.path.join(FULLDIR_CATALOGS, f'figures/aper_{KRON_MATCH_BAND}_nmad.pdf')
p, pcov, sigma1 = fit_apercurve(stats[KRON_MATCH_BAND], plotname=plotname, stat_type=['fit_std'], pixelscale=PIXEL_SCALE)
alpha, beta = p['fit_std']
sig1 = sigma1['fit_std']
wht_ref = maincat[f'{KRON_MATCH_BAND}_SRC_MEDWHT']
f_ref_auto = maincat[f'{KRON_MATCH_BAND}_FLUX_AUTO{mask}'].copy()
sel_badkron = (f_ref_auto <= 0) | ~np.isfinite(f_ref_auto)
sel_badkron |= ~np.isfinite(wht_ref) | (wht_ref <= 0)
# sel_badkron |= (maincat['flag'] > 0)

# some extra columns
maincat['iso_area'] = maincat['tnpix'] * PIXEL_SCALE**2
maincat['iso_area'].unit = u.arcsec**2


# Check if object is really bright in its group, if so then sel_badkron = False
isofluxes = maincat[f'{KRON_MATCH_BAND}_FLUX_ISO']
is_dominant = np.ones(len(isofluxes), dtype=bool) # assume dominance. Things without friends should not be flagged as blends.
import pickle
assoc = pickle.load(open(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_friends.pickle'), 'rb'))
for i, id in enumerate(maincat['ID']):
    if len(assoc[id]) == 0: continue # it has no friends :(
    friends = np.isin(maincat['ID'], assoc[id])
    is_dominant[i] = np.all(isofluxes[i] > isofluxes[friends])
kronrad_area = np.pi * (maincat[f'{KRON_MATCH_BAND}_KRON_RADIUS_CIRC{mask}'] * PIXEL_SCALE)**2

# is_dominant = np.zeros(len(is_dominant), dtype=bool) # !!!!!!!!!!!!!

sel_badkron[(maincat['flag'] > 0)] = True
sel_badkron[sel_badkron & (kronrad_area < maincat[f'iso_area'])] = False
sel_badkron[sel_badkron & (is_dominant & (kronrad_area < 1.5*maincat[f'iso_area']))] = False
print(f'Found {np.sum(sel_badkron)} objects with unreliable kron radii (e.g. blends)')

for filter in USE_FILTERS:
    relwht = maincat[f'{filter}_SRC_MEDWHT'] / maincat[f'{filter}_MAX_WHT']
    # relwht[~np.isfinite(wht_ref) | np.isnan(wht_ref)] = np.nan
    newcoln = f'{filter}_RELWHT'
    maincat.add_column(Column(relwht, newcoln))

for apersize in PHOT_APER:
    str_aper = str(apersize).replace('.', '_')

    # use KRON_MATCH_BAND Kron to correct to total fluxes and ferr
    f_ref_auto = maincat[f'{KRON_MATCH_BAND}_FLUX_AUTO{mask}'].copy()
    kronrad_circ = maincat[f'{KRON_MATCH_BAND}_KRON_RADIUS_CIRC{mask}'].copy()
    kronrad = maincat[f'{KRON_MATCH_BAND}_KRON_RADIUS{mask}'].copy()

    f_ref_aper = maincat[f'{KRON_MATCH_BAND}_FLUX_APER{str_aper}'].copy()

    sel_badkron |= np.isnan(f_ref_aper)
    use_circle = (kronrad_circ < (apersize / PIXEL_SCALE / 2.)) | (f_ref_auto <= f_ref_aper) | sel_badkron # either too small (not flagged) OR not reliable.

    kronrad_circ[use_circle] = apersize / PIXEL_SCALE / 2.
    # print(apersize, np.sum(use_circle), np.min(kronrad_circ), apersize / PIXEL_SCALE / 2.)
    f_ref_auto[use_circle] = f_ref_aper[use_circle]
    f_ref_auto[~np.isfinite(maincat[f'{KRON_MATCH_BAND}_RELWHT'])] = np.nan # if you don't have a weight then you don't have a flux_err, so you don't have flux, and so you shouldn't show auto either.

    psffrac_ref_auto = psf_cog(conv_psfmodel, MATCH_BAND.upper(), nearrad = kronrad_circ) # in pixels
    # F160W kernel convolved MATCH_BAND PSF + missing flux from F160W beyond 2" radius
    f_ref_total = f_ref_auto / psffrac_ref_auto # equation 9
    # if apersize == PHOT_APER[0]:
    newcoln =f'{KRON_MATCH_BAND}_FLUX_REF_AUTO_APER{str_aper}'
    maincat.add_column(Column(f_ref_auto, newcoln))

    min_corr = 1. / psf_cog(conv_psfmodel, MATCH_BAND.upper(), nearrad=(apersize / PIXEL_SCALE / 2.)) # defaults to EE(<R_aper)
    tot_corr = f_ref_total / f_ref_aper

    use_circle |= tot_corr < min_corr
    use_circle |= ~np.isfinite(tot_corr)
    sel_badkron |= ~np.isfinite(tot_corr)
    tot_corr[use_circle] = min_corr

    kronrad[use_circle] = np.nan #
    kronrad_circ[use_circle] = np.nan #

    # idx = np.argwhere(tot_corr < min_corr)
    # for i in idx:
    #     print(i[0], min_corr, np.array(tot_corr[i]), np.array(f_ref_auto / f_ref_aper / psffrac_ref_auto)[i], np.array(1/ psffrac_ref_auto)[i])
    #           #np.array(f_ref_total[i]), np.array(f_ref_auto[i]), np.array(f_ref_aper[i]), psffrac_ref_auto[i], np.array(kronrad_circ*2*PIXEL_SCALE)[i])
    # print(np.sum(tot_corr < min_corr))
    # # print(min_corr, np.nanmin(tot_corr), f_ref_total[np.argmin(tot_corr)], f_ref_aper[np.argmin(tot_corr)])
    if np.any(tot_corr < min_corr):
        print(f'ERROR: A TOTAL CORRECTION IS LESS THAN THE MINIMUM POSSIBLE AT {apersize}')
        raise

    maincat.add_column(MaskedColumn(tot_corr, f'TOTAL_CORR_APER{str_aper}', mask=np.isnan(tot_corr)))

    sig_ref_aper = sigma_aper(KRON_MATCH_BAND.upper(), wht_ref, apersize) # sig_aper,KRON_MATCH_BAND
    sig_total_ref = sigma_total(sig_ref_aper, tot_corr) # sig_total,KRON_MATCH_BAND
    sig_ref_total = sigma_ref_total(sig1, alpha, beta, kronrad_circ, wht_ref)

    newcoln =f'{KRON_MATCH_BAND}_USE_CIRCLE_APER{str_aper}'
    maincat.add_column(Column(use_circle.astype(int), newcoln))
    newcoln =f'{KRON_MATCH_BAND}_FLAG_KRON_APER{str_aper}'
    maincat.add_column(Column(sel_badkron.astype(int), newcoln))
    newcoln =f'{KRON_MATCH_BAND}_KRON_RADIUS_CIRC_APER{str_aper}'
    maincat.add_column(Column(kronrad_circ, newcoln))
    newcoln =f'{KRON_MATCH_BAND}_KRON_RADIUS_APER{str_aper}'
    maincat.add_column(Column(kronrad, newcoln))
    newcoln =f'{KRON_MATCH_BAND}_PSFFRAC_REF_AUTO_APER{str_aper}'
    maincat.add_column(Column(psffrac_ref_auto, newcoln))
    newcoln =f'{KRON_MATCH_BAND}_FLUX_REFTOTAL_APER{str_aper}'
    maincat.add_column(Column(f_ref_total, newcoln))
    newcoln =f'{KRON_MATCH_BAND}_FLUXERR_REFTOTAL_MINDIAM{str_aper}'
    maincat.add_column(Column(sig_ref_total, newcoln))

    for filter in USE_FILTERS:
        f_aper =maincat[f'{filter}_FLUX_APER{str_aper}']
        f_total = flux_total(f_aper, tot_corr)  # f_aper * tot_corr
        wht = maincat[f'{filter}_SRC_MEDWHT']
        # medwht = maincat[f'{filter}_MED_WHT']

        # get the flux uncertainty in the aperture for this band
        sig_aper = sigma_aper(filter, wht, apersize)
        sig_aper[np.isnan(f_aper)] = np.nan
        f_aper[np.isnan(sig_aper)] = np.nan
        f_total[np.isnan(sig_aper) | np.isnan(f_aper)] = np.nan
        # do again for each aperture
        sig_total = sigma_total(sig_aper, tot_corr)
        # sig_full = sigma_full(sig_total, sig_ref_total, sig_total_ref)

        # add new columns
        newcoln = f'{filter}_FLUX_APER{str_aper}_COLOR'
        maincat.add_column(Column(f_aper, newcoln))
        newcoln =f'{filter}_FLUXERR_APER{str_aper}_COLOR'
        maincat.add_column(Column(sig_aper, newcoln))

        newcoln = f'{filter}_FLUX_APER{str_aper}_TOTAL'
        maincat.add_column(Column(f_total, newcoln))
        newcoln = f'{filter}_FLUXERR_APER{str_aper}_TOTAL'
        maincat.add_column(Column(sig_total, newcoln))

        # newcoln =f'{filter}_FLUXERR_APER{str_aper}_FULL'
        # maincat.add_column(Column(sig_full, newcoln))


# ADD SFD maps (2011 scales by 0.86, which is default. otherwise use scaling=1.0)
m = sfdmap.SFDMap(DIR_SFD)
ebmv = m.ebv(maincat['RA'], maincat['DEC'])
maincat.add_column(Column(ebmv, name='EBV'), 1+np.where(np.array(maincat.colnames) == 'DEC')[0][0])


if APPLY_MWDUST == 'MEDIAN':
    Av = np.median(ebmv)*3.1
elif APPLY_MWDUST == 'VAR':
    Av = 3.1 * ebmv

# Perform a MW correction (add new columns to the master)
if APPLY_MWDUST is not None:
    filter_table = vstack([SvoFps.get_filter_list('JWST'),\
                        SvoFps.get_filter_list('HST')])
    filter_pwav = OrderedDict()
    print('Building directory of pivot wavelengths')
    for filter in FILTERS:
        filter_pwav[filter] = np.nan # ensures the order
        for i, tryfilt in enumerate(filter_table['filterID']):
            if filter == 'f410m':
                if 'NIRCam' in tryfilt:
                    if tryfilt.endswith(filter.upper()):
                        filter_pwav[filter] = filter_table[i]['WavelengthPivot']
            if filter != 'f410m':
                if 'ACS' in tryfilt or 'WFC3' in tryfilt or 'NIRCam' in tryfilt: # ADD OTHERS HERE
                    if tryfilt.endswith(filter.upper()):
                        filter_pwav[filter] = filter_table[i]['WavelengthPivot'] # angstrom
                        # print(filter, filter_table[i]['filterID'], filter_pwav[filter])

    atten_mag = extinction.fm07(np.array(list(filter_pwav.values())), Av) # atten_mag in magnitudes from Fitzpatrick + Massa 2007
    atten_factor = 10** (-0.4 * atten_mag) # corresponds in order to FILTERS
    for i, filter in enumerate(FILTERS):
        print(f'{filter} ::  {atten_factor[i]:2.5f}x or {atten_mag[i]:2.5f} AB')

    print('Applying Milky Way Attenuation correction (FM+07)')
    for coln in maincat.colnames:
        if 'RADIUS' in coln:
            continue
        filtname = coln.split('_')[0]
        if filtname in FILTERS:
            if 'FLUX' in coln:
                maincat[coln] /= atten_factor[np.array(FILTERS) == filtname][0]

            elif 'MAG' in coln:
                maincat[coln] -= atten_mag[np.array(FILTERS) == filtname][0]

# low-snr flag
# for coln in maincat.colnames: print(coln)
str_aper = str(SCI_APER).replace('.', '_')
snr_ref = maincat[f'{KRON_MATCH_BAND}_FLUX_APER{str_aper}_COLOR'] / maincat[f'{KRON_MATCH_BAND}_FLUXERR_APER{str_aper}_COLOR']
snr_ref[maincat[f'{KRON_MATCH_BAND}_FLUXERR_APER{str_aper}_COLOR']<=0] = -1
SEL_LOWSNR = (snr_ref < 3) | np.isnan(maincat[f'{KRON_MATCH_BAND}_RELWHT'])
print(f'Flagged {np.sum(SEL_LOWSNR)} objects as having low SNR < 3')
maincat.add_column(Column(SEL_LOWSNR.astype(int), name='lowsnr_flag'))

SEL_STAR = np.zeros(len(maincat),dtype=bool)

# Select in F160W
if PS_HST_USE:
    str_aper = str(PS_HST_APERSIZE).replace('.', '_')
    mag_hst = TARGET_ZP - 2.5*np.log10(maincat[f'{PS_HST_FILT}_FLUX_APER{str_aper}_TOTAL'])
    size_hst = maincat_unmatched[f'{PS_HST_FILT}_FLUX_APER{str(PS_HST_FLUXRATIO[0]).replace(".", "_")}']  \
                    / maincat_unmatched[f'{PS_HST_FILT}_FLUX_APER{str(PS_HST_FLUXRATIO[1]).replace(".", "_")}']

    SEL_HST = (size_hst > PS_HST_FLUXRATIO_RANGE[0]) & (size_hst < PS_HST_FLUXRATIO_RANGE[1]) & (mag_hst < PS_HST_MAGLIMIT)
    print(f'Flagged {np.sum(SEL_HST)} objects as point-like (stars) from {PS_HST_FILT}')
    maincat.add_column(Column(SEL_HST.astype(int), name='star_hst_flag'))
    SEL_STAR |= SEL_HST

# star-galaxy flag
# Select from Webb band
if PS_WEBB_USE:
    str_aper = str(PS_WEBB_APERSIZE).replace('.', '_')
    mag = TARGET_ZP - 2.5*np.log10(maincat[f'{PS_WEBB_FILT}_FLUX_APER{str_aper}_TOTAL'])
    size = maincat_unmatched[f'{PS_WEBB_FILT}_FLUX_APER{str(PS_WEBB_FLUXRATIO[0]).replace(".", "_")}']  \
                    / maincat_unmatched[f'{PS_WEBB_FILT}_FLUX_APER{str(PS_WEBB_FLUXRATIO[1]).replace(".", "_")}']
    SEL_WEBB = (size > PS_WEBB_FLUXRATIO_RANGE[0]) & (size < PS_WEBB_FLUXRATIO_RANGE[1]) & (mag < PS_WEBB_MAGLIMIT)
    print(f'Flagged {np.sum(SEL_WEBB)} objects as point-like (stars) from {PS_WEBB_FILT}')
    maincat.add_column(Column(SEL_WEBB.astype(int), name='star_webb_flag'))
    fsize = maincat[f'{PS_WEBB_FILT}_FLUX_RADIUS_FRAC0_5'] * PIXEL_SCALE
    SEL_STAR |= SEL_WEBB

# GAIA selection
if GAIA_USE:
    fn_gaia = os.path.join(DIR_OUTPUT, 'gaia.fits')
    if os.path.exists(fn_gaia):
        tab_gaia = Table.read(fn_gaia)
    else:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = GAIA_ROW_LIMIT  # Ensure the default row limit.
        cra, cdec = np.mean(RA_RANGE)*u.deg, np.mean(DEC_RANGE)*u.deg
        coord = SkyCoord(ra=cra, dec=cdec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity(0.15, u.deg)
        j = Gaia.cone_search_async(coord, radius)
        gaia = j.get_results()
        # gaia.pprint()
        tab_gaia = Table(gaia)['solution_id', 'source_id', 'ra', 'dec', 'ref_epoch', 'pmra', 'pmdec']
        tab_gaia.write(os.path.join(DIR_OUTPUT, 'gaia.fits'), format='fits', overwrite=True)

    from webb_tools import crossmatch
    mCATALOG_gaia, mtab_gaia = crossmatch(maincat, tab_gaia, [GAIA_XMATCH_RADIUS,])
    has_pm = np.hypot(mtab_gaia['pmra'], mtab_gaia['pmdec']) > 0
    SEL_GAIA = np.isin(maincat['ID'], mCATALOG_gaia['ID'][has_pm])
    print(f'Flagged {np.sum(SEL_GAIA)} objects as point-like (stars) from GAIA')
    maincat.add_column(Column(SEL_GAIA.astype(int), name='star_gaia_flag'))
    SEL_STAR |= SEL_GAIA

# Select by WEBB weight saturation (for stars AND bad pixels)
if BADWHT_USE:
    weightmap = fits.getdata(FN_BADWHT) # this is lazy, but OK.
    str_aper = str(SATURATEDSTAR_APERSIZE).replace('.', '_')
    mag_sat = TARGET_ZP - 2.5*np.log10(maincat[f'{SATURATEDSTAR_FILT}_FLUX_APER{str_aper}_TOTAL'])
    SEL_BADWHT = (weightmap[maincat['y'].astype(int).value, maincat['x'].astype(int).value] == 0)
    sw_wht = maincat[f'{SATURATEDSTAR_FILT}_SRC_MEDWHT'].copy()
    sw_wht[np.isnan(sw_wht)] = 0
    SEL_BADWHT[sw_wht <= 0] = 0
    maincat.add_column(Column(SEL_BADWHT.astype(int), name='bad_wht_flag'))
    SEL_SATSTAR = SEL_BADWHT & (mag_sat < SATURATEDSTAR_MAGLIMIT)
    maincat.add_column(Column(SEL_SATSTAR.astype(int), name='saturated_star_flag'))
    print(f'Flagged {np.sum(SEL_SATSTAR)} objects as saturated bright sources (stars) from {FN_BADWHT}')
    SEL_STAR |= SEL_BADWHT

print(f'Flagged {np.sum(SEL_STAR)} total objects as stars ({np.sum(SEL_STAR)/len(SEL_STAR)*100:2.2f}%)')
maincat.add_column(Column(SEL_STAR.astype(int), name='star_flag'))

SEL_GEN = SEL_LOWSNR | SEL_STAR


# bad pixel flag
if BP_USE:
    str_aper = str(BP_APERSIZE).replace('.', '_')
    BP_FILT_SEL = BP_FILT[DET_NICKNAME.split('_')[0]]
    mag_bp = TARGET_ZP - 2.5*np.log10(maincat[f'{BP_FILT_SEL}_FLUX_APER{str_aper}_TOTAL'])
    size_bp = maincat_unmatched[f'{BP_FILT_SEL}_FLUX_APER{str(BP_FLUXRATIO[0]).replace(".", "_")}']  \
                    / maincat_unmatched[f'{BP_FILT_SEL}_FLUX_APER{str(BP_FLUXRATIO[1]).replace(".", "_")}']
    SEL_LWBADPIXEL = (size_bp > BP_FLUXRATIO_RANGE[0]) & (size_bp < BP_FLUXRATIO_RANGE[1])
    SEL_LWBADPIXEL &= (mag_bp < BP_MAGLIMIT)
    print(f'Flagged {np.sum(SEL_LWBADPIXEL)} objects as bad pixels')
    maincat.add_column(Column(SEL_LWBADPIXEL.astype(int), name='bad_pixel_lw_flag'))
    # SEL_BADPIX = SEL_LWBADPIXEL | SEL_BADWHT
    SEL_GEN |= SEL_LWBADPIXEL

# diagnostic plot
if PS_WEBB_USE or PS_HST_USE or BP_USE:
    fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
    plot_elts = []
    if PS_WEBB_USE:
        axes[0].text(16, 0.8, f'{PS_WEBB_FILT}-selected stars', fontsize=15, color='royalblue')
        axes[0].hlines(PS_WEBB_FLUXRATIO_RANGE[0], 0, PS_WEBB_MAGLIMIT, alpha=0.5, color='royalblue')
        axes[0].hlines(PS_WEBB_FLUXRATIO_RANGE[1], 0, PS_WEBB_MAGLIMIT, alpha=0.5, color='royalblue')
        axes[0].vlines(PS_WEBB_MAGLIMIT, PS_WEBB_FLUXRATIO_RANGE[0], PS_WEBB_FLUXRATIO_RANGE[1], alpha=0.5, color='royalblue')
        axes[0].scatter(mag, size, s=3, alpha=0.2, c='grey')
        axes[0].invert_yaxis()
        axes[0].set(xlim=(15.2, 30.2), ylim=(0, 5), ylabel=('$\mathcal{F}\,'+f'({PS_WEBB_FLUXRATIO[0]} / {PS_WEBB_FLUXRATIO[1]})$'), xlabel=f'${PS_WEBB_FILT}$ Mag (AB)')
        plot_elts.append((SEL_WEBB, 'royalblue', None))

    if PS_HST_USE:
        axes[1].text(16, 0.8, 'f160w-selected stars', fontsize=15, color='orange')
        axes[1].hlines(PS_HST_FLUXRATIO_RANGE[0], 0, PS_HST_MAGLIMIT, alpha=0.5, color='orange')
        axes[1].hlines(PS_HST_FLUXRATIO_RANGE[1], 0, PS_HST_MAGLIMIT, alpha=0.5, color='orange')
        axes[1].vlines(PS_HST_MAGLIMIT, PS_HST_FLUXRATIO_RANGE[0], PS_HST_FLUXRATIO_RANGE[1], alpha=0.5, color='orange')
        axes[1].scatter(mag_hst, size_hst, s=3, alpha=0.2, c='grey')
        axes[1].invert_yaxis()
        axes[1].set(xlim=(15.2, 30.2), ylim=(0, 5), xlabel=f'${PS_HST_FILT}$ Mag (AB)', ylabel=('$\mathcal{F}\,'+f'({PS_HST_FLUXRATIO[0]} / {PS_HST_FLUXRATIO[1]})$'))
        plot_elts.append((SEL_HST, 'orange', None))

    if GAIA_USE:
        plot_elts.append((SEL_GAIA, 'green', 'GAIA stars'))

    if BADWHT_USE:
        plot_elts.append((SEL_SATSTAR, 'purple', f'{SATURATEDSTAR_FILT} saturated stars'))

    if BP_USE:
        axes[0].scatter(mag[SEL_LWBADPIXEL], size[SEL_LWBADPIXEL], s=12, alpha=0.8, c='firebrick', label='Bad LW pixel')
        axes[1].scatter(mag_hst[SEL_LWBADPIXEL], size_hst[SEL_LWBADPIXEL], s=12, alpha=0.8, c='firebrick')

        axes[2].scatter(mag, fsize, s=3, alpha=0.2, c='grey')
        axes[2].scatter(mag[SEL_LWBADPIXEL], fsize[SEL_LWBADPIXEL], s=12, alpha=0.8, c='firebrick', label='Bad LW pixel')
        axes[2].invert_yaxis()
        axes[2].set(xlim=(15.2, 32), ylim=(0, 0.4), ylabel=(f'${PS_WEBB_FILT}$ Flux Radius (arcsec)'), xlabel=f'${PS_WEBB_FILT}$ Mag (AB)')

        axes[3].text(17, 1.13, 'Bad Pixels in LW bands', fontsize=15, color='firebrick')
        axes[3].hlines(BP_FLUXRATIO_RANGE[0], 0, BP_MAGLIMIT, alpha=0.5, color='firebrick')
        axes[3].hlines(BP_FLUXRATIO_RANGE[1], 0, BP_MAGLIMIT, alpha=0.5, color='firebrick')
        axes[3].vlines(BP_MAGLIMIT, BP_FLUXRATIO_RANGE[0], BP_FLUXRATIO_RANGE[1], alpha=0.5, color='firebrick')

        axes[3].scatter(mag_bp, size_bp, s=3, alpha=0.2, c='grey')

        axes[3].scatter(mag_bp[SEL_LWBADPIXEL], size_bp[SEL_LWBADPIXEL], s=12, alpha=0.8, c='firebrick')
        axes[3].invert_yaxis()
        axes[3].set(xlim=(15.2, 30.2), ylim=(0, 2), ylabel=('$\mathcal{F}\,'+f'({BP_FLUXRATIO[0]} / {BP_FLUXRATIO[1]})$'), xlabel=f'${BP_FILT_SEL}$ Mag (AB)')


    for stars, color, label in plot_elts:
        if PS_WEBB_USE:
            axes[0].scatter(mag[stars], size[stars], s=12, alpha=1, c=color, label=label)
        if PS_HST_USE:
            axes[1].scatter(mag_hst[stars], size_hst[stars], s=12, alpha=1, c=color)
        if BP_USE:
            axes[2].scatter(mag[stars], fsize[stars], s=12, alpha=1, c=color)
            axes[3].scatter(mag_bp[stars], size_bp[stars], s=12, alpha=1, c=color)

    axes[0].legend(loc='upper left', ncol=1, fontsize=11, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FULLDIR_CATALOGS, f'figures/{DET_NICKNAME}_K{KERNEL}_star_id.png'))


# # bad kron radius flag
# krc = maincat[f'{KRON_MATCH_BAND}_KRON_RADIUS_CIRC{mask}'] * PIXEL_SCALE
# snr = (maincat[f'{KRON_MATCH_BAND}_FLUX_APER{str_aper}_COLOR']/maincat[f'{KRON_MATCH_BAND}_FLUXERR_APER{str_aper}_COLOR'])
# sel_badkron = (snr < BK_SLOPE*(krc - BK_MINSIZE))  #| (krc > 9)
# print(f'Flagged {np.sum(sel_badkron)} objects as having enormous kron radii for their SNR')
# maincat.add_column(Column(sel_badkron.astype(int), name='badkron_flag'))

# SEL_GEN |= sel_badkron # NO! leave this off. We want to use them still.
# low-weight source flag -  TODO

# HACK for glass
if GLASS_MASK is not None:
    SEL_BADGLASS = np.zeros(len(maincat), dtype=bool)
    in_glass = GLASS_MASK[maincat['y'].astype(int), maincat['x'].astype(int)] == 1
    print(f'{np.sum(in_glass)} objects in GLASS region')
    sw_wht = maincat['f200w_SRC_MEDWHT'].copy()
    sw_wht[np.isnan(sw_wht)] = 0
    sw_snr = maincat['f200w_FLUX_APER0_32_COLOR'] / maincat['f200w_FLUXERR_APER0_32_COLOR']
    SEL_BADGLASS = (((sw_snr < 3) & (sw_wht > 0)) | (sw_wht <= 0)) & in_glass
    print(f'Flagged {np.sum(SEL_BADGLASS)} objects as bad in the GLASS region')
    maincat.add_column(Column(SEL_BADGLASS.astype(int), name='badglass_flag'))
    SEL_GEN |= SEL_BADGLASS

# user supplied bad objects
if BADOBJECT_USE:
    SEL_BADOBJECT = np.zeros(len(maincat), dtype=bool)
    id_badobjects = np.load(PATH_BADOBJECT, allow_pickle=True).getitem()
    SEL_BADOBJECT = np.isin(maincat['ID'], id_badobjects)
    print(f'Flagged {np.sum(SEL_BADOBJECT)} objects as bad based on user-supplied ID')
    print(f'   based on file {PATH_BADOBJECT}')
    maincat.add_column(Column(SEL_BADOBJECT.astype(int), name='badobject_flag'))
    SEL_GEN |= SEL_BADOBJECT

# extra bad flag (e.g. bCG)
if EXTRABAD_USE:
    SEL_EXTRABAD = np.zeros(len(maincat), dtype=bool)
    tab_badobj = Table.read(FN_EXTRABAD)
    mCATALOG_badobj, mtab_badobj = crossmatch(maincat, tab_badobj, [EXTRABAD_XMATCH_RADIUS], plot=True)
    SEL_EXTRABAD = np.isin(maincat['ID'], mCATALOG_badobj['ID'])
    print(f'Flagged {np.sum(SEL_EXTRABAD)} objects as bad from the extra table ({EXTRABAD_LABEL})')
    maincat.add_column(Column(SEL_EXTRABAD.astype(int), name='extrabad_flag'))
    SEL_GEN |= SEL_EXTRABAD

# z-spec
ztable = Table.read(ZSPEC)
conf_constraint = np.ones(len(ztable), dtype=bool)
if ZCONF is not None:
    conf_constraint = np.isin(ztable[ZCONF[0]], np.array(ZCONF[1]))
ztable = ztable[conf_constraint & (ztable[ZDEC] >= -90.) & (ztable[ZDEC] <= 90.)]
zcoords = SkyCoord(ztable[ZRA]*u.deg, ztable[ZDEC]*u.deg)
catcoords = SkyCoord(maincat['RA'], maincat['DEC'])
idx, d2d, d3d = catcoords.match_to_catalog_sky(zcoords)
max_sep = MAX_SEP
sep_constraint = d2d < max_sep
print(f'Matched to {np.sum(sep_constraint)} objects with spec-z')

maincat.add_column(Column(d2d.to(u.arcsec), name='z_spec_radius'))
for colname in ztable.colnames:
    filler = np.zeros(len(maincat), dtype=ztable[colname].dtype)
    try:
        np.nan * filler
    except:
        pass
    filler[sep_constraint] = ztable[idx[sep_constraint]][colname]
    if colname == ZCOL:
        colname = 'z_spec'
        filler[filler<=0] = np.nan
    else:
        colname = f'z_spec_{colname}'
    maincat.add_column(Column(filler, name=colname))

# use flag (minimum SNR cut + not a star)
use_phot = np.zeros(len(maincat)).astype(int)
use_phot[~SEL_LOWSNR] = 1
use_phot[SEL_GEN] = 0
print(f'Flagged {np.sum(use_phot)} objects as reliable ({np.sum(use_phot)/len(use_phot)*100:2.1f}%)')
maincat.add_column(Column(use_phot, name='use_phot'))
# use 1 only

# Spit it out!
from datetime import date
today = date.today().strftime("%d/%m/%Y")
maincat.meta['CREATED'] = today
maincat.meta['MW_CORR'] = str(APPLY_MWDUST)
maincat.meta['KERNEL'] = KERNEL
maincat.meta['PHOT_ZP'] = TARGET_ZP
maincat.meta['PHOT_UNIT'] = FLUX_UNIT
maincat.meta['PIXSCALE'] = PIXEL_SCALE
maincat.meta['WEBBSTARFILT'] = PS_WEBB_FILT
maincat.meta['HSTSTARFILT'] = PS_HST_FILT
if FN_EXTRABAD is not None:
    maincat.meta['EXTRABAD'] = EXTRABAD_LABEL

for i, colname in enumerate(maincat.colnames):
    if 'FLAG' in colname:
        continue
    elif 'RADIUS' in colname:
        if ('KRON_RADIUS_CIRC' in colname) | ('FLUX_RADIUS' in colname):
            maincat[colname].unit = u.arcsec
            maincat[colname] *= PIXEL_SCALE
    elif 'FLUX' in colname:
        maincat[colname].unit = FLUX_UNIT
    elif colname in ('a', 'b', 'x', 'y'):
        maincat[colname].unit = u.pixel
    elif colname == 'theta':
        maincat[colname].unit = u.deg

    # print(i, colname)


try:
    maincat.write(outfilename, overwrite=True)
    print(f'Added date stamp! ({today})')
    print('Wrote first-pass combined catalog to ', outfilename)
except:
    print('WARNING: Could not make combined file. It may be too large!')


for apersize in PHOT_APER:
    if apersize==SCI_APER or MAKE_SCIREADY_ALL:
        str_aper = str(apersize).replace('.', '_')
        # restrict + rename columns to be a bit more informative
        cols = OrderedDict()
        cols['ID'] = 'id'
        cols['x'] = 'x'
        cols['y'] = 'y'
        cols['RA'] = 'ra'
        cols['DEC'] = 'dec'
        cols['EBV'] = 'ebv_mw'
        cols[f'{KRON_MATCH_BAND}_FLUX_APER{str_aper}_COLOR'] = f'faper_{KRON_MATCH_BAND}'
        cols[f'{KRON_MATCH_BAND}_FLUXERR_APER{str_aper}_COLOR'] = f'eaper_{KRON_MATCH_BAND}'
        cols[f'{KRON_MATCH_BAND}_FLUX_REF_AUTO_APER{str_aper}'] = f'fauto_{KRON_MATCH_BAND}'
        cols[f'{KRON_MATCH_BAND}_RELWHT'] = f'w_{KRON_MATCH_BAND}'

        for filter in FILTERS:
            cols[f'{filter}_FLUX_APER{str_aper}_TOTAL'] = f'f_{filter}'
            cols[f'{filter}_FLUXERR_APER{str_aper}_TOTAL'] = f'e_{filter}'
            cols[f'{filter}_RELWHT'] = f'w_{filter}'

        cols[f'TOTAL_CORR_APER{str_aper}'] = 'tot_cor'

        # cols[f'{KRON_MATCH_BAND}_FLAG_AUTO{mask}'] = 'flag_auto'
        cols[f'{KRON_MATCH_BAND}_KRON_RADIUS_APER{str_aper}'] = 'kron_radius'   # this is the modified KR where KR = sci_aper/2 for small things.
        cols[f'{KRON_MATCH_BAND}_KRON_RADIUS_CIRC_APER{str_aper}'] = 'kron_radius_circ' # ditto
        cols[f'{KRON_MATCH_BAND}_USE_CIRCLE_APER{str_aper}'] = 'use_circle' # where kron radius is not used.
        cols[f'{KRON_MATCH_BAND}_FLAG_KRON_APER{str_aper}'] = 'flag_kron'
        cols['iso_area'] = 'iso_area'
        cols['a'] = 'a_image'
        cols['b'] = 'b_image'
        cols['theta'] = 'theta_J2000' # double check this!
        cols[f'{KRON_MATCH_BAND}_FLUX_RADIUS_FRAC0_5'] = 'flux_radius'
        cols['use_phot'] = 'use_phot'
        cols['lowsnr_flag'] = 'flag_lowsnr'
        cols['star_flag'] = 'flag_star'
        # cols['bad_wht_flag'] = 'flag_badwht'
        if BP_USE:
            cols['bad_pixel_lw_flag'] = 'flag_artifact'
        if EXTRABAD_USE:
            cols['extrabad_flag'] = 'flag_nearbcg'
        if PATH_BADOBJECT is not None:
            cols['badobject_flag'] = 'flag_badobject'
        # cols['badkron_flag'] = 'flag_badkron'
        if GLASS_MASK is not None:
            cols['badglass_flag'] = 'flag_badflat'
        cols['z_spec'] = 'z_spec'

        subcat = maincat[list(cols.keys())].copy()
        subcat.meta['APER_DIAM'] = apersize

        for coln in subcat.colnames:
            newcol = cols[coln]
            print(f'   {subcat[coln].name} --> {newcol}')
            subcat[coln].name = newcol

        if XCAT_FILENAME is not None:
            # Crossmatch to DR1 and make a new column (ID + radius)
            from catalog_tools import crossmatch
            from astropy.table import Table, Column, MaskedColumn
            cat_old = Table.read(XCAT_FILENAME)
            mcat_new, mcat_old, idx1, idx2, dsky  = crossmatch(subcat, cat_old, thresh=[0.08*u.arcsec], plot=True, return_idx=True)
            ids = np.zeros(len(subcat))
            ids[idx1] = cat_old['id'][idx2]
            matchrad = np.zeros(len(subcat))
            matchrad[idx1] = dsky[idx1]
            subcat.add_column(MaskedColumn(ids, name=f'id_{XCAT_NAME}', mask=ids<=0, dtype='i4'))
            subcat.add_column(MaskedColumn(matchrad*u.arcsec, name=f'match_radius_{XCAT_NAME}', mask=ids<=0))

        # Kill any weight that has no flux
        for coln in subcat.colnames:
            if 'f_' in coln:
                badsel = np.isnan(subcat[coln]) | ~np.isfinite(subcat[coln])
                subcat[coln.replace('f_', 'w_')][badsel] = np.nan

        fluxes = np.array([subcat[coln] for coln in subcat.colnames if 'f_' in coln])
        badsel = np.nansum(np.isfinite(fluxes), 0) == 0
        print(f'Found {np.sum(badsel)} objects with NO viable photometry whatsoever. {np.sum(badsel & (subcat["use_phot"]==0))} already flagged. Flagging rest.')
        subcat['use_phot'][badsel] = 0
        subcat.add_column(Column(np.where(badsel, 1, 0), name='flag_nophot', dtype='i4'), 1+subcat.colnames.index('use_phot'))
        for coln in subcat.colnames:
            if 'radius' in coln:
                subcat[coln][badsel] = np.nan
        # print(np.sum(subcat['use_phot']==0))

        # # use flag (minimum SNR cut + not a star)
        # snr_ref = subcat[f'f_{KRON_MATCH_BAND}'] / subcat[f'e_{KRON_MATCH_BAND}']
        # snr_ref[subcat[f'e_{KRON_MATCH_BAND}']<=0] = -1
        # use_phot = np.zeros(len(subcat))
        # use_phot[snr_ref >= 3] = 1
        # use_phot[SEL_STAR | SEL_BADPIX | SEL_EXTRABAD | SEL_BADOBJECT | SEL_BADGLASS | sel_badkron] = 0
        # subcat['use_phot'] = use_phot
        str_aper = str_aper.replace('_', '')
        if len(str_aper) == 2:
            str_aper += '0' # 07 -> 070
        sub_outfilename = outfilename.replace('COMBINED', f'D{str_aper}')
        sub_outfilename = sub_outfilename.replace(DET_NICKNAME+'_K', f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K")
        subcat.write(sub_outfilename, overwrite=True)
        print('Wrote formatted combined catalog to ', sub_outfilename)
