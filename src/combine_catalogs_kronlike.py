from collections import OrderedDict
from astropy.io import fits, ascii
from astropy.table import Table, hstack, Column, vstack
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

from config import FILTERS, DIR_SFD, APPLY_MWDUST, DIR_CATALOGS,\
    REF_BAND, PIXEL_SCALE, PHOT_APER, DIR_KERNELS, DIR_PSFS, FIELD, ZSPEC, \
    MAX_SEP, SCI_APER, MAKE_SCIREADY_ALL, TARGET_ZP, ZCONF, ZRA, ZDEC, ZCOL, FLUX_UNIT, \
    PS_FLUXRATIO, PS_FLUXRATIO_RANGE, PS_FILT, PS_MAGLIMIT, PS_APERSIZE, \
        BP_FLUXRATIO, BP_FLUXRATIO_RANGE, BP_FILT, BP_MAGLIMIT, BP_APERSIZE


DET_NICKNAME =  sys.argv[2] #'LW_f277w-f356w-f444w'
KERNEL = sys.argv[3] #'f444w'

DET_TYPE = 'noise-equal'
FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

def DIR_KERNEL(band):
    return glob.glob(os.path.join(DIR_KERNELS, f'{KERNEL}*/{band.lower()}_kernel.fits'))[0]
stats = np.load(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_emptyaper_stats.npy'), allow_pickle=True).item()

FNAME_REF_PSF = f'{DIR_PSFS}/psf_{FIELD}_{REF_BAND.upper()}_4arcsec.fits'

def sigma_aper(filter, weight, weight_med, apersize=0.7):
    # Equation 5
    # apersize = str(apersize).replace('.', '_') + 'arcsec'
    sigma_nmad_filt = stats[filter.lower()][apersize]['ksnmad']
    # sigma_nmad_filt = ERROR_TABLE[f'e{apersize}'][ERROR_TABLE['filter']==filter.lower()][0]
    # g_i = 1.*2834.508 # here's to hoping.  EFFECTIVE GAIN!
    fluxvar = ( sigma_nmad_filt / np.sqrt(weight / weight_med) )**2  #+ (flux_aper / g_i)
    fluxvar[weight<=0] = np.inf
    return np.sqrt(fluxvar)

def sigma_total(sigma_aper, flux_ref_total, flux_ref_aper):
    # equation 6
    return sigma_aper * (flux_ref_total / flux_ref_aper)

def sigma_ref_total(sigma1, alpha, beta, kronrad_circ, wht_ref, medwht_ref, flux_refauto):
    # equation 7
    term1 = ((sigma1 * alpha * (np.pi * kronrad_circ**2)**(beta/2.)) / np.sqrt(wht_ref / medwht_ref) )**2
    # g_ref = 1.
    # term2 = flux_refauto / g_ref
    return np.sqrt(term1) # + term2)

# def sigma_full(sigma_total, sigma_ref_total, sigma_total_ref):
#     # equation 8
#     sig_full = sigma_total**2 + sigma_ref_total**2 - sigma_total_ref**2
#     # print(np.sum(sig_full < 0) / len(sig_full))
#     return np.sqrt(sig_full)

def flux_ref_total(flux_ref_auto, frac):
    # equation 9
    return flux_ref_auto  / frac

def flux_total(flux_aper, flux_ref_total, flux_ref_aper):
    # equation 10
    return flux_aper * ( flux_ref_total / flux_ref_aper )


# loop over filters
for filter in FILTERS:
    filename = os.path.join(FULLDIR_CATALOGS, f'{filter}_{DET_NICKNAME}_K{KERNEL}_PHOT_CATALOG.fits')
    if not os.path.exists(filename):
        print(f'ERROR :: {filename} not found!')
        sys.exit()

    cat = Table.read(filename)
    print(filter)

    # rename columns if needed:
    for coln in cat.colnames:
        if 'RADIUS' in coln or 'APER' in coln or 'FLAG' in coln or 'AUTO' in coln or 'WHT' in coln:

            newcol = f'{filter}_{coln}'.replace('.', '_')
            print(f'   {cat[coln].name} --> {newcol}')
            cat[coln].name = newcol

            try:
                cat[newcol] = cat[newcol].filled(np.nan)
                print('Filled with NaN!')
            except:
                pass

    if filter == FILTERS[0]:
        maincat = cat
    else:
        newcols = [coln for coln in cat.colnames if coln not in maincat.colnames]
        maincat = hstack([maincat, cat[newcols]])

# add ID at the end
maincat.add_column(Column(1+np.arange(len(maincat)), name='ID'), 0)
outfilename = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_COMBINED_CATALOG.fits')

# grab REF_BAND PSF convolved to kernel
psfmodel = fits.getdata(FNAME_REF_PSF)
if KERNEL == REF_BAND:
    conv_psfmodel = psfmodel.copy()
else:
    kernel = fits.getdata(DIR_KERNEL(REF_BAND))
    conv_psfmodel = convolve(psfmodel, kernel)

# Get some static refband stuff 
plotname = os.path.join(FULLDIR_CATALOGS, f'figures/aper_{REF_BAND}_nmad.pdf')
p, pcov, sigma1 = fit_apercurve(stats[REF_BAND], plotname=plotname, stat_type=['ksnmad'], pixelscale=PIXEL_SCALE)
alpha, beta = p['ksnmad']
sig1 = sigma1['ksnmad']
wht_ref = maincat[f'{REF_BAND}_SRC_MEDWHT']
medwht_ref = maincat[f'{REF_BAND}_MED_WHT']

for apersize in PHOT_APER:
    str_aper = str(apersize).replace('.', '_')

    # use REF_BAND Kron to correct to total fluxes and ferr
    f_ref_auto = maincat[f'{REF_BAND}_FLUX_AUTO'].copy()
    kronrad_circ = maincat[f'{REF_BAND}_KRON_RADIUS_CIRC'].copy()
    f_ref_aper = maincat[f'{REF_BAND}_FLUX_APER{str_aper}']

    use_circle = kronrad_circ < apersize / 2.
    kronrad_circ[use_circle] = apersize / 2.
    f_ref_auto[use_circle] = f_ref_aper[use_circle]

    psffrac_ref_auto = psf_cog(conv_psfmodel, nearrad = kronrad_circ) # in pixels
    # F160W kernel convolved REF_BAND PSF + missing flux from F160W beyond 2" radius
    f_ref_total = f_ref_auto / psffrac_ref_auto # equation 9
    sig_ref_total = sigma_ref_total(sig1, alpha, beta, kronrad_circ, wht_ref, medwht_ref, f_ref_auto)
    newcoln =f'{REF_BAND}_FLUXERR_REFTOTAL_MINDIAM{str_aper}'
    maincat.add_column(Column(sig_ref_total, newcoln))
    
    tot_corr = f_ref_total / f_ref_aper
    maincat.add_column(Column(1./tot_corr, f'TOTAL_CORR_APER{str_aper}'))
    sig_ref_aper = sigma_aper(REF_BAND.upper(), wht_ref, medwht_ref, apersize) # sig_aper,REF_BAND
    sig_total_ref = sigma_total(sig_ref_aper, f_ref_total, f_ref_aper) # sig_total,REF_BAND

    for filter in FILTERS:
        f_aper =maincat[f'{filter}_FLUX_APER{str_aper}']
        f_total = flux_total(f_aper, f_ref_total, f_ref_aper)
        wht = maincat[f'{filter}_SRC_MEDWHT']
        medwht = maincat[f'{filter}_MED_WHT']

        # get the flux uncertainty in the aperture for this band
        sig_aper = sigma_aper(filter, wht, medwht, apersize)
        # do again for each aperture
        sig_total = sigma_total(sig_aper, f_ref_total, f_ref_aper)
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
print(ebmv)


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
        if 'FLUX' in coln:
            maincat[coln] /= atten_factor[np.array(FILTERS) == filtname][0]

        elif 'MAG' in coln:
            maincat[coln] -= atten_mag[np.array(FILTERS) == filtname][0]

# star-galaxy flag
str_aper = str(PS_APERSIZE).replace('.', '_')
mag = TARGET_ZP - 2.5*np.log10(maincat[f'{PS_FILT}_FLUX_APER{str_aper}_TOTAL'])
size = maincat[f'{PS_FILT}_FLUX_APER{str(PS_FLUXRATIO[0]).replace(".", "_")}_COLOR']  \
                / maincat[f'{PS_FILT}_FLUX_APER{str(PS_FLUXRATIO[1]).replace(".", "_")}_COLOR']
is_star = (size > PS_FLUXRATIO_RANGE[0]) & (size < PS_FLUXRATIO_RANGE[1])
is_star &= (mag < PS_MAGLIMIT)

fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.scatter(mag, size, s=3, alpha=0.2, c='grey')
ax.scatter(mag[is_star], size[is_star], s=5, alpha=0.5)
# ax.invert_xaxis()
ax.set(ylim=(0,6), xlim=(18 ,31), ylabel=('$\mathcal{F}\,'+f'({PS_FLUXRATIO[0]}/{PS_FLUXRATIO[1]})$'), xlabel=f'${PS_FILT.upper()}$ Mag (AB)')
plotname = os.path.join(FULLDIR_CATALOGS, f'figures/{DET_NICKNAME}_K{KERNEL}_pointlikeflag.pdf')
fig.savefig(plotname)

maincat.add_column(Column(is_star.astype(int), name='star_flag'))

# bad pixel flag
str_aper = str(BP_APERSIZE).replace('.', '_')
BP_FILT_SEL = BP_FILT[DET_NICKNAME[:2]]
mag = TARGET_ZP - 2.5*np.log10(maincat[f'{BP_FILT_SEL}_FLUX_APER{str_aper}_TOTAL'])
size = maincat[f'{BP_FILT_SEL}_FLUX_APER{str(BP_FLUXRATIO[0]).replace(".", "_")}_COLOR']  \
                / maincat[f'{BP_FILT_SEL}_FLUX_APER{str(BP_FLUXRATIO[1]).replace(".", "_")}_COLOR']
is_badpixel = (size > BP_FLUXRATIO_RANGE[0]) & (size < BP_FLUXRATIO_RANGE[1])
is_badpixel &= (mag < BP_MAGLIMIT)

fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.scatter(mag, size, s=3, alpha=0.2, c='grey')
ax.scatter(mag[is_badpixel], size[is_badpixel], s=5, alpha=0.5)
# ax.invert_xaxis()
ax.set(ylim=(0,2), xlim=(18 ,31), ylabel=('$\mathcal{F}\,'+f'({BP_FLUXRATIO[0]}/{BP_FLUXRATIO[1]})$'), xlabel=f'${BP_FILT_SEL.upper()}$ Mag (AB)')
plotname = os.path.join(FULLDIR_CATALOGS, f'figures/{DET_NICKNAME}_K{KERNEL}_badpixelflag.pdf')
fig.savefig(plotname)

maincat.add_column(Column(is_badpixel.astype(int), name='bad_pixel_flag'))

# z-spec
ztable = Table.read(ZSPEC)
conf_constraint = np.ones(len(ztable), dtype=bool)
if ZCONF is not None:
    conf_constraint = np.isin(ztable[ZCONF[0]], np.array(ZCONF[1]))
ztable = ztable[conf_constraint & (ztable[ZDEC] >= -90.) & (ztable[ZDEC] <= 90.)]
zcoords = SkyCoord(ztable[ZRA]*u.deg, ztable[ZDEC]*u.deg)
catcoords = SkyCoord(maincat[ZRA], maincat[ZDEC])
idx, d2d, d3d = catcoords.match_to_catalog_sky(zcoords)
max_sep = MAX_SEP
sep_constraint = d2d < max_sep

for colname in ztable.colnames:
    filler = np.zeros(len(maincat), dtype=ztable[colname].dtype)
    try:
        np.nan * filler
    except:
        pass
    filler[sep_constraint] = ztable[idx[sep_constraint]][colname]
    if colname == ZCOL:
        colname = 'z_spec'
    else:
        colname = f'z_spec_{colname}'
    maincat.add_column(Column(filler, name=colname))

# use flag (minimum SNR cut + not a star)
snr_ref = maincat[f'{REF_BAND}_FLUX_APER0_7_COLOR'] / maincat[f'{REF_BAND}_FLUXERR_APER0_7_COLOR']
use_phot = np.zeros(len(maincat))
use_phot[snr_ref >= 3] = 1
use_phot[is_star | is_badpixel] = 0
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

    print(i, colname)

maincat.write(outfilename, overwrite=True)
print(f'Added date stamp! ({today})')
print('Wrote first-pass combined catalog to ', outfilename)


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
        cols[f'{REF_BAND}_FLUX_APER{str_aper}_COLOR'] = f'faper_{REF_BAND}'
        cols[f'{REF_BAND}_FLUXERR_APER{str_aper}_COLOR'] = f'eaper_{REF_BAND}'

        for filter in FILTERS:
            cols[f'{filter}_FLUX_APER{str_aper}_TOTAL'] = f'f_{filter}'
            cols[f'{filter}_FLUXERR_APER{str_aper}_TOTAL'] = f'e_{filter}'

        cols[f'TOTAL_CORR_APER{str_aper}'] = 'tot_cor'
        # cols[f'{REF_BAND}_FLUXERR_REFTOTAL'] = 'tot_ekron_f444w'

        # wmin?
        cols[f'{REF_BAND}_KRON_RADIUS'] = 'kron_radius'
        cols[f'{REF_BAND}_KRON_RADIUS_CIRC'] = 'kron_radius_circ'
        cols['a'] = 'a_image'
        cols['b'] = 'b_image'
        cols['theta'] = 'theta_J2000' # double check this!
        cols[f'{REF_BAND}_FLUX_RADIUS_0_5'] = 'flux_radius'
        cols['use_phot'] = 'use_phot'
        cols['z_spec'] = 'z_spec'
        cols['star_flag'] = 'star_flag'

        subcat = maincat[list(cols.keys())].copy()
        subcat.meta['APER_DIAM'] = apersize

        for coln in subcat.colnames:
            newcol = cols[coln]
            print(f'   {subcat[coln].name} --> {newcol}')
            subcat[coln].name = newcol

        # use flag (minimum SNR cut + not a star)
        snr_ref = subcat[f'faper_{REF_BAND}'] / subcat[f'eaper_{REF_BAND}']
        use_phot = np.zeros(len(subcat))
        use_phot[snr_ref >= 3] = 1
        use_phot[is_star | is_badpixel] = 0
        subcat['use_phot'] = use_phot

        sub_outfilename = outfilename.replace('COMBINED', f'SCIREADY_{str_aper}')
        subcat.write(sub_outfilename, overwrite=True)
        print('Wrote formatted combined catalog to ', sub_outfilename)
