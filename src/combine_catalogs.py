from typing import OrderedDict
from astropy.io import fits, ascii
from astropy.table import Table, hstack, Column, vstack
import numpy as np
import os, sys
import sfdmap
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.svo_fps import SvoFps
import extinction

# DET_NICKNAME = 'SW_f150w-f200w'
# DET_TYPE = 'noise-equal'
# PHOT_NICKNAMES = ('f150w', 'f200w') #  will loop over these so you only run detection once!

# DET_NICKNAME =  'LW_f356w-f444w'  
DET_NICKNAME = 'SW_f150w-f200w' 
KERNEL = 'f160w'

DET_TYPE = 'noise-equal'
PHOT_ZP = OrderedDict()
PHOT_ZP['f435w'] = 28.9
PHOT_ZP['f606w'] = 28.9
PHOT_ZP['f814w'] = 28.9
## PHOT_ZP['f098m'] = 28.9
PHOT_ZP['f105w'] = 28.9
PHOT_ZP['f125w'] = 28.9
PHOT_ZP['f140w'] = 28.9
PHOT_ZP['f160w'] = 28.9
PHOT_ZP['f115w'] = 28.9
PHOT_ZP['f150w'] = 28.9
PHOT_ZP['f200w'] = 28.9
PHOT_ZP['f277w'] = 28.9
PHOT_ZP['f410m'] = 28.9
PHOT_ZP['f356w'] = 28.9
PHOT_ZP['f444w'] = 28.9
PHOT_NICKNAMES = list(PHOT_ZP.keys())


DIR_CATALOGS = f'./data/output/v4/{DET_NICKNAME}_{DET_TYPE}_{KERNEL}/'
DIR_SFD = '~/Projects/Common/py_tools/sfddata-master'

ERROR_TABLE = ascii.read('./scripts/egs-grizli-v2.errors.v1.0.dat') # TODO update!


def sigma_aper(filter, weight, weight_med, apersize=0.7):
    # Equation 5
    apersize = str(apersize).replace('.', '_') + 'arcsec'
    sigma_nmad_filt = ERROR_TABLE[f'e{apersize}'][ERROR_TABLE['filter']==filter.lower()][0] #TODO LOOK AT THIS LINE
    print(filter, apersize, sigma_nmad_filt)
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
    g_ref = 1.
    # term2 = flux_refauto / g_ref
    return np.sqrt(term1) # + term2)

def sigma_full(sigma_total, sigma_ref_total, sigma_total_ref):
    # equation 8
    sig_full = sigma_total**2 + sigma_ref_total**2 - sigma_total_ref**2
    # print(np.sum(sig_full < 0) / len(sig_full))
    return np.sqrt(sig_full)

def flux_ref_total(flux_ref_auto, frac):
    # equation 9
    return flux_ref_auto  / frac

def flux_total(flux_aper, flux_ref_total, flux_ref_aper):
    # equation 10
    return flux_aper * ( flux_ref_total / flux_ref_aper )


# loop over filters
for filter in PHOT_NICKNAMES:
    filename = os.path.join(DIR_CATALOGS, f'{filter}_{DET_NICKNAME}_PHOT_CATALOG.fits')
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

    if filter == PHOT_NICKNAMES[0]:
        maincat = cat
    else:
        newcols = [coln for coln in cat.colnames if coln not in maincat.colnames]
        maincat = hstack([maincat, cat[newcols]])

# add ID at the end
maincat.add_column(Column(1+np.arange(len(maincat)), name='ID'), 0)
outfilename = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_COMBINED_CATALOG.fits')

# grab F444W PSF convolved to F160W COG #TODO -- do this on the fly!
# px is radius in pixels, cumcurve is fraction normalized to *true* total (including missing 5%!)
px = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,
         5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5,
        11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5, 15. , 15.5, 16. ,
        16.5, 17. , 17.5, 18. , 18.5, 19. , 19.5, 20. , 20.5, 21. , 21.5,
        22. , 22.5, 23. , 23.5, 24. , 24.5, 25. , 25.5, 26. , 26.5, 27. ,
        27.5, 28. , 28.5, 29. , 29.5, 30. , 30.5, 31. , 31.5, 32. , 32.5,
        33. , 33.5, 34. , 34.5, 35. , 35.5, 36. , 36.5, 37. , 37.5, 38. ,
        38.5, 39. , 39.5, 40. , 40.5, 41. , 41.5, 42. , 42.5, 43. , 43.5,
        44. , 44.5, 45. , 45.5, 46. , 46.5, 47. , 47.5, 48. , 48.5, 49. ,
        49.5, 50. ])
cumcurve = np.array([0.        , 0.13354114, 0.13354114, 0.36966026, 0.36966026,
        0.36966026, 0.5365828 , 0.5365828 , 0.5365828 , 0.64065045,
        0.64065045, 0.64065045, 0.70989686, 0.70989686, 0.70989686,
        0.76322675, 0.76322675, 0.80428576, 0.80428576, 0.80428576,
        0.8325678 , 0.8325678 , 0.8325678 , 0.8503909 , 0.8503909 ,
        0.8503909 , 0.86169165, 0.86169165, 0.86169165, 0.8698046 ,
        0.8698046 , 0.8698046 , 0.8768134 , 0.8768134 , 0.8834177 ,
        0.8834177 , 0.8834177 , 0.8895124 , 0.8895124 , 0.8895124 ,
        0.8951266 , 0.8951266 , 0.8951266 , 0.90045726, 0.90045726,
        0.90045726, 0.90538174, 0.90538174, 0.90538174, 0.90961045,
        0.90961045, 0.9131284 , 0.9131284 , 0.9131284 , 0.91612226,
        0.91612226, 0.91612226, 0.9187267 , 0.9187267 , 0.9187267 ,
        0.9210595 , 0.9210595 , 0.9210595 , 0.9232772 , 0.9232772 ,
        0.9232772 , 0.9254668 , 0.9254668 , 0.92760575, 0.92760575,
        0.92760575, 0.92965627, 0.92965627, 0.92965627, 0.9316173 ,
        0.9316173 , 0.9316173 , 0.93349534, 0.93349534, 0.93349534,
        0.9352747 , 0.9352747 , 0.9352747 , 0.9369267 , 0.9369267 ,
        0.9384339 , 0.9384339 , 0.9384339 , 0.939796  , 0.939796  ,
        0.939796  , 0.9410236 , 0.9410236 , 0.9410236 , 0.9421429 ,
        0.9421429 , 0.9421429 , 0.9431869 , 0.9431869 , 0.9441742 ,
        0.9441742 ])

# use F444W Kron to correct to total fluxes and ferr
alpha, beta, sig1 = 0.20, 1.72, 0.0010 #TODO -- do this on the fly!
f_ref_auto = maincat[f'f444w_FLUX_AUTO']
kronrad_circ = np.sqrt(maincat['a'] * maincat['b'] * maincat['f444w_KRON_RADIUS']**2)
kronrad_circ[kronrad_circ<3.5] = 3.5 # PHOT_AUTOPARAMS[1]
psffrac_ref_auto = cumcurve[np.array([np.argmin(abs(px - i)) for i in kronrad_circ])]
# F160W kernel convolved F444W PSF + missing flux from F160W beyond 2" radius
f_ref_total = f_ref_auto / psffrac_ref_auto # equation 9
wht_ref = maincat[f'f444w_SRC_MEDWHT']
medwht_ref = maincat[f'f444w_MED_WHT']
sig_ref_total = sigma_ref_total(sig1, alpha, beta, kronrad_circ, wht_ref, medwht_ref, f_ref_auto)
newcoln =f'f444w_FLUXERR_REFTOTAL'
maincat.add_column(Column(sig_ref_total, newcoln))
REF_PIXEL_SCALE = 0.039999999999996004 # arcsec / px


for apersize in (0.16, 0.35, 0.7, 2.0):
    str_aper = str(apersize).replace('.', '_')
    f_ref_aper = maincat[f'f444w_FLUX_APER{str_aper}']
    tot_corr = f_ref_total / f_ref_aper
    maincat.add_column(Column(tot_corr, f'TOTAL_CORR_APER{str_aper}'))
    sig_ref_aper = sigma_aper('F444W', wht_ref, medwht_ref, apersize) # sig_aper,F444W
    sig_total_ref = sigma_total(sig_ref_aper, f_ref_total, f_ref_aper) # sig_total,F444W

    for filter in PHOT_NICKNAMES:
        f_aper =maincat[f'{filter}_FLUX_APER{str_aper}']
        f_total = flux_total(f_aper, f_ref_total, f_ref_aper)
        wht = maincat[f'{filter}_SRC_MEDWHT']
        medwht = maincat[f'{filter}_MED_WHT']

        # get the flux uncertainty in the aperture for this band
        sig_aper = sigma_aper(filter, wht, medwht, apersize)

        # Convert the flux uncertainty in the aperture to the total using F444W auto and aper
        # if filter != 'f444w':

        # else:
        #     sig_total = sigma_ref_total # defined already for kron radii

        sig_total = sigma_total(sig_aper, f_ref_total, f_ref_aper) # do again for each aperture
        sig_full = sigma_full(sig_total, sig_ref_total, sig_total_ref)


        # add new columns
        newcoln = f'{filter}_FLUX_APER{str_aper}_COLOR'
        maincat.add_column(Column(f_aper, newcoln))
        newcoln =f'{filter}_FLUXERR_APER{str_aper}_COLOR'
        maincat.add_column(Column(sig_aper, newcoln))

        newcoln = f'{filter}_FLUX_APER{str_aper}_TOTAL'
        maincat.add_column(Column(f_total, newcoln))
        newcoln = f'{filter}_FLUXERR_APER{str_aper}_TOTAL'
        maincat.add_column(Column(sig_total, newcoln))

        newcoln =f'{filter}_FLUXERR_APER{str_aper}_FULL'
        maincat.add_column(Column(sig_full, newcoln))

        # magnitudes
        # mag_total = 25. - 2.5*np.log10(f_total)
        # merr_full = 2.5 * np.log(10) / (f_total / sig_total)


        # newcoln = f'{filter}_MAG_APER{str_aper}_TOTAL'
        # maincat.add_column(Column(mag_total, newcoln))
        # newcoln =f'{filter}_MAGERR_APER{str_aper}_FULL'
        # maincat.add_column(Column(merr_full, newcoln))


# ADD SFD maps (2011 scales by 0.86, which is default. otherwise use scaling=1.0)
m = sfdmap.SFDMap(DIR_SFD)
ebmv = m.ebv(maincat['RA'], maincat['DEC'])
maincat.add_column(Column(ebmv, name='EBV'), 1+np.where(np.array(maincat.colnames) == 'DEC')[0][0])
print(ebmv)
Av_mean = np.mean(ebmv)*3.1

# Perform a MW correction (add new columns to the master)
filter_table = vstack([SvoFps.get_filter_list('JWST'),\
                     SvoFps.get_filter_list('HST')])
filter_pwav = OrderedDict()
print('Building directory of pivot wavelengths')
for filter in PHOT_NICKNAMES:
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

atten_mag = extinction.fm07(np.array(list(filter_pwav.values())), Av_mean) # atten_mag in magnitudes from Fitzpatrick + Massa 2007
atten_factor = 10** (-0.4 * atten_mag) # corresponds in order to PHOT_NICKNAMES
for i, filter in enumerate(PHOT_NICKNAMES):
    print(f'{filter} ::  {atten_factor[i]:2.5f}x or {atten_mag[i]:2.5f} AB')

print('Applying Milky Way Attenuation correction (FM+07)')
for coln in maincat.colnames:
    if 'RADIUS' in coln:
        continue
    filtname = coln.split('_')[0]
    if 'FLUX' in coln:
        maincat[coln] /= atten_factor[np.array(PHOT_NICKNAMES) == filtname][0]

    elif 'MAG' in coln:
        maincat[coln] -= atten_mag[np.array(PHOT_NICKNAMES) == filtname][0]

# star-galaxy flag # TODO LATER
is_star = np.zeros(len(maincat), dtype=bool)
maincat.add_column(Column(is_star.astype(int), name='star_flag'))

# z-spec
ztable = Table.read('./data/external/EGS_z_spec.fits')
ztable = ztable[(ztable['dec'] >= -90.) & (ztable['dec'] <= 90.)]
zcoords = SkyCoord(ztable['ra']*u.deg, ztable['dec']*u.deg)
catcoords = SkyCoord(maincat['RA'], maincat['DEC'])
idx, d2d, d3d = catcoords.match_to_catalog_sky(zcoords)
z_spec = -99. * np.ones(len(maincat))
max_sep = 0.3 * u.arcsec
sep_constraint = d2d < max_sep
z_spec[sep_constraint] = ztable[idx[sep_constraint]]['z_spec']
maincat.add_column(Column(z_spec, name='z_spec'))

# use flag (minimum SNR cut + not a star)
snr_ref = maincat['f444w_FLUX_APER0_7_COLOR'] / maincat['f444w_FLUXERR_APER0_7_COLOR']
use_phot = np.zeros(len(maincat))
use_phot[snr_ref >= 3] = 1
use_phot[is_star] = 0
maincat.add_column(Column(use_phot, name='use_phot'))
# use 1 only

# Spit it out!
maincat.write(outfilename, overwrite=True)
print('Wrote first-pass combined catalog to ', outfilename)

# restrict + rename columns to be a bit more informative
cols = OrderedDict()
cols['ID'] = 'id'
cols['x'] = 'x'
cols['y'] = 'y'
cols['RA'] = 'ra'
cols['DEC'] = 'dec'
cols['EBV'] = 'ebv_mw'
cols['f444w_FLUX_APER0_7_COLOR'] = 'faper_F444W'
cols['f444w_FLUXERR_APER0_7_COLOR'] = 'eaper_F444W'

for filter in PHOT_NICKNAMES:
    cols[f'{filter}_FLUX_APER0_7_TOTAL'] = f'f_{filter}'
    cols[f'{filter}_FLUXERR_APER0_7_TOTAL'] = f'e_{filter}'

cols['f444w_FLUXERR_REFTOTAL'] = 'tot_ekron_F444w'
cols['TOTAL_CORR_APER0_7'] = 'tot_cor'
# wmin?
cols['z_spec'] = 'z_spec'
cols['star_flag'] = 'star_flag'
cols['f444w_KRON_RADIUS'] = 'kron_radius'
cols['a'] = 'a_image'
cols['b'] = 'b_image'
cols['theta'] = 'theta_J2000' # double check this!
cols['f444w_FLUX_RADIUS_0_5'] = 'flux_radius' # arcsec
cols['use_phot'] = 'use_phot'

maincat = maincat[list(cols.keys())]

for coln in maincat.colnames:
    newcol = cols[coln]
    print(f'   {maincat[coln].name} --> {newcol}')
    maincat[coln].name = newcol

maincat['flux_radius'] *= REF_PIXEL_SCALE # from pixel to arcsec
# maincat['theta_J2000'] = np.rad2deg(maincat['theta_J2000'])  # radians to degrees

outfilename = outfilename.replace('COMBINED', 'SCIREADY')
maincat.write(outfilename, overwrite=True)

from datetime import date
hdul = fits.open(outfilename)
today = date.today().strftime("%d/%m/%Y")
hdul[0].header['CREATED'] = today
hdul.writeto(outfilename, overwrite=True)
print(f'Added date stamp! ({today})')
print('Wrote formatted combined catalog to ', outfilename)
