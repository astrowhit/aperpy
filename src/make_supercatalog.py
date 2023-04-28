import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

import numpy as np
import os, sys
from astropy.table import hstack
import astropy.units as u
from astropy.table import Table

from config import DIR_CATALOGS, PHOT_APER, VERSION, PROJECT, BLEND_SHRINK_FACTOR, PIXEL_SCALE

DET_NICKNAME =  sys.argv[2]
KERNEL = sys.argv[3]

DET_TYPE = 'noise-equal'
FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')


# First determine which set of photometry that object is going to get.
# Begin by assuming smallest aperture
straper = f'{np.min(PHOT_APER):2.2f}'.replace('.', '')
RELEASE = Table.read(os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_D{straper}_CATALOG.fits"))
use_aper = PHOT_APER[0] * np.ones(len(RELEASE))
shrink_factor = BLEND_SHRINK_FACTOR

# If isolated: Use the largest circular aperture that is smaller than iso_area
# If blended: Determine best aperture by dividing iso_diam by a shrink factor ~2
# already, blended things have f_XXX set to color apertures and tot_corr = 1!
for aper in np.sort(PHOT_APER):
    straper = f'{aper:2.2f}'.replace('.', '')
    RELEASE = Table.read(os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_D{straper}_CATALOG.fits"))
    is_blended = RELEASE['flag_kron'] != 0 # blended, or some other detection issue. Don't trust it.
    iso_aper = np.sqrt(RELEASE['iso_area'] / np.pi) * 2. #2.
    use_phot = RELEASE['use_phot'] == 1
    use_aper[(iso_aper > aper) & (use_phot) & (~is_blended)] = aper
    use_aper[((iso_aper/shrink_factor) > (aper)) & (use_phot) & is_blended] = aper
    
# Then loop over 
for aper in PHOT_APER:
    straper = f'{aper:2.2f}'.replace('.', '')
    CATALOG = Table.read(os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_D{straper}_CATALOG.fits"))
    RELEASE[use_aper == aper] = CATALOG[use_aper == aper]

# add column
from astropy.table import Column
RELEASE.add_column(Column(use_aper*u.arcsec, name='use_aper'), RELEASE.colnames.index('iso_area')+1)


for i in np.array(np.unique(np.array(RELEASE['use_aper']), return_counts=True)).T:
    print(f'N({i[0]}'+'\"'+f') = {int(i[1])} ({100*i[1]/len(RELEASE):2.0f}%)')
print('------')
for col in RELEASE.colnames:
    if ('flag' in col) | ('use' in col):
        print(f'N({col}) =  {np.sum(RELEASE[col]!=0)}  ({100*RELEASE[col].sum()/len(RELEASE):2.0f}%)' )

RELEASE.meta['APER_DIAM'] = 'ADAPTIVE'
RELEASE.meta['SHRINK_FACTOR'] = str(BLEND_SHRINK_FACTOR)

RELEASE.write(os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_SUPER_CATALOG.fits"), overwrite=True)
    

from regions import EllipseSkyRegion, Regions, CircleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

detcoords = SkyCoord(RELEASE['ra'], RELEASE['dec'])
pixel_scale = PIXEL_SCALE
# mag = TARGET_ZPT - 2.5*np.log10(RELEASE['f_f444w'])

print('BUILDING REGION FILE...')
regs = []
i = 0
for coord, obj in zip(detcoords, RELEASE):
    
    objid = str(obj['id'])

    if obj['use_phot'] == 0: continue
    
    if obj['use_circle'] == 0:
        ellip = obj['b_image'] / obj['a_image']
        kr = obj['kron_radius']
        krcirc = kr * np.sqrt(obj['a_image'] * obj['b_image'])

        width = 2* kr * obj['a_image'] * pixel_scale / 3600. * u.deg
        height = 2* kr * obj['b_image'] * pixel_scale / 3600. * u.deg
        angle = np.rad2deg(obj['theta_J2000']) * u.deg
        regs.append(CircleSkyRegion(coord, obj['use_aper']/2.*u.arcsec))
        regs.append(EllipseSkyRegion(coord, width, height, angle, meta={'text':objid}))

        
    else:
        regs.append(CircleSkyRegion(coord, obj['use_aper']/2.*u.arcsec, meta={'text':objid}))
        
regs = np.array(regs)
bigreg = Regions(regs)
bigreg.write(os.path.join(FULLDIR_CATALOGS, f"{PROJECT}_v{VERSION}_{DET_NICKNAME.split('_')[0]}_K{KERNEL}_SUPER_CATALOG.reg"), overwrite=True, format='ds9')