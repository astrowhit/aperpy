
import numpy as np 
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS, utils
import astropy.units as u
import sep
import os, sys, glob
from astropy.convolution import Gaussian2DKernel
from regions import EllipseSkyRegion, Regions, CircleSkyRegion
from webb_tools import emtpy_apertures

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import TARGET_ZPT, PHOT_APER, PHOT_AUTOPARAMS, PHOT_FLUXFRAC, DETECTION_PARAMS,\
         DIR_IMAGES, PHOT_ZP, PHOT_NICKNAMES, DIR_OUTPUT, DIR_CATALOGS

# MAIN PARAMETERS
DET_NICKNAME = sys.argv[2] #'LW_f277w-f356w-f444w' 
KERNEL = sys.argv[3] # f444w or f160w or None

DET_TYPE = 'noise-equal'
FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')
if not os.path.exists(FULLDIR_CATALOGS):
    os.mkdir(FULLDIR_CATALOGS)

# SECONDARY PARAMETERS
PATH_DETSCI = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{DET_NICKNAME}_{DET_TYPE}.fits.gz')
PATH_DETWHT = 'None'
PATH_DETMASK = 'None'


def conv_flux(in_zpt, out_zpt=TARGET_ZPT):
    return 10** (-0.4 * (in_zpt - out_zpt))

# 1 DETECTION

# READ IN IMAGES
print('READING DETECTION IMAGES...')
detsci = fits.getdata(PATH_DETSCI).byteswap().newbyteorder()
print(PATH_DETSCI)
if PATH_DETWHT != 'None':
    detwht = fits.getdata(PATH_DETWHT).byteswap().newbyteorder()
    print(PATH_DETWHT)
else:
    detwht = np.ones_like(detsci)
if PATH_DETMASK != 'None':
    detmask =  fits.getdata(PATH_DETMASK).byteswap().newbyteorder().astype(float)
    print(PATH_DETMASK)
else:
    detmask = np.zeros_like(detsci)
dethead = fits.getheader(PATH_DETSCI, 0)
detwcs = WCS(dethead)
print(detwcs)

detsci[detmask == 1.0] = 0.
if PATH_DETWHT != 'None':
    deterr = np.where(detwht==0, np.inf, 1./np.sqrt(detwht))
else:
    deterr = None

# SOME BASIC INFO
pixel_scale = utils.proj_plane_pixel_scales(detwcs)[0] * 3600
print(f'Pixel scale: {pixel_scale}')
area = np.sum((detwht!=0) & (detmask==0)) * (pixel_scale  / 3600)**2
print(f'Area of detection image: {area} deg2')

# SOURCE DETECTION
print('SOURCE DETECTION...')
kernel = np.array(Gaussian2DKernel(DETECTION_PARAMS['kernelfwhm']/2.35, factor=1))
sep.set_extract_pixstack(10000000) # big image...
del DETECTION_PARAMS['kernelfwhm']
objects, segmap = sep.extract(
                detsci, 
                err=deterr, 
                filter_type='matched',
                clean=True, filter_kernel=kernel,
                segmentation_map=True,
                **DETECTION_PARAMS
                )

print(f'Detected {len(objects)} objects.')

hdul = fits.HDUList()
hdul.append(fits.ImageHDU(name='SEGMAP', data=segmap, header=dethead))
hdul.writeto(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_SEGMAP.fits.gz'), overwrite=True)

# CLEAN UP
catalog = Table(objects)
detcoords = detwcs.pixel_to_world(catalog['x'], catalog['y'])
catalog['RA'] = [c.ra for c in detcoords]
catalog['DEC'] = [c.dec for c in detcoords]

print('BUILDING REGION FILE...')
regs = []
for coord, obj in zip(detcoords, catalog):
    width = 2*obj['a'] * pixel_scale / 3600. * u.deg
    height = 2*obj['b'] * pixel_scale / 3600. * u.deg
    angle = np.rad2deg(obj['theta']) * u.deg
    regs.append(EllipseSkyRegion(coord, width, height, angle))
    # regs.append(PointSkyRegion(coord))
regs = np.array(regs)
bigreg = Regions(regs)
bigreg.write(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_OBJECTS.reg'), overwrite=True, format='ds9')

del detsci
del detwht
del detmask
del deterr

if PHOT_NICKNAMES == 'None':
    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    catalog.write(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_DET_CATALOG.fits.gz'), overwrite=True)
    sys.exit()

areas = {}
stats = {}
for ind, PHOT_NICKNAME in enumerate(PHOT_NICKNAMES):

    print(PHOT_NICKNAME)
    ext = ''
    dir_weight = DIR_IMAGES
    if KERNEL != 'None':
        ext=f'_{KERNEL}-matched'
        dir_weight = DIR_OUTPUT
    print(DIR_OUTPUT)
    print(f'*{PHOT_NICKNAME}*_sci_skysubvar{ext}.fits.gz')
    PATH_PHOTSCI = glob.glob(os.path.join(DIR_OUTPUT, f'*{PHOT_NICKNAME}*_sci_skysubvar{ext}.fits.gz'))[0]
    PATH_PHOTHEAD = PATH_PHOTSCI
    PATH_PHOTWHT = glob.glob(os.path.join(dir_weight, f'*{PHOT_NICKNAME}*_wht{ext}.fits.gz'))[0]
    PATH_PHOTMASK = 'None'

    PHOT_ZPT = PHOT_ZP[PHOT_NICKNAME.lower()] #calc_zpt(PHOT_NICKNAME)
    print(f'Zeropoint for {PHOT_NICKNAME}: {PHOT_ZPT}')

    # 2 FORCED PHOTOMETRY + measurements
    # READ IN IMAGES
    print('READING PHOTOMETRY IMAGES...')
    photsci = fits.getdata(PATH_PHOTSCI).byteswap().newbyteorder()
    print(PATH_PHOTSCI)
    photwht = fits.getdata(PATH_PHOTWHT).byteswap().newbyteorder()
    photsci[photwht<=0.] = 0. # double check!
    print(PATH_PHOTWHT)
    if PATH_PHOTMASK != 'None':
        photmask = fits.getdata(PATH_PHOTMASK).byteswap().newbyteorder().astype(float)
        print(PATH_PHOTMASK)
        photsci[photmask==1.0] = 0
    else:
        photmask = None
    phothead = fits.getheader(PATH_PHOTHEAD, 0)
    photwcs = WCS(phothead)
    print(photwcs)

    photerr = np.where((photwht==0) | np.isnan(photwht), np.inf, 1./np.sqrt(photwht))
    photerr[~np.isfinite(photerr)] = np.median(photerr[np.isfinite(photerr)]) # HACK fill in holes with median weight.
    # fits.ImageHDU(photerr).writeto('PHOTERR.fits')

    # SOME BASIC INFO
    pixel_scale = utils.proj_plane_pixel_scales(photwcs)[0] * 3600
    print(f'Pixel scale: {pixel_scale}')
    area = np.sum(np.isfinite(photwht) & (photwht > 0.)) * (pixel_scale  / 3600)**2
    print(f'Area of photometry image: {area} deg2')
    areas[PHOT_NICKNAME] = area

    # Hack the x,y coords
    xphot,yphot = photwcs.wcs_world2pix(catalog['RA'], catalog['DEC'],1)

    # APERTURE PHOTOMETRY
    sep_fluxes = {}
    for diam in PHOT_APER:
        rad = diam / 2. / pixel_scale
        print(f"{PHOT_NICKNAME} :: MEASURING PHOTOMETRY in {diam:2.2f}\" apertures... ({2*rad:2.1f} px)")
        flux, fluxerr, flag = sep.sum_circle(photsci, xphot, yphot, #objects['x'], objects['y'],
                                            # mask = photmask,
                                            err = photerr, subpix=0,
                                            r=rad, gain=1.0)

        badflux = (flux == 0.) | ~np.isfinite(flux) 
        badfluxerr = (fluxerr <= 0.) | ~np.isfinite(fluxerr)
        pc_badflux = np.sum(badflux) / len(flux)
        pc_badfluxerr = np.sum(badfluxerr) / len(flux)
        pc_ORbad = np.sum(badflux | badfluxerr) / len(flux)
        pc_ANDbad = np.sum(badflux & badfluxerr) / len(flux)
        print(f'{pc_badflux*100:2.5f}% have BAD fluxes')
        print(f'{pc_badfluxerr*100:2.5f}% have BAD fluxerrs')
        print(f'{pc_ORbad*100:2.5f}% have BAD fluxes OR fluxerrs')
        print(f'{pc_ANDbad*100:2.5f}% have BAD fluxes AND fluxerrs')

        # diagnostic region files
        # regs = []
        # for coord, obj in zip(detcoords, catalog):
        #     regs.append(CircleSkyRegion(coord, rad*u.arcsec))
        # regs = np.array(regs)
        # bigreg = Regions(regs[badflux])
        # bigreg.write(os.path.join(FULLDIR_CATALOGS, f'{PHOT_NICKNAME}_{DET_NICKNAME}_BADFLUX{diam}_OBJECTS.reg'), overwrite=True, format='ds9')
        # bigreg = Regions(regs[badfluxerr])
        # bigreg.write(os.path.join(FULLDIR_CATALOGS, f'{PHOT_NICKNAME}_{DET_NICKNAME}_BADFLUXERR{diam}_OBJECTS.reg'), overwrite=True, format='ds9')

        bad = badflux | badfluxerr

        flux[bad] = np.nan
        fluxerr[bad] = np.nan
        flag[bad] = 1
        
        sep_fluxes[diam] = (flux, fluxerr, flag)

        # show the first 10 objects results:
        for i in range(3):
            print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))
    
        catalog[f'FLUX_APER{diam}'] = flux * conv_flux(PHOT_ZPT)
        catalog[f'FLUXERR_APER{diam}'] = fluxerr * conv_flux(PHOT_ZPT)
        # catalog[f'MAG_APER{diam}'] = PHOT_ZPT - 2.5*np.log10(flux)
        # catalog[f'MAGERR_APER{diam}'] = 2.5 / np.log(10) / ( flux / fluxerr )
        catalog[f'FLAG_APER{diam}'] = flag

    # KRON RADII AND MAG_AUTO
    print(f"{PHOT_NICKNAME} :: MEASURING PHOTOMETRY in kron-corrected AUTO apertures...")
    kronrad, krflag = sep.kron_radius(photsci, xphot, yphot, #objects['x'], objects['y'],
                                        objects['a'], objects['b'], objects['theta'], 6.0) # SE uses 6
    kronrad[np.isnan(kronrad)] = 0.
    print(np.isnan(kronrad).sum(), np.max(kronrad), np.min(kronrad))
    objects['theta'][objects['theta'] > np.pi / 2.] = np.pi / 2. # numerical rounding correction!
    flux, fluxerr, flag = sep.sum_ellipse(photsci, xphot, yphot, #objects['x'], objects['y'],
                                        objects['a'], objects['b'], objects['theta'], PHOT_AUTOPARAMS[0]*kronrad,
                                        err = photerr,
                                        subpix=1)

    badflux = (flux == 0.) | ~np.isfinite(flux) 
    badfluxerr = (fluxerr <= 0.) | ~np.isfinite(fluxerr)
    pc_badflux = np.sum(badflux) / len(flux)
    pc_badfluxerr = np.sum(badfluxerr) / len(flux)
    pc_ORbad = np.sum(badflux | badfluxerr) / len(flux)
    pc_ANDbad = np.sum(badflux & badfluxerr) / len(flux)
    print(f'{pc_badflux*100:2.5f}% have BAD fluxes')
    print(f'{pc_badfluxerr*100:2.5f}% have BAD fluxerrs')
    print(f'{pc_ORbad*100:2.5f}% have BAD fluxes OR fluxerrs')
    print(f'{pc_ANDbad*100:2.5f}% have BAD fluxes AND fluxerrs')

    bad = badflux | badfluxerr

    flux[bad] = np.nan
    fluxerr[bad] = np.nan
    flag[bad] = 1

    flag |= krflag  # combine flags into 'flag'


    r_min = PHOT_AUTOPARAMS[1]  # minimum radius = 3.5
    use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
    cflux, cfluxerr, cflag = sep.sum_circle(photsci, xphot[use_circle], yphot[use_circle],
                                            r=r_min, subpix=1,
                                            err = photerr, gain=1.0
                                            )

    badflux = (cflux == 0.) |  np.isnan(cflux) | ~np.isfinite(cflux) 
    badfluxerr = (cfluxerr <= 0.) | np.isnan(cfluxerr) | ~np.isfinite(cfluxerr)
    pc_badflux = np.sum(badflux) / len(cflux)
    pc_badfluxerr = np.sum(badfluxerr) / len(cflux)
    pc_ORbad = np.sum(badflux | badfluxerr) / len(cflux)
    pc_ANDbad = np.sum(badflux & badfluxerr) / len(cflux)
    print(f'{pc_badflux*100:2.5f}% have BAD fluxes')
    print(f'{pc_badfluxerr*100:2.5f}% have BAD fluxerrs')
    print(f'{pc_ORbad*100:2.5f}% have BAD fluxes OR fluxerrs')
    print(f'{pc_ANDbad*100:2.5f}% have BAD fluxes AND fluxerrs')

    bad = badflux | badfluxerr

    cflux[bad] = np.nan
    cfluxerr[bad] = np.nan
    cflag[bad] = 1

    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    flag[use_circle] = cflag
    kronrad[use_circle] = r_min

    catalog[f'FLUX_AUTO'] = flux * conv_flux(PHOT_ZPT)
    catalog[f'FLUXERR_AUTO'] = fluxerr * conv_flux(PHOT_ZPT)
    catalog[f'MAG_AUTO'] = PHOT_ZPT - 2.5*np.log10(flux)
    catalog[f'MAGERR_AUTO'] = 2.5 / np.log(10) / ( flux / fluxerr )
    catalog[f'KRON_RADIUS'] = kronrad
    catalog[f'FLAG_AUTO'] = flag

    # FLUX RADIUS
    print(f"{PHOT_NICKNAME} :: MEASURING FLUX RADIUS...")
    """In Source Extractor, the FLUX_RADIUS parameter gives the radius of a circle enclosing a desired fraction of the total flux."""
    r, flag = sep.flux_radius(photsci,  xphot, yphot, 6.*objects['a'],
                            PHOT_FLUXFRAC, normflux=flux, subpix=5)
    rt = r.T
    for i, fluxfrac in enumerate(PHOT_FLUXFRAC):
        catalog[f'FLUX_RADIUS_{fluxfrac}'] = rt[i]
    catalog['FLUX_RADIUS_FLAG'] = flag


    # SOURCE WEIGHT + MEDIAN WEIGHT 
    srcmedwht = np.nan * np.ones(len(catalog))
    srcmeanwht = np.nan * np.ones(len(catalog))
    for i, (ixphot, iyphot) in enumerate(zip(xphot, yphot)):
        intx, inty = int(ixphot), int(iyphot)
        boxwht = photwht[inty-4:inty+5, intx-4:intx+5]
        srcmedwht[i] = np.nanmedian(boxwht[boxwht>0])
        srcmeanwht[i] = np.nanmean(boxwht[boxwht>0])

    srcmeanwht[srcmeanwht<=0.] = np.nan
    srcmedwht[srcmedwht<=0.] = np.nan

    catalog['SRC_MEDWHT'] = srcmedwht
    catalog['SRC_MEANWHT'] = srcmeanwht
    medwht = np.nanmedian(photwht[photwht>0])
    catalog['MED_WHT'] = medwht

    # COMPUTE EMPTY APERTURE ERRORS + SAVE TO MASTER FILE
    empty_aper = list(PHOT_APER)+list(np.linspace(PHOT_APER[0], PHOT_APER[-1], 30))
    empty_aper = np.sort(empty_aper)
    plotname = os.path.join(FULLDIR_CATALOGS, f'{PHOT_NICKNAME}_K{KERNEL}emptyaper.pdf')
    zpt_factor = conv_flux(PHOT_ZPT)
    print(np.shape(photsci), np.shape(photwht), np.shape(segmap))
    stats[PHOT_NICKNAME] = emtpy_apertures(photsci, photwht, segmap, N=int(1e3), aper=empty_aper, plotname=plotname, zpt_factor=zpt_factor)

    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    catalog.write(os.path.join(FULLDIR_CATALOGS, f'{PHOT_NICKNAME}_{DET_NICKNAME}_PHOT_CATALOG.fits'), overwrite=True)

np.save(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_emptyaper_stats.npy'), stats)
with open(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_AREAS.dat'), 'w') as f:
    for filt in PHOT_NICKNAMES:
        area = areas[filt]
        f.write(f'{filt} {area}')
        f.write('\n')
