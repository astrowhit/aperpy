from typing import OrderedDict
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, utils
import astropy.units as u
import sep
import os, sys, glob
from photutils.aperture import aperture_photometry, CircularAperture
from astropy.convolution import Gaussian2DKernel
from regions import EllipseSkyRegion, Regions, CircleSkyRegion
from webb_tools import compute_background, emtpy_apertures

import convenience as conv

# MAIN PARAMETERS
# DET_NICKNAME = 'LW_f356w-f444w' 
DET_NICKNAME = 'SW_f150w-f200w'
DET_TYPE = 'noise-equal'
KERNEL = 'f160w'

PHOT_ZP = OrderedDict()
PHOT_ZP['f435w'] = 28.9
PHOT_ZP['f606w'] = 28.9
PHOT_ZP['f814w'] = 28.9
# PHOT_ZP['f098m'] = 28.9
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

# PHOT_NICKNAMES = 'None' # detection only!
# Set to 'None' if you don't want to run photometry

# SECONDARY PARAMETERS
DIR_OUTPUT = f'./data/output/v4/{DET_NICKNAME}_{DET_TYPE}_{KERNEL}/'
if not os.path.exists(DIR_OUTPUT):
    os.system(f'mkdir {DIR_OUTPUT}') # AUTO MAKES OUTPUT DIRECTORY IF IT DOESNT EXIST!

PATH_DETSCI = f'./data/intermediate/v4/{DET_NICKNAME}_{DET_TYPE}.fits.gz'
PATH_DETWHT = 'None'
PATH_DETMASK = 'None' # 1 is masked
HEADEXT = 1

DET_BACKPARAMS = dict(bw=64, bh=64, fw=8, fh=8, maskthresh=1, fthresh=0.)
DET_BACKTYPE = 'NONE' # VAR, GLOBAL, NONE
DETECTION_PARAMS = dict(
    thresh =  2,
    minarea = 10,
    kernelfwhm = 1.00170,
    deblend_nthresh = 16,
    deblend_cont = 0.00315,
    clean_param = 1.66776,
    )

PHOT_BACKPARAMS = dict(bw=64, bh=64, fw=8, fh=8, maskthresh=1, fthresh=0.)
PHOT_BACKTYPE = 'NONE' # VAR, GLOBAL, NONE

PHOT_APER = [0.16, 0.24, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0] # diameter in arcsec
PHOT_AUTOPARAMS = 2.5, 3.5 # for MAG_AUTO
PHOT_FLUXFRAC = 0.5, 0.6 # FLUX_RADIUS at 50% and 60% of flux


def conv_flux(in_zpt, out_zpt=25.0):
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
dethead = fits.getheader(PATH_DETSCI, HEADEXT)
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

# BACKGROUNDS
back = compute_background(detsci, None, DET_BACKTYPE, DET_BACKPARAMS)
detsci -= back

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
hdul.writeto(os.path.join(DIR_OUTPUT, f'{DET_NICKNAME}_SEGMAP.fits.gz'), overwrite=True)

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
bigreg.write(os.path.join(DIR_OUTPUT, f'{DET_NICKNAME}_OBJECTS.reg'), overwrite=True, format='ds9')

del detsci
del detwht
del detmask
del deterr

if PHOT_NICKNAMES == 'None':
    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    catalog.write(os.path.join(DIR_OUTPUT, f'{DET_NICKNAME}_DET_CATALOG.fits.gz'), overwrite=True)
    sys.exit()

areas = {}
stats = {}
for ind, PHOT_NICKNAME in enumerate(PHOT_NICKNAMES):

    print(PHOT_NICKNAME)

    PATH_PHOTSCI = glob.glob(f'./data/intermediate/v4/ceers-full-grizli-v4.0-{PHOT_NICKNAME}*_sci_skysubvar_{KERNEL}-matched.fits.gz')[0]
    PATH_PHOTHEAD = PATH_PHOTSCI
    PATH_PHOTWHT = glob.glob(f'./data/intermediate/v4/ceers-full-grizli-v4.0-{PHOT_NICKNAME}*_wht_{KERNEL}-matched.fits.gz')[0]
    PATH_PHOTMASK = 'None'
    HEADEXT = 1

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
    phothead = fits.getheader(PATH_PHOTHEAD, HEADEXT)
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


    # REGISTER BACKGROUNDS
    back = compute_background(photsci, None, PHOT_BACKTYPE, PHOT_BACKPARAMS)
    photsci -= back

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

        # # diagnostic region files
        # regs = []
        # for coord, obj in zip(detcoords, catalog):
        #     regs.append(CircleSkyRegion(coord, rad*u.arcsec))
        # regs = np.array(regs)
        # bigreg = Regions(regs[badflux])
        # bigreg.write(os.path.join(DIR_OUTPUT, f'{PHOT_NICKNAME}_{DET_NICKNAME}_BADFLUX{diam}_OBJECTS.reg'), overwrite=True, format='ds9')
        # bigreg = Regions(regs[badfluxerr])
        # bigreg.write(os.path.join(DIR_OUTPUT, f'{PHOT_NICKNAME}_{DET_NICKNAME}_BADFLUXERR{diam}_OBJECTS.reg'), overwrite=True, format='ds9')

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
        # print(intx, inty, np.shape(boxwht), srcmedwht[i], srcmeanwht[i])

    srcmeanwht[srcmeanwht<=0.] = np.nan
    srcmedwht[srcmedwht<=0.] = np.nan

    catalog['SRC_MEDWHT'] = srcmedwht
    catalog['SRC_MEANWHT'] = srcmeanwht
    medwht = np.nanmedian(photwht[photwht>0])
    catalog['MED_WHT'] = medwht

    # COMPUTE EMPTY APERTURE ERRORS + SAVE TO MASTER FILE
    empty_aper = list(PHOT_APER)+list(np.linspace(PHOT_APER[0], PHOT_APER[-1], 30))
    empty_aper = np.sort(empty_aper)
    plotname = os.path.join(DIR_OUTPUT, f'{PHOT_NICKNAME}_emptyaper.pdf')
    stats[PHOT_NICKNAME] = emtpy_apertures(photsci, segmap, N=int(1e3), aper=empty_aper, plotname=plotname)

    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    catalog.write(os.path.join(DIR_OUTPUT, f'{PHOT_NICKNAME}_{DET_NICKNAME}_PHOT_CATALOG.fits'), overwrite=True)

    # conv.jarvis(f'Photometry of {PHOT_NICKNAME} has finished!')

np.save(os.path.join(DIR_OUTPUT, f'{DET_NICKNAME}_emptyaper_stats.npy'), stats)
with open(os.path.join(DIR_OUTPUT, f'{DET_NICKNAME}_AREAS.dat'), 'w') as f:
    for filt in PHOT_NICKNAMES:
        area = areas[filt]
        f.write(f'{filt} {area}')
        f.write('\n')

conv.jarvis(f'Photometry of bands from {DET_NICKNAME} is complete')