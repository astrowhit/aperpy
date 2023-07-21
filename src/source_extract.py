
import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
from astropy.wcs import WCS, utils
import astropy.units as u
import sep
import os, sys, glob
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from regions import EllipseSkyRegion, Regions, CircleSkyRegion
from webb_tools import empty_apertures, compute_isofluxes, find_friends

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import TARGET_ZP, PHOT_APER, PHOT_AUTOPARAMS, PHOT_FLUXRADIUS, DETECTION_PARAMS, SKYEXT,\
         DIR_IMAGES, PHOT_ZP, FILTERS, DIR_OUTPUT, DIR_CATALOGS, IS_COMPRESSED, PIXEL_SCALE, PHOT_KRONPARAM,\
             USE_COMBINED_KRON_IMAGE, KRON_COMBINED_BANDS, KRON_ZPT, PHOT_EMPTYAPER_DIAMS

# MAIN PARAMETERS
DET_NICKNAME = sys.argv[2] #'LW_f277w-f356w-f444w'
KERNEL = sys.argv[3] # f444w or f160w or None


DET_TYPE = 'noise-equal'
FULLDIR_CATALOGS = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')
if not os.path.exists(FULLDIR_CATALOGS):
    os.mkdir(FULLDIR_CATALOGS)
    os.mkdir(os.path.join(FULLDIR_CATALOGS, 'figures/'))

# SECONDARY PARAMETERS
DETSCI_NAME = f'{DET_NICKNAME}_{DET_TYPE}/{DET_NICKNAME}_{DET_TYPE}.fits'
if IS_COMPRESSED:
     DETSCI_NAME +='.gz'
PATH_DETSCI = os.path.join(DIR_CATALOGS, DETSCI_NAME)
PATH_DETWHT = 'None'
PATH_DETMASK = 'None'


def conv_flux(in_zpt, out_zpt=TARGET_ZP):
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
area = np.sum((detwht!=0) & (detmask==0) & (~np.isnan(detsci))) * (pixel_scale  / 3600)**2
print(f'Area of detection image: {area} deg2')

# SOURCE DETECTION
print('SOURCE DETECTION...')
kerneldict = {}
if 'kerneltype' not in DETECTION_PARAMS.keys():
    kernel_func = Gaussian2DKernel
    kerneldict['x_stddev'] = DETECTION_PARAMS['kernelfwhm']/2.35
else:
    if DETECTION_PARAMS['kerneltype'] == 'gauss':
        kernel_func = Gaussian2DKernel
        kerneldict['x_stddev'] = DETECTION_PARAMS['kernelfwhm']/2.35

    elif DETECTION_PARAMS['kerneltype'] == 'tophat':
        kernel_func = Tophat2DKernel
        kerneldict['radius'] = DETECTION_PARAMS['kernelfwhm']



if 'kernelsize' in DETECTION_PARAMS.keys():
    kerneldict['x_size'] = DETECTION_PARAMS['kernelsize']
    kerneldict['y_size'] = DETECTION_PARAMS['kernelsize']

kerneldict['factor']=1

kernel = np.array(kernel_func(**kerneldict))
sep.set_extract_pixstack(10000000) # big image...
del DETECTION_PARAMS['kernelfwhm']
if 'kerneltype' in DETECTION_PARAMS.keys():
    del DETECTION_PARAMS['kerneltype']
if 'kernelsize' in DETECTION_PARAMS.keys():
    del DETECTION_PARAMS['kernelsize']
objects, segmap = sep.extract(
                detsci,
                err=deterr,
                filter_type='matched',
                filter_kernel=kernel,
                segmentation_map=True,
                **DETECTION_PARAMS
                )

print(f'Detected {len(objects)} objects.')

hdul = fits.HDUList()
hdul.append(fits.ImageHDU(name='SEGMAP', data=segmap, header=dethead))
SEGMAP_NAME = f'{DET_NICKNAME}_SEGMAP.fits'
if IS_COMPRESSED:
     SEGMAP_NAME +='.gz'
hdul.writeto(os.path.join(FULLDIR_CATALOGS, SEGMAP_NAME), overwrite=True)

# CLEAN UP
catalog = Table(objects)
catalog.add_column(Column(1+np.arange(len(catalog)), name='ID'), 0)
detcoords = detwcs.pixel_to_world(catalog['x'], catalog['y'])
catalog['RA'] = [c.ra for c in detcoords]
catalog['DEC'] = [c.dec for c in detcoords]

print('CONSTRUCTING ASSOCIATION TABLE OF NEIGHBORS...')
friends = find_friends(segmap)
import pickle
with open(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_friends.pickle'), 'wb') as handle:
    pickle.dump(friends, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('BUILDING REGION FILE...')
regs = []
for coord, obj in zip(detcoords, catalog):
    width = 2*obj['a'] * pixel_scale / 3600. * u.deg
    height = 2*obj['b'] * pixel_scale / 3600. * u.deg
    angle = np.rad2deg(obj['theta']) * u.deg
    objid = str(obj['ID'])
    regs.append(EllipseSkyRegion(coord, width, height, angle, meta={'label':objid}))
    # regs.append(PointSkyRegion(coord))
regs = np.array(regs)
bigreg = Regions(regs)
bigreg.write(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_OBJECTS.reg'), overwrite=True, format='ds9')

segmap[np.isnan(detsci)] = -99
# del detsci
del detwht
del detmask
del deterr

if FILTERS is None:
    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    DETCATALOG_NAME = f'{DET_NICKNAME}_DET_CATALOG.fits'
    if IS_COMPRESSED:
        DETCATALOG_NAME += '.gz'
    catalog.write(os.path.join(FULLDIR_CATALOGS, DETCATALOG_NAME), overwrite=True)
    sys.exit()

PATH_KRONSCI = None
if (KERNEL != 'None') & (USE_COMBINED_KRON_IMAGE):
    print(f"Constructing a noise-equalized co-add for Kron measurements based on {KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]]}")
    outpath = os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_KRON_K{KERNEL}')
    trypath = f'{outpath}_optavg.fits'
    if IS_COMPRESSED:
        trypath += '.gz'
    if os.path.exists(trypath):
        print('Kron image exists! I will use the existing one...')
        PATH_KRONSCI = f'{outpath}_optavg.fits' # inverse variance image!!!
        PATH_KRONERR = f'{outpath}_opterr.fits' # inverse variance image!!!
        if IS_COMPRESSED:
            PATH_KRONSCI += '.gz'
            PATH_KRONERR += '.gz'
    else:
        from build_detection import noise_equalized
        science_fnames = {}
        weight_fnames = {}

        # gather directories
        for PHOT_NICKNAME in KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]]:
            ext=f'_{KERNEL}-matched'
            dir_weight = DIR_OUTPUT
            print(DIR_OUTPUT)
            PHOTSCI_NAME = f'*{PHOT_NICKNAME}*_sci{SKYEXT}{ext}.fits'
            PHOTWHT_NAME = f'*{PHOT_NICKNAME}*_wht{ext}.fits'
            if IS_COMPRESSED:
                PHOTSCI_NAME += '.gz'
                PHOTWHT_NAME += '.gz'
            science_fnames[PHOT_NICKNAME] = glob.glob(os.path.join(DIR_OUTPUT, PHOTSCI_NAME))[0]
            weight_fnames[PHOT_NICKNAME] = glob.glob(os.path.join(DIR_OUTPUT, PHOTWHT_NAME))[0]

        print(science_fnames)

        # run it
        noise_equalized(KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]], outpath,
                    science_fnames= science_fnames,
                    weight_fnames= weight_fnames,
                    is_compressed=IS_COMPRESSED)
        PATH_KRONSCI = f'{outpath}_optavg.fits' # inverse variance image!!!
        PATH_KRONERR = f'{outpath}_opterr.fits' # inverse variance image!!!
        if IS_COMPRESSED:
            PATH_KRONSCI += '.gz'
            PATH_KRONERR += '.gz'
        print(f'Wrote Kron image to {PATH_KRONSCI}')


areas = {}
stats = {}

KRON_MATCH_BAND = None
USE_FILTERS = FILTERS
if (KERNEL != 'None') & (USE_COMBINED_KRON_IMAGE):
    KRON_MATCH_BAND = '+'.join(KRON_COMBINED_BANDS[DET_NICKNAME.split('_')[0]])
    if '+' not in KRON_MATCH_BAND:
        KRON_MATCH_BAND = 'sb-' + KRON_MATCH_BAND
    USE_FILTERS = [KRON_MATCH_BAND, ] + list(FILTERS)


for ind, PHOT_NICKNAME in enumerate(USE_FILTERS):

    print(PHOT_NICKNAME)
    if PHOT_NICKNAME != KRON_MATCH_BAND:
        skyext = SKYEXT #'_skysubvar'
        ext = ''
        dir_weight = DIR_IMAGES
        if KERNEL != 'None':
            ext=f'_{KERNEL}-matched'
            dir_weight = DIR_OUTPUT
        print(DIR_OUTPUT)
        PHOTSCI_NAME = f'*{PHOT_NICKNAME}*_sci{skyext}{ext}.fits'
        PHOTWHT_NAME = f'*{PHOT_NICKNAME}*_wht{ext}.fits'
        if IS_COMPRESSED:
            PHOTSCI_NAME += '.gz'
            PHOTWHT_NAME += '.gz'
        print(PHOTSCI_NAME)
        PATH_PHOTSCI = glob.glob(os.path.join(DIR_OUTPUT, PHOTSCI_NAME))[0]
        PATH_PHOTHEAD = PATH_PHOTSCI
        PATH_PHOTWHT = glob.glob(os.path.join(dir_weight, PHOTWHT_NAME))[0]
        PATH_PHOTMASK = 'None'

        PHOT_ZPT = PHOT_ZP[PHOT_NICKNAME.lower()] #calc_zpt(PHOT_NICKNAME)
        print(f'Zeropoint for {PHOT_NICKNAME}: {PHOT_ZPT}')

        # 2 FORCED PHOTOMETRY + measurements
        # READ IN IMAGES
        print('READING PHOTOMETRY IMAGES...')
        photsci = fits.getdata(PATH_PHOTSCI).byteswap().newbyteorder()
        print(PATH_PHOTSCI)
        photwht = fits.getdata(PATH_PHOTWHT).byteswap().newbyteorder()
        photmask = np.where((photwht<=0.)|~np.isfinite(photwht), 1., 0.)
        print(PATH_PHOTWHT)
        if PATH_PHOTMASK != 'None':
            photmask_user = fits.getdata(PATH_PHOTMASK).byteswap().newbyteorder().astype(float)
            print(PATH_PHOTMASK)
            photmask[photmask_user] = 1.

        phothead = fits.getheader(PATH_PHOTHEAD, 0)
        photwcs = WCS(phothead)
        print(photwcs)

        photerr = np.where((photwht==0) | np.isnan(photwht), np.inf, 1./np.sqrt(photwht))
        photerr[~np.isfinite(photerr)] = np.median(photerr[np.isfinite(photerr)]) # HACK fill in holes with median weight.
        # fits.ImageHDU(photerr).writeto('PHOTERR.fits')

    # IMAGE FOR KRON RADII
    # We actually run AUTO fluxes on each band
    # So just do again for each band and take their coverage -- uber consistent this way.)
    elif PHOT_NICKNAME == KRON_MATCH_BAND:
            photsci = fits.getdata(PATH_KRONSCI).byteswap().newbyteorder()
            photerr = fits.getdata(PATH_KRONERR).byteswap().newbyteorder()
            photwht = np.where(photerr<=0., 0, 1/(photerr**2))
            phothead = fits.getheader(PATH_KRONSCI, 0)
            photwcs = WCS(phothead)
            print(photwcs)
            PHOT_ZPT = KRON_ZPT
            photmask = np.where((photerr<=0.)|~np.isfinite(photerr), 1., 0.)


    # SOME BASIC INFO
    pixel_scale = utils.proj_plane_pixel_scales(photwcs)[0] * 3600
    print(f'Pixel scale: {pixel_scale}')
    area = np.sum(np.isfinite(photwht) & (photwht > 0.) & ~np.isnan(detsci)) * (pixel_scale  / 3600)**2
    print(f'Usable area of photometry image: {area} deg2')
    areas[PHOT_NICKNAME] = area

    # Compute isophotal fluxes based on segmentation
    print(f"{PHOT_NICKNAME} :: MEASURING PHOTOMETRY in isophotal segments...")
    isofluxes = compute_isofluxes(segmap.ravel().astype(np.int64), photsci.ravel().astype(np.float64))
    catalog[f'FLUX_ISO'] = isofluxes * conv_flux(PHOT_ZPT)

    # Hack the x,y coords
    xphot,yphot = photwcs.wcs_world2pix(catalog['RA'], catalog['DEC'],1)

    # APERTURE PHOTOMETRY -- NOTE: we do NOT apply ANY masking here. Color apertures are small enough.
    sep_fluxes = {}
    for diam in PHOT_APER:
        rad = diam / 2. / pixel_scale
        print(f"{PHOT_NICKNAME} :: MEASURING PHOTOMETRY in {diam:2.2f}\" apertures... ({2*rad:2.1f} px)")
        flux, fluxerr, flag = sep.sum_circle(photsci, xphot, yphot, #objects['x'], objects['y'],
                                            mask = photmask,
                                            err = photerr, subpix=0,
                                            r=rad, gain=1.0)
        
        badflux = (flux == 0.) | ~np.isfinite(flux) | (flag > 0)
        badfluxerr = (fluxerr <= 0.) | ~np.isfinite(fluxerr) | (flag > 0)
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

        sep_fluxes[diam] = (flux, fluxerr, flag)

        # show the first 10 objects results:
        for i in range(3):
            print("object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))

        catalog[f'FLUX_APER{diam}'] = flux * conv_flux(PHOT_ZPT)
        catalog[f'FLUXERR_APER{diam}'] = fluxerr * conv_flux(PHOT_ZPT)
        # catalog[f'MAG_APER{diam}'] = PHOT_ZPT - 2.5*np.log10(flux)
        # catalog[f'MAGERR_APER{diam}'] = 2.5 / np.log(10) / ( flux / fluxerr )
        catalog[f'FLAG_APER{diam}'] = flag

    # Compute Kron, flux radii with and without segmap masking
    for seg, seg_id, ext in ((None, None, ''), (segmap, catalog['ID'], '_masked')):
        # KRON RADII AND MAG_AUTO
        print(f"{PHOT_NICKNAME} :: MEASURING PHOTOMETRY in kron-corrected AUTO apertures...")
        kronrad, krflag = sep.kron_radius(photsci, xphot, yphot, #catalog['x'], catalog['y'],
                                            catalog['a'], catalog['b'], catalog['theta'], PHOT_KRONPARAM,
                                            mask=photmask,
                                            segmap=seg, seg_id=seg_id) # SE uses 6
        kronrad[np.isnan(kronrad)] = 0.
        kronrad *= PHOT_AUTOPARAMS[0]
        kronrad = np.maximum(kronrad, PHOT_AUTOPARAMS[1])
        # print(np.isnan(kronrad).sum(), np.max(kronrad), np.min(kronrad))
        catalog['theta'][catalog['theta'] > np.pi / 2.] = np.pi / 2. # numerical rounding correction!
        flux, fluxerr, flag = sep.sum_ellipse(photsci, xphot, yphot, #catalog['x'], catalog['y'],
                                            catalog['a'], catalog['b'], catalog['theta'], kronrad,
                                            err = photerr, mask=photmask,
                                            subpix=0, segmap=seg, seg_id=seg_id)

        badflux = (flux == 0.) | ~np.isfinite(flux) #| (flag > 0)
        badfluxerr = (fluxerr <= 0.) | ~np.isfinite(fluxerr) #| (flag > 0)
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

        flag |= krflag  # combine flags into 'flag'

        kronrad_circ = kronrad * np.sqrt(catalog['a'] * catalog['b'])

        catalog[f'FLUX_AUTO{ext}'] = flux * conv_flux(PHOT_ZPT)
        catalog[f'FLUXERR_AUTO{ext}'] = fluxerr * conv_flux(PHOT_ZPT)
        # catalog[f'MAG_AUTO{ext}'] = PHOT_ZPT - 2.5*np.log10(flux)
        # catalog[f'MAGERR_AUTO{ext}'] = 2.5 / np.log(10) / ( flux / fluxerr )
        catalog[f'KRON_RADIUS{ext}'] = kronrad
        catalog[f'KRON_RADIUS_CIRC{ext}'] = kronrad_circ
        catalog[f'FLAG_AUTO{ext}'] = flag
        catalog[f'FLAG_KRON_RADIUS{ext}'] = krflag


        # FLUX RADIUS
        print(f"{PHOT_NICKNAME} :: MEASURING FLUX RADIUS...")
        """In Source Extractor, the FLUX_RADIUS parameter gives the radius of a circle enclosing a desired fraction of the total flux."""
        r, flag = sep.flux_radius(photsci,  xphot, yphot, 6.*catalog['a'],
                                PHOT_FLUXRADIUS, mask=photmask,
                                normflux=flux, subpix=5, segmap=seg, seg_id=seg_id)
        rt = r.T
        for i, fluxfrac in enumerate(PHOT_FLUXRADIUS):
            catalog[f'FLUX_RADIUS_FRAC{fluxfrac}{ext}'] = rt[i]
        catalog[f'FLAG_FLUX_RADIUS{ext}'] = flag


    # SOURCE WEIGHT + MEDIAN WEIGHT
    photwht_corr = photwht / (conv_flux(PHOT_ZPT)**2) # puts all weights (incl. errors later on) in the requested target units!
    srcmedwht = np.nan * np.ones(len(catalog))
    srcmeanwht = np.nan * np.ones(len(catalog))
    for i, (ixphot, iyphot) in enumerate(zip(xphot, yphot)):
        intx, inty = int(ixphot), int(iyphot)
        boxwht = photwht_corr[inty-4:inty+5, intx-4:intx+5]
        srcmedwht[i] = np.nanmedian(boxwht[boxwht>0])
        srcmeanwht[i] = np.nanmean(boxwht[boxwht>0])

    srcmeanwht[srcmeanwht<=0.] = np.nan
    srcmedwht[srcmedwht<=0.] = np.nan

    catalog['SRC_MEDWHT'] = srcmedwht
    catalog['SRC_MEANWHT'] = srcmeanwht
    catalog['MED_WHT'] = np.nanmedian(photwht_corr[photwht_corr>0])
    catalog['MAX_WHT'] = np.nanpercentile(photwht_corr[photwht_corr>0], 99)

    # COMPUTE EMPTY APERTURE ERRORS + SAVE TO MASTER FILE
    empty_aper = list(PHOT_APER)+list(PHOT_EMPTYAPER_DIAMS)
    empty_aper = np.sort(empty_aper)
    plotname = os.path.join(FULLDIR_CATALOGS, f'figures/{PHOT_NICKNAME}_K{KERNEL}_emptyaper.pdf')
    # zpt_factor = conv_flux(PHOT_ZPT)
    noise_equal = photsci * np.sqrt(photwht)
    noise_equal[photwht<=0] = 0.
    stats[PHOT_NICKNAME] = empty_apertures(noise_equal, photwht, segmap, N=int(1e4), pixscl=PIXEL_SCALE,
                                        aper=empty_aper, plotname=plotname)

    # WRITE OUT
    print(f'DONE. Writing out catalog.')
    catalog.write(os.path.join(FULLDIR_CATALOGS, f'{PHOT_NICKNAME}_{DET_NICKNAME}_K{KERNEL}_PHOT_CATALOG.fits'), overwrite=True)

np.save(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_emptyaper_stats.npy'), stats)
with open(os.path.join(FULLDIR_CATALOGS, f'{DET_NICKNAME}_K{KERNEL}_AREAS.dat'), 'w') as f:
    for filt in FILTERS:
        area = areas[filt]
        f.write(f'{filt} {area}')
        f.write('\n')
