# This is just a wrapper around the webb_tool
# from webb_tools import get_webbpsf, get_date
from astropy.io import fits
import numpy as np
import os, sys
from astropy.nddata import block_reduce
from photutils.psf.matching import create_matching_kernel, SplitCosineBellWindow
from scipy.ndimage import zoom
from astropy.visualization import simple_norm
from photutils.centroids import centroid_2dg
from astropy.convolution import convolve_fft


PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_PSFS, PIXEL_SCALE, DIR_IMAGES, DIR_OUTPUT, FILTERS, PHOT_ZP, \
                MATCH_BAND, SKYEXT, DIR_KERNELS, MAGLIM, PSF_DICT, OVERWRITE
from psf_tools import *


outdir = DIR_PSFS
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(DIR_KERNELS):
    os.mkdir(DIR_KERNELS)
plotdir = os.path.join(outdir,'../diagnostics/')
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

target_filter = MATCH_BAND
image_dir = DIR_OUTPUT
print('target filter',target_filter)
# hdr = fits.getheader(glob.glob(os.path.join(DIR_OUTPUT, f'*{target_filter}*sci{SKYEXT}.fits*'))[0])
if SKYEXT == '':
    filename = glob.glob(os.path.join(DIR_IMAGES, f'*{target_filter}*sci{SKYEXT}.fits*'))[0]
else:
    filename = glob.glob(os.path.join(DIR_OUTPUT, f'*{target_filter}*sci{SKYEXT}.fits*'))[0]
hdr = fits.getheader(filename)


### Make ePSFs
use_filters = [MATCH_BAND] + [f for f in FILTERS if f != MATCH_BAND]

for pfilt in use_filters:

    psfname = os.path.join(DIR_PSFS, f'{pfilt}_psf.fits')
    if not OVERWRITE and os.path.exists(psfname):
        print(f'{pfilt} kernel already exists, will not overwrite.\n'
              'Check OVERWRITE param in config if this is not the desired effect.')
        continue

    print()
    print(f'Finding stars for {pfilt}...')
    if SKYEXT == '':
        filename = glob.glob(os.path.join(DIR_IMAGES, f'*{pfilt}*sci{SKYEXT}.fits*'))[0]
    else:
        filename = glob.glob(os.path.join(DIR_OUTPUT, f'*{pfilt}*sci{SKYEXT}.fits*'))[0]
    suffix = '.fits' + filename.split('.fits')[-1]
    starname = filename.replace(suffix, '_star_cat.fits').replace(DIR_OUTPUT, DIR_PSFS)
    outname = os.path.join(DIR_PSFS, f'{pfilt}.fits')

    # if len(glob.glob(DIR_PSFS+'*'+pfilt.lower()+'*'+'psf.fits')) > 0:
    #     print(f'PSFs already exist for {pfilt} -- skipping!')
    #     if pfilt == target_filter:
    #         target_psf = fits.getdata(glob.glob(DIR_PSFS+'*'+target_filter.lower()+'*'+'psf.fits')[0])

    print(filename)
    print(starname)

    cutout_size = PSF_DICT['cutout_size'][pfilt]
    range = PSF_DICT['range'][pfilt]
    threshold_max = PSF_DICT['threshold_max'][pfilt]
    mag_lim = PSF_DICT['mag_lim'][pfilt]
    snr_lim = PSF_DICT['snr_lim'][pfilt]
    sigma = PSF_DICT['sigma'][pfilt]
    npeaks = PSF_DICT['npeaks'][pfilt]
    aper_scale = PSF_DICT['aper_scale'][pfilt]
    try:
        radii = PSF_DICT['radii'][pfilt]
    except:
        radii = np.array([0.5,1.,2.,4.,7.5]) * aper_scale
    print(f"apertures={radii}")

    showme = False

    peaks, stars = find_stars(filename, outdir=DIR_PSFS, plotdir=plotdir,
        label=pfilt, zp=PHOT_ZP[pfilt], range=range, radii=radii,
        threshold_max=threshold_max, mag_lim=mag_lim, npeaks=npeaks)
    
    print(f'Found {len(peaks)} bright sources')

    maglim = MAGLIM
    ok = (peaks['mag'] > maglim[0]) & ( peaks['mag'] < maglim[1] )
    ra, dec, ids = peaks['ra'][ok], peaks['dec'][ok], peaks['id'][ok]

    print(f'Processing PSF...')
    pixsize=int(cutout_size/PIXEL_SCALE)
    if pixsize % 2 == 0: pixsize+=1
    if pixsize < 101: pixsize=101
    psf = PSF(image=filename, x=ra, y=dec, ids=ids, pixsize=pixsize, pixelscale=PIXEL_SCALE)
    psf.center()
    psf.measure()
    psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.99, showme=showme, nsig=30)
    psf.stack(sigma=sigma)
    # psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.4, showme=True, nsig=30)
    psf.save(outname.replace('.fits',''))

    psfmodel = renorm_psf(psf.psf_average, filt=pfilt)
    fits.writeto('_'.join([outname.replace('.fits',''), 'psf_norm.fits']), np.array(psfmodel),overwrite=True)
    imshow(psf.data[psf.ok],nsig=50,title=psf.cat['id'][psf.ok].data)
    plt.savefig(outname.replace('.fits','_stamps_used.pdf').replace(outdir,plotdir),dpi=300)
    show_cogs([psf.psf_average],title=pfilt, label=['oPSF'],outname=plotdir+pfilt)
    plots=glob.glob(outdir+'*.pdf')
    plots+=glob.glob(outdir+'*_cat.fits')
    plots+=glob.glob(outdir+'*_loc.reg')
    for plot in plots:
        os.rename(plot,plot.replace(outdir,plotdir))
    

### Make matching kernels
for pfilt in use_filters:
    if pfilt == MATCH_BAND:
        continue

    psfname = os.path.join(DIR_PSFS, f'{pfilt}_psf.fits')
    outname = psfname.replace('psf','kernel').replace(DIR_PSFS, DIR_KERNELS)
    if not OVERWRITE and os.path.exists(outname):
        print(f'{pfilt} kernel already exists, will not overwrite.\n'
              'Check OVERWRITE param in config if this is not the desired effect.')
        continue

    target_psf = fits.getdata(glob.glob(DIR_PSFS+'*'+target_filter.lower()+'*'+'psf.fits')[0])

    filt_psf = fits.getdata(psfname)

    method = PSF_DICT['method'][pfilt]
    oversample = PSF_DICT['oversample'][pfilt]
    pypher_r = PSF_DICT['pypher_r'][pfilt]
    alpha = PSF_DICT['alpha'][pfilt]
    beta = PSF_DICT['beta'][pfilt]

    if oversample > 1:
        print(f'Oversampling PSF by {oversample}x...')
        filt_psf = zoom(filt_psf, oversample)
        target_psf = zoom(target_psf, oversample)

    print(f'Normalizing PSF to unity...')
    filt_psf /= filt_psf.sum()
    target_psf /= target_psf.sum()

    print(f'Building {pfilt}-->{MATCH_BAND} kernel...')
    assert(filt_psf.shape == target_psf.shape, f'Shape of filter psf ({filt_psf.shape}) must match target psf ({target_psf.shape})')
    if method == 'pypher':
        fits.writeto(DIR_KERNELS+'psf_a.fits',filt_psf,header=hdr,overwrite=True)
        fits.writeto(DIR_KERNELS+'psf_b.fits',target_psf,header=hdr,overwrite=True)
        os.system(f'pypher {DIR_KERNELS}psf_a.fits {DIR_KERNELS}psf_b.fits {DIR_KERNELS}kernel_a_to_b.fits -r {pypher_r:.3g}')
        kernel = fits.getdata(DIR_KERNELS+'kernel_a_to_b.fits')
        os.remove(DIR_KERNELS+'psf_a.fits')
        os.remove(DIR_KERNELS+'psf_b.fits')
        os.remove(DIR_KERNELS+'kernel_a_to_b.fits')
        os.remove(DIR_KERNELS+'kernel_a_to_b.log')

    elif method == 'photutils':
        window = SplitCosineBellWindow(alpha=alpha, beta=beta)
        kernel =  create_matching_kernel(filt_psf, target_psf, window=window)
    
    else:
        print("PSF_DICT['method'] must be 'pypher' or 'photutils'. See config.")
        raise

    if oversample > 1:
        kernel = block_reduce(kernel,block_size=oversample, func=np.sum)
        kernel /= kernel.sum()

    print(f'Writing {pfilt}-->{MATCH_BAND} kernel to {outname.lower()}')
    fits.writeto(outname.lower(), np.float32(np.array(kernel/kernel.sum())),overwrite=True)

nfilt = len(use_filters[1:])
plt.figure(figsize=(30,nfilt*4))
npanel = 7

target_psf = fits.getdata(glob.glob(DIR_PSFS+'*'+MATCH_BAND.lower()+'*'+'psf.fits')[0])
target_psf /= target_psf.sum()

print(f'Plotting kernel checkfile...')
for i, pfilt in enumerate(use_filters[1:]):

    print(DIR_PSFS)
    print(pfilt.lower())
    psfname = glob.glob(DIR_PSFS+'*'+pfilt.lower()+'*'+'psf.fits')[0]
    outname = DIR_KERNELS+os.path.basename(psfname).replace('psf','kernel')

    filt_psf = fits.getdata(psfname)
    filt_psf /= filt_psf.sum()

    kernel = fits.getdata(outname.lower())

    simple = simple_norm(kernel,stretch='linear',power=1, vmin=-5e-4, vmax=5e-4)

    plt.subplot(nfilt,npanel,1+i*npanel)
    plt.title('psf '+pfilt)
    plt.imshow(filt_psf, norm=simple, interpolation='antialiased',origin='lower')
    plt.subplot(nfilt,npanel,2+i*npanel)
    plt.title('target psf '+target_filter)
    plt.imshow(target_psf, norm=simple, interpolation='antialiased',origin='lower')
    plt.subplot(nfilt,npanel,3+i*npanel)
    plt.title("kernel "+pfilt)

    plt.imshow(kernel, norm=simple, interpolation='antialiased',origin='lower')

    filt_psf_conv = convolve_fft(filt_psf, kernel)

    plt.subplot(nfilt,npanel,4+i*npanel)
    plt.title("convolved "+pfilt)
    plt.imshow(filt_psf_conv, norm=simple, interpolation='antialiased',origin='lower')

    plt.subplot(nfilt,npanel,5+i*npanel)
    plt.title("residual "+pfilt)
    res = filt_psf_conv-target_psf
    plt.imshow(res, norm=simple, interpolation='antialiased',origin='lower')

    plt.subplot(nfilt,npanel,7+i*npanel)
    r,pf,pt = plot_profile(filt_psf_conv,target_psf)
    plt.plot(r*PIXEL_SCALE, pf/pt)
    fr,fpf,fpt = plot_profile(filt_psf,target_psf)
    plt.plot(fr*PIXEL_SCALE, fpf/fpt)
    plt.ylim(0.95,1.05)

    method = PSF_DICT['method'][pfilt]
    pypher_r = PSF_DICT['pypher_r'][pfilt]
    alpha = PSF_DICT['alpha'][pfilt]
    beta = PSF_DICT['beta'][pfilt]
    if method == 'pypher':
        plt.title('pypher r={}'.format(pypher_r))
    elif method == 'photutils':
        plt.title('alpha={}, beta={}'.format(alpha,beta))

    plt.axvline(x=0.16,ls=':')
    plt.xlabel('radius arcsec')
    plt.ylabel('ee_psf_conv / ee_psf_target')

    plt.subplot(nfilt,npanel,6+i*npanel)
    plt.title('COG / COG_target')
    plt.plot(r*PIXEL_SCALE,pf,lw=3)
    plt.plot(r*PIXEL_SCALE,pt,'--',alpha=0.7,lw=3)
    plt.xlabel('radius arcsec')
    plt.ylabel('ee')

plt.tight_layout()
plt.savefig(plotdir+'kernels.pdf',dpi=300)
