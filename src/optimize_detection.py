from distutils.command.clean import clean
import os
from typing import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import sep
import glob
from astropy.io import fits, ascii
from astropy.table import Table
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm, Normalize
import emcee
import sys
import corner
from astropy.convolution import Gaussian2DKernel
from multiprocessing import Pool
import convenience as conv

from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)

plt.ioff()
sep.set_extract_pixstack(10000000) # big image...

PLOT = 1
max_sep = 0.1 * u.arcsec
size = 0.5 * u.arcmin
max_n = 50
ncpu = 4
nwalkers = 100

# Load in the data
BAND = 'F150W'
WORKING_DIR = '/Volumes/External1/Projects/Current/CEERS_Counts/data/' 
img_fname = os.path.join(WORKING_DIR, f'external/CUTOUT_2-ceers5_{BAND}_i2d_sci.fits')
wht_fname = os.path.join(WORKING_DIR, f'external/CUTOUT_2-ceers5_{BAND}_i2d_wht.fits')
print(f'Loading in {BAND}...')
img = fits.getdata(img_fname)
raw_wcs = WCS(fits.getheader(img_fname))
ny, nx = np.shape(img)
center_coord = raw_wcs.pixel_to_world(nx/2., ny/2.)
cutout = Cutout2D(img, wcs=raw_wcs, position=center_coord, size=size, copy=True)
img = cutout.data
wcs = cutout.wcs
print(img_fname)
wht = fits.getdata(wht_fname)
cutout = Cutout2D(wht, wcs=raw_wcs, position=center_coord, size=size, copy=True)
wht = cutout.data
print(wht_fname)

# Load in truth
truth = Table.read(os.path.join(WORKING_DIR, 'intermediate/CEERS_SDR3_SAM_input_small.fits'))
print('loaded truth.')
bright = truth[f'NIRCam_{BAND}'] < 28.9
truth = truth[bright]
tcoords = SkyCoord(truth['ra']*u.deg, truth['dec']*u.deg)
in_foot = [wcs.footprint_contains(c) for c in tcoords]
compare = in_foot
tcoords = tcoords[compare]
truth = truth[compare]
print('cleaned truth.')

# Modify the data
img = img.byteswap().newbyteorder()
wht = wht.byteswap().newbyteorder()
mask = wht == 0
mask.astype(np.int32) # SEP wants 0 and 1

err = np.where(wht==0, 0, 1/np.sqrt(wht))
rms = np.nanmedian(err[err>0])
print('byteswap done.')

# Visualize it
if PLOT > 0:
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    axes[0].imshow(img, norm=LogNorm(rms, 10*rms), cmap='Greys')
    axes[1].imshow(wht, cmap='Greys')
    axes[2].imshow(mask)

    plt.savefig(f'output/{BAND}_input.pdf')

    plt.close('all')


# Estimate background
bkg = sep.Background(img, mask=mask, maskthresh=1, bw=64, bh=64, fw=3, fh=3, fthresh=0.)
img -= bkg.globalback
print(f'Subtracting global background {bkg.globalback:2.2f}')

# initalize walkers
init_kernelfwhm = 1
init_thresh = 2
init_minarea = 10
init_deblend_nthresh = 8
init_deblend_cont = 1e-2
init_clean_param = 3.0
init_theta = np.array([init_thresh, init_minarea, init_kernelfwhm, init_deblend_nthresh, init_deblend_cont, init_clean_param])

# threshes = [1.5,]
# minareas = [5,]
# deblend_nthreshes = [16,]
# deblend_conts = [1e-2,]
# clean_params = [0.0,]

def log_prior(theta):
    thresh, minarea, kernelfwhm, deblend_nthresh, deblend_cont, clean_param = theta
    if thresh > 1 and thresh < 5 and \
        minarea > 5 and minarea < 25 and \
        kernelfwhm > 1 and kernelfwhm < 10 and \
        deblend_nthresh > 6 and deblend_nthresh < 128 and \
        deblend_cont > 0 and deblend_cont < 0.02 and \
        clean_param > 0 and clean_param < 15:
        return 0.0
    return -np.inf
                            
# functions
def process(theta):
    thresh, minarea, kernelfwhm, deblend_nthresh, deblend_cont, clean_param = theta

    kernel = np.array(Gaussian2DKernel(kernelfwhm/2.35, factor=1))
    print(theta)    
    try:
    # if True:
        objects = sep.extract(img, thresh=thresh, err=err, mask=mask, minarea=minarea,
                                filter_kernel=kernel, filter_type='matched',
                                deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean=True,
                                clean_param=clean_param, segmentation_map=False)
        print(f'Found N = {len(objects)} objects')
    except:
        print('Failed!')
        return -np.inf, -np.inf, -np.inf

    objects = Table(objects)

    c = wcs.pixel_to_world(objects['x'], objects['y'])
    objects['ra'] = [j.ra for j in c]
    objects['dec'] = [j.dec for j in c]

    # compare with input for precision and recall
    coords = SkyCoord(objects['ra'], objects['dec'])
    idx2, d2d, d3d = coords.match_to_catalog_sky(tcoords)
    sep_constraint = d2d < max_sep
    mcatalog = coords[sep_constraint]
    mtruth = truth[idx2[sep_constraint]]
    pc_above = len(mcatalog) / len(truth)
    nmatch = len(mcatalog)
    print(f'  ... with N = {nmatch} matches within {max_sep} ({100*pc_above:2.2f}% < 28.9AB)') 
    if nmatch == 0:
        return -np.inf, -np.inf, -np.inf

    nobj = len(objects)
    recall = nmatch / len(truth) # TP / (TP + FN)
    precision = nmatch / nobj # TP / (TP + FP)
    fscore =  (precision * recall) / (0.5 * (precision + recall))       # (Precision x Recall)/[(Precision + Recall)/2]
    print(f'  ... recall = {recall:2.2f} | precision = {precision:2.2f} | fscore = {fscore:2.2f}')

    del objects
    return recall, precision, fscore

def process_total(theta, img, err, mask, truth):
    # Why an additional function? Because MCMC doesn't play nice with scope!
    thresh, minarea, kernelfwhm, deblend_nthresh, deblend_cont, clean_param = theta

    kernel = np.array(Gaussian2DKernel(kernelfwhm/2.35, factor=1))
    print(theta)    
    objects, segmap = sep.extract(img, thresh=thresh, err=err, mask=mask, minarea=minarea,
                            filter_kernel=kernel, filter_type='matched',
                            deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean=True,
                            clean_param=clean_param, segmentation_map=True)
    print(f'Found N = {len(objects)} objects')

    objects = Table(objects)

    c = wcs.pixel_to_world(objects['x'], objects['y'])
    objects['ra'] = [j.ra for j in c]
    objects['dec'] = [j.dec for j in c]

    # compare with input for precision and recall
    coords = SkyCoord(objects['ra'], objects['dec'])
    idx2, d2d, d3d = coords.match_to_catalog_sky(tcoords)
    sep_constraint = d2d < max_sep
    mcatalog = coords[sep_constraint]
    mtruth = truth[idx2[sep_constraint]]
    pc_above = len(mcatalog) / len(truth)
    nmatch = len(mcatalog)
    print(f'  ... with N = {nmatch} matches within {max_sep} ({100*pc_above:2.2f}% < 28.9AB)') 
    if nmatch == 0:
        return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

    nobj = len(objects)
    recall = nmatch / len(truth) # TP / (TP + FN)
    precision = nmatch / nobj # TP / (TP + FP)
    fscore =  (precision * recall) / (0.5 * (precision + recall))       # (Precision x Recall)/[(Precision + Recall)/2]
    print(f'  ... recall = {recall:2.2f} | precision = {precision:2.2f} | fscore = {fscore:2.2f}')

    return objects, segmap, recall, precision, fscore, truth, mtruth

def lnp_maxrecall(theta):
    lp = log_prior(theta)
    recall, __, __ = process(theta)
    return lp + recall

def lnp_maxprecision(theta):
    lp = log_prior(theta)
    __, precision, __ = process(theta)
    return lp + precision

def lnp_maxfscore(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return lp
    __, __, fscore = process(theta)
    print(lp, fscore, lp + fscore)
    return lp + np.exp(fscore)

pos = init_theta + 0.1 * np.random.randn(nwalkers, len(init_theta)) * init_theta
nwalkers, nparam = pos.shape

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

plabel = 'maxfscore'
func = lnp_maxfscore

print(f'Spinning up {ncpu} CPUs')
with Pool(processes=ncpu) as pool:
# ncpu = 'NaN'
# if True:
    print(f'Constructing samplers ({nwalkers} walkers over {nparam} parameters)')
    #conv.jarvis(f'Constructing samplers ({nwalkers} walkers over {nparam} parameters)')

    sampler = emcee.EnsembleSampler(nwalkers, nparam, func, pool=pool)

    print(f'Running Markov Chains...({max_n} max steps on {ncpu} CPUs)')
    #conv.jarvis(f'Running Markov Chains...({max_n} max steps on {ncpu} CPUs)')
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 10 steps
        # if sampler.iteration  == 5:
        #     break
            # continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 10 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        print(f'Converged? {converged} ({np.sum(tau*10 < sampler.iteration)}/{len(tau)} < {sampler.iteration}, max {np.nanmax(tau*10):2.2f})')
        if converged:
            break
        old_tau = tau

    if sampler.iteration == max_n:
        print('WARNING: Chains did not converge')
        #conv.jarvis('WARNING: Chains did not converge')

# tau = sampler.get_autocorr_time()
if np.isnan(tau).all(): tau = 0
burnin = int(2 * np.nanmax(tau))
burnin = 0
thin = int(0.5 * np.nanmin(tau))
thin = 1

samples = sampler.get_chain(discard=burnin, flat=False, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=False, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=False, thin=thin)
log_prior_samples = np.nan * log_prob_samples

all_samples = np.concatenate(
    (samples, log_prob_samples[:, :, None]), axis=2
)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))

labels = ["thresh", "minarea", "kernel", "deblend_nthresh", "deblend_cont", "clean_param", "score"]

idx_ml = np.argmax(all_samples[-1, :, -1])
like_ml = all_samples[-1, idx_ml, -1]
maxn = len(all_samples)
maxml_theta = all_samples[-1, idx_ml][:nparam]
print(f'Maximum Score: {like_ml:2.2f} ({idx_ml}/{maxn})')
for (label, value) in zip(labels, maxml_theta): 
    print(f'{label}: {value:2.5f}')
# #conv.jarvis('Detection optimization completed. Running plots.')


# Load in the data so we can apply on entire mosaic!
print(f'Loading in {BAND}...')
max_sep = 0.1 * u.arcsec
size = 1.5 * u.arcmin
img = fits.getdata(img_fname)
raw_wcs = WCS(fits.getheader(img_fname))
ny, nx = np.shape(img)
center_coord = raw_wcs.pixel_to_world(nx/2., ny/2.)
cutout = Cutout2D(img, wcs=raw_wcs, position=center_coord, size=size, copy=True)
img = cutout.data
wcs = cutout.wcs
print(img_fname)
wht = fits.getdata(wht_fname)
cutout = Cutout2D(wht, wcs=raw_wcs, position=center_coord, size=size, copy=True)
wht = cutout.data
print(wht_fname)

# Modify the data
img = img.byteswap().newbyteorder()
wht = wht.byteswap().newbyteorder()
mask = wht == 0
mask.astype(np.int32) # SEP wants 0 and 1

err = np.where(wht==0, 0, 1/np.sqrt(wht))
rms = np.nanmedian(err[err>0])

# Visualize it
if PLOT > 0:
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    axes[0].imshow(img, norm=LogNorm(rms, 10*rms), cmap='Greys')
    axes[1].imshow(wht, cmap='Greys')
    axes[2].imshow(mask)

    plt.savefig(f'output/{BAND}_input_total.pdf')

    plt.close('all')

# Estimate background
bkg = sep.Background(img, mask=mask, maskthresh=1, bw=64, bh=64, fw=3, fh=3, fthresh=0.)
img -= bkg.globalback
print(f'Subtracting global background {bkg.globalback:2.2f}')

# Load in truth
truth = Table.read(os.path.join(WORKING_DIR, 'intermediate/CEERS_SDR3_SAM_input_small.fits'))
tcoords = SkyCoord(truth['ra']*u.deg, truth['dec']*u.deg)
in_foot = [wcs.footprint_contains(c) for c in tcoords]
compare = in_foot & (truth[f'NIRCam_F150W'] < 28.9)
tcoords = tcoords[compare]
truth = truth[compare]

objects, segmap, recall, precision, fscore, truth, mtruth = process_total(maxml_theta, img, err, mask, truth)

# Plot diagnostics
if PLOT > 0:
    fig, axes = plt.subplots(nrows=nparam+1, figsize=(10, 3*(nparam+2)), sharex=True)
    for i in range(nparam+1):
        ax = axes[i]
        ax.plot(all_samples[:, :,  i], "k", alpha=0.3)
        ax.set_xlim(0, len(all_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    fig.savefig('output/chain_diagnostic.pdf')

    fsamples = all_samples.reshape(-1, all_samples.shape[-1])
    fsamples[fsamples<0] = 0
    fsamples[np.isnan(fsamples)] = 0
    fig = corner.corner(fsamples, labels=labels)
    fig.savefig('output/corner_diagnostic.pdf')

    # tell me things
    mfig, maxes = plt.subplots(ncols=3, figsize=(20, 5), sharey=True)
    maxes[0].axhline(0, ls='dotted', c='grey')
    maxes[0].axhline(1, ls='dotted', c='grey')
    maxes[1].axhline(0, ls='dotted', c='grey')
    maxes[1].axhline(1, ls='dotted', c='grey')
    maxes[2].axhline(0, ls='dotted', c='grey')
    maxes[2].axhline(1, ls='dotted', c='grey')


    color = 'royalblue'
    # plot an ellipse for each object
    fig, axes = plt.subplots(ncols=2, figsize=(40, 20))
    im = axes[0].imshow(img, interpolation='nearest', norm=LogNorm(rms, 100*rms), cmap='Greys')
    im = axes[1].imshow(img, interpolation='nearest', norm=Normalize(-5*rms, 5*rms), cmap='RdGy')
    for ax in axes:
        for k in range(len(objects)):
            e = Ellipse(xy=(objects['x'][k], objects['y'][k]),
                        width=6*objects['a'][k],
                        height=6*objects['b'][k],
                        angle=objects['theta'][k] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

    x, y = tcoords.to_pixel(wcs)
    for ax in axes:
        ax.scatter(x, y, c='g', s=0.1)

    fig.savefig(f'output/{plabel}.pdf')

    # TODO -- try a spatially binned Recall...

    # plot completeness vs. z, mag
    inbins = np.arange(0, 8.5, 1)
    sel, bins = np.histogram(mtruth['redshift'], inbins)
    struth, bins = np.histogram(truth['redshift'], inbins)
    bincenter = bins[:-1] + np.diff(bins)
    maxes[0].plot(bincenter, sel/struth, label=label, c=color)
    maxes[0].fill_between(bincenter, sel/struth - np.sqrt(sel)/struth, sel/struth + np.sqrt(sel)/struth, color=color, alpha=0.2)
    maxes[0].set(xlabel='Redshift $z$', ylabel='Fraction matched', ylim=(-0.1, 1.1))

    inbins = np.arange(22, 30, 1)
    sel, bins = np.histogram(mtruth[f'NIRCam_{BAND}'], inbins)
    struth, bins = np.histogram(truth[f'NIRCam_{BAND}'], inbins)
    bincenter = bins[:-1] + np.diff(bins)
    maxes[1].plot(bincenter, sel/struth, c=color)
    maxes[1].fill_between(bincenter, sel/struth - np.sqrt(sel)/struth, sel/struth + np.sqrt(sel)/struth, color=color, alpha=0.2)
    maxes[1].set(xlabel=f'Mag {BAND}', ylim=(-0.1, 1.1))

    inbins = np.linspace(0, 4000, 10)
    sel, bins = np.histogram(mtruth[f'NIRCam_{BAND}'] / mtruth['angular_size']**2, inbins)
    struth, bins = np.histogram(truth[f'NIRCam_{BAND}'] / truth['angular_size']**2, inbins)
    bincenter = bins[:-1] + np.diff(bins)
    maxes[2].plot(bincenter, sel/struth, c=color)
    maxes[2].fill_between(bincenter, sel/struth - np.sqrt(sel)/struth, sel/struth + np.sqrt(sel)/struth, color=color, alpha=0.2)
    maxes[2].set(xlabel=f'Mag {BAND} / arcsec'+r'$^2$', ylim=(-0.1, 1.1))
    maxes[2].invert_xaxis()


    maxes[0].legend()
    plt.tight_layout()
    mfig.savefig(f'output/recovery.pdf')


