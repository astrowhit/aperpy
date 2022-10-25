import sys, os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.ioff()

# DIR_CONFIG = '/Volumes/External1/Projects/Current/UNCOVER/scripts/'
DIR_CONFIG = sys.argv[1]
sys.path.insert(0, DIR_CONFIG)
from config import FILTERS, DET_TYPE, DIR_CATALOGS

DET_NICKNAME = sys.argv[2]
KERNEL = sys.argv[3]
STR_APER = sys.argv[4]

CATALOG = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/{DET_NICKNAME}_K{KERNEL}_SCIREADY_{STR_APER}_CATALOG.fits')
DIR_FIGURES = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}/{KERNEL}/')

cat = Table.read(CATALOG)


# Plot field by denfity, flux, error, S/N
cmap = 'viridis'
flim = 10**(-0.4 * (28 - 25))
fmax = 10**(-0.4 * (22 - 25))
opt_d = dict(norm=LogNorm(vmin=1, vmax=100), cmap='Greys', ec='none', reduce_C_function=np.sum)
opt_f = dict(norm=LogNorm(vmin=flim, vmax=fmax), cmap='plasma', ec='none', reduce_C_function=np.mean)
opt_e = dict(norm=LogNorm(vmin=flim, vmax=fmax*0.1), cmap='plasma', ec='none', reduce_C_function=np.mean)
opt_fe = dict(norm=LogNorm(vmin=1, vmax=100), cmap='plasma', ec='none', reduce_C_function=np.mean)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
areas = {}
for filt in FILTERS:
    filt = filt.lower()
    fig, axes = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, figsize=(5*4, 5*1))
    
    axes[0].set_ylabel(CATALOG.split('/')[-1])
    axes[0].set_title(f'{filt}')
    axes[1].set_title(f'f_{filt}')
    axes[2].set_title(f'e_{filt}')
    axes[3].set_title(f'f / e_{filt}')
    ra, dec = cat['ra'], cat['dec']
    flux, ferr = cat[f'f_{filt}'], cat[f'e_{filt}']
    extent = (ra.max(), ra.min(), dec.min(), dec.max())
#         print(extent)
    gridsize = (50, 50)
    cut = (flux / ferr > 0)
    ra, dec, flux, ferr = ra[cut], dec[cut], flux[cut], ferr[cut]
    
    H, xe, ye = np.histogram2d(ra, dec, bins=(50, 50), range=((3.487, 3.687), (-30.5, -30.2)))
    area = np.sum(H>5) * 1.5462919561606767e-05 # ~deg2 per cell at this grid
    areas[filt] = area
    print(filt, area, area*3600.)
    
    im = axes[0].hexbin(ra, dec, extent=extent, gridsize=gridsize, **opt_d)
    cbaxes = inset_axes(axes[0], width="40%", height="3%", loc='upper right') 
    plt.colorbar(im, cax=cbaxes, orientation='horizontal')
    
    im = axes[1].hexbin(ra, dec, C=flux, extent=extent, gridsize=gridsize, **opt_f)
    cbaxes = inset_axes(axes[1], width="40%", height="3%", loc='upper right') 
    plt.colorbar(im, cax=cbaxes, orientation='horizontal')
    
    im = axes[2].hexbin(ra, dec, C=ferr, extent=extent, gridsize=gridsize, **opt_e)
    cbaxes = inset_axes(axes[2], width="40%", height="3%", loc='upper right') 
    plt.colorbar(im, cax=cbaxes, orientation='horizontal')
    
    im = axes[3].hexbin(ra, dec, C=flux/ferr, extent=extent, gridsize=gridsize, **opt_fe)
    cbaxes = inset_axes(axes[3], width="40%", height="3%", loc='upper right') 
    plt.colorbar(im, cax=cbaxes, orientation='horizontal')
        
#     [ax.invert_xaxis() for ax in axes.flatten()]
    axes[0].set_xlim(3.687, 3.487)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(os.path.join(DIR_FIGURES, f'field_{filt}_{STR_APER}.pdf'))

# Number counts

cmap = 'viridis'
flim = 10**(-0.4 * (28 - 25))
fmax = 10**(-0.4 * (22 - 25))
db = 0.5
colors = plt.cm.get_cmap('tab20b', len(FILTERS))

for i, filt in enumerate(FILTERS):
    fig, ax = plt.subplots(figsize=(5*1, 5*1))

    area = areas[filt]
    print(filt, area, areas[filt])
    
    ra, dec = cat['ra'], cat['dec']
    flux, ferr = cat[f'f_{filt}'], cat[f'e_{filt}']
    
    color = colors(i)
    
    cut = (flux / ferr > 0)
    ra, dec, flux, ferr = ra[cut], dec[cut], flux[cut], ferr[cut]
    mag = np.where(flux > 0, 25 - 2.5*np.log10(flux), np.nan)
    
    bins = np.arange(20, 29.5, db)
    
    bin_count, bin_edges = np.histogram(mag, bins=bins) #, weights=weights)
    bin_centers = bins[:-1] + np.ediff1d(bins)[0]
    bin_width = db / 2.
    bin_err = np.sqrt(bin_count)
#         area = 1.
    bin_density = bin_count / area / db
    yerr = bin_err / area / db

    # All else
    ax.errorbar(bin_centers, bin_density, xerr=bin_width, yerr=yerr, 
                c=color, fmt='o', label=f'{filt}') # (N={bin_count.sum()}, A


    ax.legend(ncol=2, loc='lower right')
    ax.semilogy()
    ax.grid()
    ax.set_xlabel('m$_{\\rm AB}$')
    ax.set_ylabel('$N\,{\\rm mag}^{-1}\,{\\rm deg}^{-2}$')
    ax.set(ylim=(1e4, 7e5))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIGURES, f'number_counts_{filt}_{STR_APER}.pdf'))

# Growth of flux error

for filt in FILTERS:
    fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, figsize=(7*2, 5*1))
    
    axes[0].set_title(CATALOG.split('/')[-1])
    axes[0].set_ylabel(f'magerr({filt})')
    axes[1].set_ylabel(f'mag(e_{filt})')
    axes[0].text(0.04, 0.9, filt, transform=axes[0].transAxes, fontsize=15)

    ra, dec = cat['ra'], cat['dec']
    flux, fluxerr = cat[f'f_{filt}'], cat[f'e_{filt}']
#         flux, fluxerr = cat[f'{filt}_FLUX_APER0_7_TOTAL'], cat[f'{filt}_FLUXERR_APER0_7_FULL']
    mag = np.where(flux > 0, 25. - 2.5*np.log10(flux), np.nan)
    magerr = 2.5 / np.log(10) / (flux / fluxerr)

    extent=(20, 33, 0, 1)
    gridsize=(30, 20)
    axes[0].hexbin(mag, magerr, extent=extent, gridsize=gridsize, cmap='Greys', norm=LogNorm())
    extent=(20, 33, 20, 33)
    axes[1].hexbin(mag, 25-2.5*np.log10(fluxerr), extent=extent, gridsize=gridsize, cmap='Greys', norm=LogNorm())
        
    axes[0].set(xlim=(20, 33), ylim=(0, 1))
    axes[1].set(ylim=(20, 33))
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIGURES, f'phot_unc_{filt}_{STR_APER}.pdf'))

# Crossmatch with some other catalog... dmag, dcolor, dphotoz...etc.