import os, time
from astropy.io import fits
import numpy as np
from scipy.stats import chi
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
from astropy.modeling.models import custom_model, Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.nddata import block_reduce
from skimage.filters import threshold_multiotsu

import sys
PATH_CONFIG = sys.argv[1]
sys.path.insert(0, PATH_CONFIG)

from config import DIR_CATALOGS, DETECTION_GROUPS, DETECTION_IMAGES,\
    DIRWHT_REPLACE, IS_COMPRESSED, WHT_REPLACE, OVERWRITE

# chi-mean, noise-equalized, stack
def simple_chi_mean(bands, outname, science_fnames, weight_fnames, is_compressed=True):
    # sum (S / N) -- but why add sigma linearly
    print(f'Building chi-mean image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            raw_img = fits.getdata(fn_sci)
            raw_img = (raw_img)**2 * fits.getdata(fn_wht)
            img = raw_img
            n = (raw_img != 0).astype(int)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            raw_img = (raw_img)**2 * fits.getdata(fn_wht)
            img += raw_img
            n += (raw_img != 0).astype(int)
            del raw_img

    mu = np.zeros_like(img)
    for ni in np.unique(n):
        mu[n == ni] = chi.stats(df=ni, moments='m')
    img = ( np.sqrt(img) - mu ) / np.sqrt( n - mu**2 )

    chiout = f'{outname}_chimean.fits'
    if is_compressed:
        chiout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(chiout, overwrite=True)

# optimum average, so "noise equalized"
def noise_equalized(bands, outname, science_fnames, weight_fnames, is_compressed=True):

    avgout = f'{outname}_optavg.fits'
    errout = f'{outname}_opterr.fits'
    neqout = f'{outname}_noise-equal.fits'
    if is_compressed:
        avgout += '.gz'
        errout += '.gz'
        neqout += '.gz'

    out_files = [avgout, errout, neqout]
    if not OVERWRITE and all(os.path.exists(path) for path in out_files):
        print('Noise-equalized detection image already exists, I will not remake.\n'
              'Check OVERWRITE param in config if this is not the desired effect.')
        return

    # SUM( X * WHT) / SUM(WHT)
    print(f'Building noise equalized image from {bands}')
    if np.isscalar(bands):
        bands = [bands,]
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(fn_sci)
        print(fn_wht)
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            raw_img = fits.getdata(fn_sci)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img) * wht
            top = raw_img
            bot = wht
            del raw_img
            del wht
        else:
            raw_img = fits.getdata(fn_sci)
            wht = fits.getdata(fn_wht)
            raw_img = (raw_img) * wht
            top += raw_img
            bot += wht
            del raw_img
            del wht

    optavg = np.where(bot==0., 0., top / bot)
    opterr = np.sqrt(np.where(bot<=0, 0., 1. / bot))
    comb = optavg / opterr # signal / noise
    del top
    del bot

    fits.PrimaryHDU(data=optavg.astype(np.float32), header=head).writeto(avgout, overwrite=True)
    del optavg
    fits.PrimaryHDU(data=opterr.astype(np.float32), header=head).writeto(errout, overwrite=True)
    del opterr
    fits.PrimaryHDU(data=comb.astype(np.float32), header=head).writeto(neqout, overwrite=True)
    del comb


def sumstack(bands, outname, science_fnames, weight_fnames, is_compressed=True):
    print(f'Building simple stack image from {bands}')
    for i, band in enumerate(bands):
        fn_sci = science_fnames[band]
        fn_wht = weight_fnames[band]
        print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])
        if i == 0:
            head = fits.getheader(fn_sci, 0)
            wht = fits.getdata(fn_wht)
            raw_img = fits.getdata(fn_sci)
            img = raw_img
            wht = fits.getdata(fn_wht)
            del raw_img
        else:
            raw_img = fits.getdata(fn_sci)
            img += raw_img
            wht += fits.getdata(fn_wht)
            del raw_img

    sciout = f'{outname}_sumstack_sci.fits'
    whtout = f'{outname}_sumstack_wht.fits'
    if is_compressed:
        sciout += '.gz'
        whtout += '.gz'
    fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(sciout, overwrite=True)
    fits.PrimaryHDU(data=wht.astype(np.float32), header=head).writeto(whtout, overwrite=True)

    
@custom_model
def p_dist_chi_mean(x, amplitude=1, N=6):
    mu = np.sqrt(2) * gamma((N + 1) / 2) / \
                               gamma(N / 2)
    g = (x*np.sqrt(N-mu**2) + mu)**2.
    fd = 2 * (x * np.sqrt(N - mu**2.) + mu) * np.sqrt(N-mu**2.)
    return amplitude * fd / (2.**(N/2.) * factorial(N/2. - 1)) * \
            np.exp(- 0.5 * g) * (g)**(N/2. - 1.)


def scaled_chi_mean(bands, outname, science_fnames, weight_fnames, nreg=3, is_compressed=True, save_reg=False):
    plotpath = os.path.join(outpath, 'detection_figures/')
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    
    chiout = f'{outname}_chi-mean.fits'
    nout = f'{outname}_nbands.fits'
    if is_compressed:
        chiout += '.gz'
        nout +='.gz'

    out_files = [chiout, nout]
    if not OVERWRITE and all(os.path.exists(path) for path in out_files):
        print('Chi-mean detection image already exists, I will not remake.\n'
              'Check OVERWRITE param in config if this is not the desired effect.')
        img=fits.getdata(chiout)
        n=fits.getdata(nout)

    else:
        print(f'Building chi-mean image from {bands}')

        # loop through filters
        for i, band in enumerate(bands):
            tstart = time.time()
            fn_sci = science_fnames[band]
            fn_wht = weight_fnames[band]
            print(f'{i+1}/{len(bands)} ', band, fn_sci.split('/')[-1], fn_wht.split('/')[-1])

            head = fits.getheader(fn_sci, 0)
            wht = fits.getdata(fn_wht)

            # Split weight maps into Nreg noise threshold regions
            print('Finding Noise Regions')
            thresholds = threshold_multiotsu(wht[wht!=0.], nreg) 

            fn_save = os.path.join(plotpath,f'{band}_noise_regions.pdf')
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5))
            ax1.hist(wht[wht!=0.])
            for threshold in thresholds:
                ax1.axvline(threshold)

            regions = np.digitize(wht, bins=thresholds) + 1
            regions[wht==0.] = 0
            ax2.imshow(block_reduce(regions,2,np.min), origin='lower')
            ax2.set_title(band)
            ax2.axis('off')
            fig1.savefig(fn_save)

            if save_reg:
                wreg_file = os.path.join(outpath,f'{band}_weight_regions.fits')
                if is_compressed: wreg_file += '.gz'
                print(f'Saving {band} weight regions file')
                fits.writeto(wreg_file,regions.astype(int), header=head, overwrite=True)

            # scale noise regions to have gaussian noise
            print('Scaling noise regions...')
            rms = np.where((wht==0) | np.isnan(wht), np.inf, 1./np.sqrt(wht))
            del wht
            rms_norm_value = np.min(rms[regions!=0])
            rms_norm = rms / rms_norm_value

            sci = fits.getdata(fn_sci)    
            sci_norm = sci / rms_norm
            del sci

            corr_ratio_rms_list = []
            fig1, axes = plt.subplots(nreg, 1)
                
            for i_noise_reg, ax in zip(range(1,nreg+1), axes):
                
                m_reg = regions == i_noise_reg

                mean_est, median_est, std_est = sigma_clipped_stats(sci_norm[m_reg])

                hist1, bins, _ = ax.hist(sci_norm[m_reg], bins=200, range=[-6*std_est, 6*std_est])
                bin_centers = (bins[1:] + bins[:-1]) / 2.

                model_g1 = Gaussian1D(amplitude=np.max(hist1)*0.9, mean=0, stddev=1.0*std_est)

                if i_noise_reg == 1:
                    model_g1.mean.fixed = True
                ins_fit = LevMarLSQFitter()
                m_fit = bin_centers < 0.05*std_est 
                fit_g1 = ins_fit(model_g1, bin_centers[m_fit], hist1[m_fit])

                ax.plot(bin_centers, fit_g1(bin_centers))
                ax = ax.axvline(0, color='red')
                fig1.suptitle(band)
                
                corr_ratio_rms = fit_g1.stddev / rms_norm_value 
                
                print(corr_ratio_rms)
                
                corr_ratio_rms_list.append(corr_ratio_rms)
            
            fig1.savefig(os.path.join(plotpath,f'{band}_gaussian_noise_regions.pdf'))

            del sci_norm
            del rms_norm

            for i_noise_reg in range(1,nreg+1):
                m = regions == i_noise_reg
                rms[m] *= corr_ratio_rms_list[i_noise_reg - 1]


            # build chi-mean image
            raw_img = fits.getdata(fn_sci)
            raw_img = (raw_img / rms)**2 
            if i == 0:
                img = raw_img
                n = (raw_img != 0).astype(int)
            else:
                img += raw_img
                n += (raw_img != 0).astype(int)
            del raw_img

            print(f'{band.upper()} added in {time.time()-tstart:2.2f}s')

        mu = np.zeros_like(img)
        for ni in np.unique(n):
            mu[n == ni] = chi.stats(df=ni, moments='m')
        img = ( np.sqrt(img) - mu ) / np.sqrt( n - mu**2 )

        fits.PrimaryHDU(data=img.astype(np.float32), header=head).writeto(chiout, overwrite=True)
        fits.PrimaryHDU(data=n, header=head).writeto(nout, overwrite=True)

    # check results
    m_stat = ~np.isnan(img)  
    m_stat &= (n == np.max(n))

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for ax in [ax1, ax2]:
        hist1, bin_edges, _ = ax.hist(img[m_stat], bins=200,
                    range=[-3,5], alpha=0.5,
                    color='blue', histtype='bar', label='data')
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.

        # Fit the chi-mean distribution
        model_dist = p_dist_chi_mean(amplitude=10E5, N=np.max(n))
        model_dist.N.fixed = True

        ins_fit = LevMarLSQFitter()
        m_fit = (bin_centers < 0.) & (bin_centers > -2.)

        fit_dist = ins_fit(model_dist, bin_centers[m_fit], hist1[m_fit])

        ax.plot(bin_centers, fit_dist(bin_centers), label='model')
        ax.plot(bin_centers, hist1 - fit_dist(bin_centers), label='data-model')

    ax1.legend()
    ax1.set_xlim(-3, 3)

    ax2.set_ylim(1, 8E5)
    ax2.set_xlim(-3, 9)
    ax2.set_yscale('log')
    fn_save = os.path.join(plotpath,'final_noise_regions.png')
    fig1.savefig(fn_save)




if __name__ == "__main__":
    DET_NICKNAME = sys.argv[2]
    DET_TYPE = DETECTION_GROUPS[DET_NICKNAME.split('_')[0]]['method']
    outpath = os.path.join(DIR_CATALOGS, f'{DET_NICKNAME}_{DET_TYPE}')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    bands = DETECTION_GROUPS[DET_NICKNAME.split('_')[0]]['filters']
    print(bands)
    science_fnames = DETECTION_IMAGES
    weight_fnames = {}
    for band in bands:
        weight_fnames[band] = DETECTION_IMAGES[band].replace(WHT_REPLACE[0], WHT_REPLACE[1]).replace(DIRWHT_REPLACE[0], DIRWHT_REPLACE[1])

    if DET_TYPE == 'noise-equal':
        noise_equalized(bands, os.path.join(outpath, f'{DET_NICKNAME}'),
                        science_fnames= science_fnames,
                        weight_fnames= weight_fnames, 
                        is_compressed=IS_COMPRESSED)
        
    elif DET_TYPE == 'chi-mean':
        if 'n_regions' not in DETECTION_GROUPS[DET_NICKNAME.split('_')[0]].keys():
            nreg = 2
        else:
            nreg = DETECTION_GROUPS[DET_NICKNAME.split('_')[0]]['n_regions']
        scaled_chi_mean(bands, os.path.join(outpath, f'{DET_NICKNAME}'),
                        science_fnames= science_fnames,
                        weight_fnames= weight_fnames,
                        nreg= nreg, is_compressed=IS_COMPRESSED)

    else:
        sys.exit('Other detecton choices are deprecated! Edit code at your own risk...')