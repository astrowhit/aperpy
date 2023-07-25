### Welcome to the documentation for Aperpy!
(Another aPERture Photometry code in pYthon)

Aperpy is a front-to-back aperture photometry pipeline written in python. It uses pythonic Source Extractor (SEP, [Barbary et al. 2016](https://ui.adsabs.harvard.edu/abs/2016JOSS....1...58B/abstract)) to detect sources in a noise-equalized (i.e. inverse variance weighted) detection image, resamples images to a common pixel scale, builds emperical PSFs from point sources and corresponding matching kernels, measure photometry in a number of apertures including Kron-like 'AUTO' ellipses, auto-selects the most suitable aperture for a given source size, flags stars and artifacts, prepares ready-to-use output catalogs, and even run EAZY ([Brammer et al. 2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...686.1503B/abstract)). A large number of diagnostic figures are provided as output. Methodology was adapted from [Whitaker et al. 2011](https://ui.adsabs.harvard.edu/abs/2011ApJ...735...86W/abstract) and [Labbe et al. 2003](https://ui.adsabs.harvard.edu/abs/2003AJ....125.1107L/abstract). Currently it is designed for use with JWST and HST data, although extensions are possible. 

If you use the software, please cite [Weaver et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230102671W/abstract):

> @ARTICLE{Weaver2023,
       author = {{Weaver}, John R. and {Cutler}, Sam E. and {Pan}, Richard and {Whitaker}, Katherine E. and {Labbe}, Ivo and {Price}, Sedona H. and {Bezanson}, Rachel and {Brammer}, Gabriel and {Marchesini}, Danilo and {Leja}, Joel and {Wang}, Bingjie and {Furtak}, Lukas J. and {Zitrin}, Adi and {Atek}, Hakim and {Coe}, Dan and {Dayal}, Pratika and {van Dokkum}, Pieter and {Feldmann}, Robert and {Forster Schreiber}, Natascha and {Franx}, Marijn and {Fujimoto}, Seiji and {Fudamoto}, Yoshinobu and {Glazebrook}, Karl and {de Graaff}, Anna and {Greene}, Jenny E. and {Juneau}, Stephanie and {Kassin}, Susan and {Kriek}, Mariska and {Khullar}, Gourav and {Maseda}, Michael and {Mowla}, Lamiya A. and {Muzzin}, Adam and {Nanayakkara}, Themiya and {Nelson}, Erica J. and {Oesch}, Pascal A. and {Pacifici}, Camilla and {Papovich}, Casey and {Setton}, David and {Shapley}, Alice E. and {Smit}, Renske and {Stefanon}, Mauro and {Taylor}, Edward N. and {Weibel}, Andrea and {Williams}, Christina C.},
>        title = "{The UNCOVER Survey: A first-look HST+JWST catalog of 50,000 galaxies near Abell 2744 and beyond}",
>      journal = {arXiv e-prints},
>     keywords = {Astrophysics - Astrophysics of Galaxies},
>         year = 2023,
>        month = jan,
>          eid = {arXiv:2301.02671},
>        pages = {arXiv:2301.02671},
>          doi = {10.48550/arXiv.2301.02671},
>archivePrefix = {arXiv},
>       eprint = {2301.02671},
> primaryClass = {astro-ph.GA},
>       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230102671W},
>      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
>}