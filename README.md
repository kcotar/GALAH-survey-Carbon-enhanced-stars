# The GALAH survey: a catalogue of carbon-enhanced stars and CEMP candidates 

The paper was published at [MNRAS](https://academic.oup.com/mnras/advance-article-abstract/doi/10.1093/mnras/sty3155/5199221).

The code used in the analysis of carbon-enhanced stars and production of the accompanying science paper

### Median spectra creation
This procedure consists of two steps. First, the collected spectra have to be resampled to the same wavelength grid using the `resample_spectra.py` script. After that script called `median_spectra.py` is used to compute the median reference spectrum for every observed spectrum based on its physical stellar properties.

### Discovery of potential objects - supervised
The interesting objects were detected by fitting a log-normal distribution to the division between observed and median spectrum in `determine_swan_band_strength.py`. Results of the procedure are analysed and visualized in multiple graphs in the `determine_swan_band_strength_analyse_results.py` script.

### Discovery of potential objects - unsupervised

### Asiago spectrum
The plots were created using the `asiago_plot_spectrum.py` script.

### Orbital analysis
In order to analyse orbits of the detected stars, especially CEMP candidates a galpy library was used in the `orbits_analysis.py` script.

### Auxiliary data
Folder `Data` contains resulting t-SNE projection with raw projected data. Match with other similar research and their finding is given in folder `Reference_lists`.
