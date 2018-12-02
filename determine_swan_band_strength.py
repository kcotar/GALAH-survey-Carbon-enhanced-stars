import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 14

from os import chdir
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, join
from scipy.signal import savgol_filter, argrelextrema, decimate, bspline
from scipy.stats import skew, kurtosis

from helper_functions import move_to_dir
from spectra_collection_functions import *

from multiprocessing import Pool
from joblib import Parallel, delayed
from functools import partial
from socket import gethostname
# PC hostname
pc_name = gethostname()


# --------------------------------------------------------
# ---------------- Read the data -------------------------
# --------------------------------------------------------
print 'Reading GALAH parameters'
date_string = '20180327'
snr_limit = 5.
remove_spikes = False

n_multi = 8  # number of parallel threads - 9 is a limit for ram amount on olimp nodes
galah_data_dir = '/data4/cotar/'
out_dir = '/data4/cotar/'

general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')
cannon_data = Table.read(galah_data_dir + 'sobject_iraf_iDR2_180325_cannon.fits')

general_data = join(general_data, cannon_data['sobject_id', 'Fe_H_cannon'], keys='sobject_id', join_type='left')

# determiine collection of original resampled spectra and median spectra
spectra_file_csv = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl'
median_spectra_file_csv = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_median_250_snr_15_teff_150_logg_0.20_feh_0.10_Guess-params.pkl'

# parse interpolation and averaging settings from filename
csv_param = CollectionParameters(median_spectra_file_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_step = csv_param.get_wvl_step()
wvl_values = csv_param.get_wvl_values()

# determine wvls that will be read from the spectra
idx_wvl_read = np.where(np.logical_and(wvl_values >= 4715,
                                       wvl_values <= 4760))[0]
wvl_values = wvl_values[idx_wvl_read]

# read limited number of columns instead of full spectral dataset
print 'Reading resampled/interpolated GALAH spectra'
spectral_data = read_pkl_spectra(galah_data_dir + spectra_file_csv, read_cols=idx_wvl_read)
print 'Reading merged median GALAH spectra'
spectral_median_data = read_pkl_spectra(galah_data_dir + median_spectra_file_csv, read_cols=idx_wvl_read)

# select initial data by FeH and SNR parameters
snr_label = 'snr_c'+ccd+'_iraf'
idx_object_ok = general_data[snr_label] >= snr_limit  # snr limit
idx_object_ok = np.logical_and(idx_object_ok, general_data['sobject_id'] > 0)  # can filter by date even later
# idx_object_ok = np.logical_and(idx_object_ok, general_data['sobject_id'] > 140301000000000)  # main run limit
idx_object_ok = np.logical_and(idx_object_ok, np.bitwise_and(general_data['red_flag'], 64) == 0)  # remove twilight flats

idx_to_process = np.isfinite(general_data['feh_guess'])

# --------------------------------------------------------
# ---------------- Determine objects to be observed ------
# --------------------------------------------------------

# determine object sobject_id numbers
idx_init = np.logical_and(idx_object_ok, idx_to_process)
sobject_ids = general_data[idx_init]['sobject_id']

# used in CEMP article
# sobject_ids = np.array([150902002901051,150603001801056,150412003601009,160515003401143,170515005101173,161217005601288,170531004301094])

move_to_dir(out_dir+'Swan_band_strength_all-spectra_guess-median_20180327__')

# binary flag describing processing step where something went wrong:
# 1000 or 8 = no reference spectra for this object
# 0100 or 4 = bad fit to reference data
# 0010 or 2 = strange discontinuities or jumps found in spectra
# 0001 or 1 = problem in fitting curve to the spectra division
results = Table(names=('sobject_id', 'swan_integ', 'swan_fit_integ', 'amp', 'sig', 'offset', 'wvl', 'amp_lin', 'offset_lin', 'flag'),
                dtype=('int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32'))
n_cols_out = len(results.columns)

# --------------------------------------------------------
# ---------------- Main analysis -------------------------
# --------------------------------------------------------
print 'Number of spectra that will be evaluated:', len(sobject_ids)


def process_selected_id(s_id):
    print 'Working on object '+str(s_id)
    # define flag parameter that will describe processing problem(s) in resulting table
    proc_flag = 0

    # get parameters of the observed object
    idx_object = np.where(general_data['sobject_id'] == s_id)[0]
    object_parameters = general_data[idx_object]

    # get both spectra of the object and it's reduced reference median comparison spectra
    spectra_object = spectral_data[idx_object, :][0]
    spectra_median = spectral_median_data[idx_object, :][0]

    # check validity of reference spectra
    if not np.isfinite(spectra_median).any():
        proc_flag += 0b1000
        results.add_row(np.hstack([s_id, np.repeat(np.nan, n_cols_out-2), proc_flag]))
        txt_out = open(results_csv_out, 'a')
        txt_out.write(','.join([str(v) for v in [s_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, proc_flag]])+'\n')
        txt_out.close()
        print ' Reference spectra is not defined'
        return False

    # compute spectra difference and division
    # spectra_dif = spectra_object - spectra_median
    spectra_dif = -np.log(spectra_median) - (-np.log(spectra_object))
    spectra_div = spectra_object / spectra_median

    # find strange things in spectra comparision:
    # - bad fit to reference spectra
    # - large continuum difference
    # - emission like ccd spikes
    # - discontinuities in observed spectra

    dif_hist_min = -0.6
    dif_hist_max = 0.6
    dif_hist_step = 0.05
    # determine fit accuracy
    spectra_dif_median = np.nanmedian(spectra_dif)
    spectra_dif_mean = np.nanmean(spectra_dif)
    spectra_dif_std = np.nanstd(spectra_dif)
    spectra_dif_kurt = kurtosis(spectra_dif)
    # print spectra_dif_hist
    print spectra_dif_median, spectra_dif_mean, spectra_dif_std
    print spectra_dif_kurt, skew(spectra_dif)
    if spectra_dif_kurt < -0.1:
        print ' Problematic spectra comparison'
        proc_flag += 0b0100

    # find strange discontinuities and jumps in spectra

    # find possible fake ccd emission spikes and remove (eg. to nan) those values from further analysis
    if remove_spikes:
        idx_spikes = np.logical_and(spectra_dif > 0.25,
                                    spectra_object > 1.1)
        if np.sum(idx_spikes) > 0:
            print ' Removing emission spikes found in spectra'
            spectra_dif[idx_spikes] = np.nan
            spectra_div[idx_spikes] = np.nan

    # # noise filtering using Savitzky-Golay filter approach
    # filter_width = 13
    # filter_order = 2
    # spectra_div_filter = savgol_filter(spectra_div, filter_width, filter_order)

    # fit curve to spectra_dif
    spectra_dif_fit = swan_lin_fit(spectra_dif, wvl_values)
    dif_fit_valid = np.isfinite(spectra_dif_fit).all()
    if dif_fit_valid:
        print ' Fitting curve to the spectra difference'
        spectra_dif_fitted = swan_lin_model(wvl_values, spectra_dif_fit[0], spectra_dif_fit[1], spectra_dif_fit[2])
    else:
        proc_flag += 0b0001

    # fit curve to spectra_div
    spectra_div_fit = swan_curve_fit(spectra_div, wvl_values)
    div_fit_valid = np.isfinite(spectra_div_fit).all()
    if div_fit_valid:
        print ' Fitting curve to the spectra division'
        spectra_div_fitted = swan_curve_model(wvl_values, spectra_div_fit[0], spectra_div_fit[1], spectra_div_fit[2], spectra_div_fit[3])
    else:
        proc_flag += 0b0010

    # integrate part of the spectrum representing SWAN bands
    swan_power_dif = integrate_swan_spectra(spectra_dif, wvl_values, offset=spectra_dif_median)
    if div_fit_valid:
        swan_power_fit = integrate_swan_spectra(spectra_div_fitted, wvl_values, offset=spectra_div_fit[2])
    else:
        swan_power_fit = np.nan

    # add to results
    results.add_row([s_id, swan_power_dif, swan_power_fit, spectra_div_fit[0], spectra_div_fit[1], spectra_div_fit[2], spectra_div_fit[3], spectra_dif_fit[0], spectra_dif_fit[1], proc_flag])

    txt_out = open(results_csv_out, 'a')
    txt_out.write(','.join([str(v) for v in [s_id, swan_power_dif, swan_power_fit, spectra_div_fit[0], spectra_div_fit[1], spectra_div_fit[2], spectra_div_fit[3], spectra_dif_fit[0], spectra_dif_fit[1], proc_flag]])+'\n')
    txt_out.close()

    # --------------------------------------------------------
    # ---------------- Plot results --------------------------
    # --------------------------------------------------------
    suffix = ''  #
    print ' Plotting results'
    plt_range = (np.nanmin(wvl_values), np.nanmax(wvl_values))
    # plot all determined comparison spectra and median spectra on top of them
    # add plot master title with all relevant fit anf integ informations
    title_str = 'Integ-dif:{:.3f}    Integ-fit:{:.3f}    Amp:{:.4f}    Sig:{:.4f}    Amp-lin:{:.4f}'\
        .format(swan_power_dif, swan_power_fit, spectra_div_fit[0], spectra_div_fit[1], spectra_dif_fit[0])
    # plot observed spectra and determined comparison spectra
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 5))
    # add individual plots
    ax[0].plot(wvl_values, spectra_object, c='C0')
    ax[0].plot(wvl_values, spectra_median, c='C3')
    ax[0].set(ylim=[0.2, 1.1], xlim=plt_range, ylabel='Flux \n ', title=title_str, yticks=[0.4, 0.6, 0.8, 1.0])
    # plot difference between then
    ax[1].axhline(y=spectra_dif_median, linewidth=2, color='black')
    ax[1].plot(wvl_values, spectra_dif, c='C0')
    if dif_fit_valid:
        ax[1].plot(wvl_values, spectra_dif_fitted, c='C1')
    ax[1].set(ylim=[-0.7, 0.7], xlim=plt_range, ylabel='Absorbance \n difference', yticks=[-0.6, -0.3, 0., 0.3, 0.6])
    # divide and plot both spectra
    ax[2].plot(wvl_values, spectra_div, c='C0')
    if div_fit_valid:
        ax[2].plot(wvl_values, spectra_div_fitted, c='C1')
    ax[2].set(ylim=[0.4, 1.4], xlim=plt_range, ylabel='Division \n ', xlabel=r'Wavelength [$\AA$]', yticks=[0.4, 0.6, 0.8, 1.0, 1.2])
    plt.tight_layout()
    s_date = np.int32(s_id/10e10)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(str(s_date)+'/'+str(s_id)+suffix+'.png', dpi=200)
    plt.close()

    return True


# create all possible output dirs
sobject_dates = np.unique(np.int32(sobject_ids/10e10))
for s_date in sobject_dates:
    move_to_dir(str(s_date))
    chdir('..')

results_csv_out = 'results_swan_line.csv'
txt_out = open(results_csv_out,  'w')
txt_out.write('sobject_id,swan_integ,swan_fit_integ,amp,sig,offset,wvl,amp_lin,offset_lin,flag\n')
txt_out.close()

# # without any multiprocessing - for test purpose only from now on
# for so_id in sobject_ids:
#     process_selected_id(so_id)

# multiprocessing
pool = Pool(processes=n_multi)
process_return = np.array(pool.map(process_selected_id, sobject_ids))
pool.close()
# process_return = np.array(Parallel(n_jobs=n_multi)(delayed(process_selected_id)(so_id) for so_id in sobject_ids))

# save results
results.write('results_swan_line.fits')

