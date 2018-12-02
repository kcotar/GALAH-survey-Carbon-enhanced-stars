import numpy as np
import time, datetime
from astropy.table import Table, join

from spectra_collection_functions import *
from helper_functions import *

TEFF_STEP = 150.
LOGG_STEP = 0.2
FEH_STEP = 0.1

print 'Reading GALAH parameters'
galah_data_dir = '/data4/cotar/'
date_string = '20180327'
galah_param = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')
cannon_param = Table.read(galah_data_dir + 'sobject_iraf_iDR2_180325_cannon.fits')['sobject_id', 'Teff_cannon', 'Fe_H_cannon','Logg_cannon','flag_cannon']
galah_param = join(galah_param, cannon_param, join_type='left', keys='sobject_id')

print 'Reading resampled GALAH spectra'
spectra_file_csv = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl'
spectral_data = read_pkl_spectra(galah_data_dir + spectra_file_csv)
spectal_data_size = np.shape(spectral_data)

# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_step = csv_param.get_wvl_step()
wvl_values = csv_param.get_wvl_values()

# euclidean sum indices
eucl_sum_start = np.argmin(np.abs(wvl_values - 4750.))
eucl_sum_end = np.argmin(np.abs(wvl_values - 4890.))
print 'Euclidean sum:', eucl_sum_start, eucl_sum_end

# select data-subset that can be used for median spectra computation
snr_limit = 15.
snr_label = 'snr_c'+ccd+'_iraf'
idx_snr_ok = galah_param[snr_label] > snr_limit
idx_snr_ok = np.logical_and(idx_snr_ok, galah_param['sobject_id'] > 0)  # main run limit
idx_snr_ok = np.logical_and(idx_snr_ok, galah_param['red_flag'] & np.int32(ccd)**2 != np.int32(ccd)**2)  # must have valid wvl solution in blue band
idx_snr_ok = np.logical_and(idx_snr_ok, galah_param['red_flag'] & 64 != 64)  # remove twilight flats

n_use = 250  # or less if they are filtered by the means of physical parameters

filename_prefix = spectra_file_csv[:-4]
path_out = galah_data_dir+filename_prefix+'_median_{:.0f}_snr_{:.0f}_teff_{:.0f}_logg_{:1.2f}_feh_{:1.2f}.pkl'.format(n_use, snr_limit, TEFF_STEP, LOGG_STEP, FEH_STEP)

spectra_array = np.ndarray((len(galah_param), len(wvl_values)))
spectra_array.fill(np.nan)

time_s = time.time()
for i_id in range(len(galah_param)):
    # get parameters of the observed object
    object_parameters = galah_param[i_id]
    spectra_object = spectral_data[i_id]
    # print 'Working on object ' + str(object_parameters['sobject_id'])
    if i_id % 150 == 0:
        time_now = time.time()
        end_time = (time_now - time_s) / (i_id + 1) * (len(galah_param) - i_id - 1)
        print i_id
        print 'Estimated finish in ' + str(datetime.timedelta(seconds=end_time))

    # first try to find objects in the neighbourhood
    idx_filter = np.logical_and(np.abs(galah_param['logg_guess'] - object_parameters['logg_guess']) < (LOGG_STEP / 2.),
                                np.abs(galah_param['teff_guess'] - object_parameters['teff_guess']) < (TEFF_STEP / 2.))
    idx_filter = np.logical_and(idx_filter,
                                np.abs(galah_param['feh_guess'] - object_parameters['feh_guess']) < (FEH_STEP / 2.))

    idx_filter = np.logical_and(idx_filter, idx_snr_ok)
    n_similar = np.sum(idx_filter)

    # second try to find objects in the neighbourhood - double the parameter thresholds for observed neighborhood
    if n_similar < 5:
        idx_filter = np.logical_and(np.abs(galah_param['logg_guess'] - object_parameters['logg_guess']) < LOGG_STEP,
                                    np.abs(galah_param['teff_guess'] - object_parameters['teff_guess']) < TEFF_STEP)
        idx_filter = np.logical_and(idx_filter,
                                    np.abs(galah_param['feh_guess'] - object_parameters['feh_guess']) < FEH_STEP)
        idx_filter = np.logical_and(idx_filter, idx_snr_ok)
        n_similar = np.sum(idx_filter)

    # print ' Number of similar objects: '+str(n_similar)
    if n_similar < 5:
        # print '  Skipping this object'
        continue

    if n_similar > n_use:
        # refine them based od spectral similarity
        # determine euclidean distance between observed and all other/selected/refined spectra
        # sum euclidean distances outside observed SWAN region as it can have a large impact on the distance estimation
        eucl_data = np.power(spectral_data[idx_filter, eucl_sum_start: eucl_sum_end] - spectra_object[eucl_sum_start: eucl_sum_end], 2)
        eucl_sum = np.nansum(eucl_data, axis=1)
        idx_select = np.argsort(eucl_sum)[:n_use]  # select spectra with the best match

        # final selection of spectra to be used for median stacking
        idx_median_use = (np.where(idx_filter)[0])[idx_select]

        # calculate median spectra of comparison spectra
        compare_spectra = spectral_data[idx_median_use]
        spectra_median = np.nanmedian(compare_spectra, axis=0)
    else:
        # faster variant for object with less than n_use neighbours in parameter space
        spectra_median = np.nanmedian(spectral_data[idx_filter], axis=0)
    # store created median spectra
    spectra_array[i_id, :] = spectra_median

# store final array with all median spectra together
save_pkl_spectra(spectra_array, path_out)

