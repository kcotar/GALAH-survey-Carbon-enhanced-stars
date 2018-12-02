from astropy.table import Table
from scipy.interpolate import interp1d
from socket import gethostname
import matplotlib
matplotlib.use('Agg')
import time, datetime
import numpy as np
import matplotlib.pyplot as plt
import varconvolve as varcon
from scipy import mgrid
from helper_functions import *
from spectra_collection_functions import read_pkl_spectra, save_pkl_spectra
from astropy.io import fits


def kernel(s):
    """
    Constructs a normalized discrete 1D gaussian kernel
    """
    size_grid = int(s*4)
    x = mgrid[-size_grid:size_grid+1]
    g = np.exp(-(x**2/float(s**2)/2.))
    return g / np.sum(g)


# input data
spectra_dir = '/data4/cotar/dr5.3/'
galah_data_dir = '/data4/cotar/'
data_target_dir = '/data4/cotar/'

# Resampling settings
renormalize_data = False
plot_individual = False
export_as_csv = False
read_ext = 4
ext0_rv_shift = True

# possible to reduce resolution of spectra
reduce_resolution = True
R_origin = 28000.
R_target = 19000.

galah_obs_fits = 'sobject_iraf_53_reduced_20180327.fits'
general_data = Table.read(galah_data_dir + galah_obs_fits)
data_date_str = galah_obs_fits.split('.')[0].split('_')[-1]
print data_date_str

suffix = ''

for ccd in list([1, 2, 3, 4]):  # 1-4
    print 'Working on ccd {0}'.format(ccd)
    min_wvl = list([4710, 5640, 6475, 7700])[ccd - 1]
    max_wvl = list([4910, 5880, 6745, 7895])[ccd - 1]
    step_wvl = list([0.04, 0.05, 0.06, 0.07])[ccd - 1]  # more or less original values
    # step_wvl = list([0.02, 0.025, 0.03, 0.035])[ccd - 1]  # subsample spectra
    # sigma_clip_fit = list([4., 3.5, 3., 3.])[ccd - 1]
    sigma_clip_fit = list([3., 2.5, 2.5, 2.5])[ccd - 1]

    target_wvl = min_wvl + np.float64(range(0, np.int32(np.ceil((max_wvl-min_wvl)/step_wvl)))) * step_wvl
    empty_data = np.zeros(shape=(len(target_wvl)), dtype='float16')
    empty_data.fill(np.nan)

    out_txt_file = 'galah_dr53_ccd{:1.0f}_{:4.0f}_{:4.0f}_wvlstep_{:01.3f}_ext{:1.0f}'.format(ccd, min_wvl, max_wvl, step_wvl, read_ext)
    if renormalize_data:
        out_txt_file += '_renorm'
    if read_ext == 0 and not ext0_rv_shift:
        out_txt_file += '_origRV'
    if reduce_resolution:
        out_txt_file += '_R{:.0f}'.format(R_target)
        # load the resolution map for the correct arm
        piv_fits = galah_data_dir+'ccd{:.0f}_piv.fits'.format(ccd)
        res_map_data = fits.open(piv_fits)
        res_map_R = res_map_data[0].data
        res_map_WVL = res_map_data[0].header['CRVAL1'] + res_map_data[0].header['CDELT1']*np.arange(res_map_R.shape[1])
        res_map_data.close()

    out_txt_file += '_' + data_date_str
    if export_as_csv:
        txt = open(data_target_dir+out_txt_file+'.csv', 'w')
    else:
        spectra_array = np.ndarray((len(general_data), len(empty_data)))

    i_r = 0
    time_s = time.time()
    for row in general_data:
        if i_r % 250 == 0:
            time_now = time.time()
            end_time = (time_now-time_s)/(i_r+1)*(len(general_data)-i_r-1)
            print i_r
            print 'Estimated finish in '+str(datetime.timedelta(seconds=end_time))
        object_id = str(row['sobject_id'])

        try:
            spectrum, wavelengths = get_spectra_dr52(object_id, root=spectra_dir, bands=[ccd], read_sigma=False, extension=read_ext)

            # determine range of wavelengths
            min_wvl_s = np.nanmin(wavelengths)
            max_wvl_s = np.nanmax(wavelengths)
            idx_target = np.logical_and(target_wvl >= min_wvl_s, target_wvl <= max_wvl_s)

            if read_ext > 0:
                if renormalize_data:
                    # continuum fit to the spectrum
                    spectra_read = spectrum[0]
                    wvl_read = wavelengths[0]
                    cont_fit = spectra_normalize(wvl_read, spectra_read,
                                                 steps=17, sigma_low=1., sigma_high=1.8, order=11, return_fit=True, sg_filter=True)
                    spectrum[0] = spectra_read/cont_fit

                    if plot_individual:
                        plt.plot(wvl_read, spectra_read, color='black', linewidth=0.5)
                        plt.plot(wvl_read, cont_fit, color='red', linewidth=1.0)
                        plt.plot(wvl_read, cont_fit2, color='green', linewidth=1.0)
                        plt.plot(wvl_read, spectra_read/cont_fit-0.5, color='blue', linewidth=0.5)
                        plt.xlim((min_wvl, max_wvl))
                        plt.ylim((0.4, 1.2))
                        plt.savefig(object_id+'_norm.png', dpi=250)
                        plt.close()

                # OPTION1: Resample part of selected spectra using interp1d
                # any higher order of interpolation than linear is just too slow to process all data in reasonable time
                func = interp1d(wavelengths[0], spectrum[0], assume_sorted=True, kind='linear')
                new_flux = func(target_wvl[idx_target])
                nex_flux_out = np.ndarray(len(target_wvl))
                nex_flux_out.fill(np.nan)
                nex_flux_out[idx_target] = new_flux
            else:
                # OPTION2: Resample and rv shift ext0
                spectrum[0] = spectra_normalize(wavelengths[0], spectrum[0], steps=5, sigma_low=1.5, sigma_high=2.5, order=1,
                                                n_min_perc=5., return_fit=False, func='poly')
                spectrum[0] = spectra_normalize(wavelengths[0], spectrum[0], steps=20, sigma_low=1.8, sigma_high=3., order=17,
                                                n_min_perc=7., return_fit=False, func='poly')
                # apply computed rv shift to the spectrum
                if ext0_rv_shift:
                    rv_shift = general_data[general_data['sobject_id'] == row['sobject_id']]['rv_guess_shift']
                    wavelengths[0] *= (1 - rv_shift / 299792.458)

                nex_flux_out = spectra_resample(spectrum[0], wavelengths[0], target_wvl, k=1)

            if reduce_resolution:
                # TODO: get correct wvl information from the ext0 and not ext4
                pivot_num = int(str(row['sobject_id'])[-3:])
                pivot_R_vals = res_map_R[pivot_num-1, :]
                pivot_R_vals_interp = np.interp(target_wvl, res_map_WVL, pivot_R_vals)

                s_orig = target_wvl / (2.3548 * pivot_R_vals_interp)
                s_targ = target_wvl / (2.3548 * R_target)
                kernel_widths = np.sqrt(s_targ**2 - s_orig**2)
                nex_flux_out = varcon.varconvolve(target_wvl, nex_flux_out, kernel, kernel_widths)

            if export_as_csv:
                txt.write(','.join(['{:.4f}'.format(f) for f in nex_flux_out]) + '\n')
            else:
                spectra_array[i_r, :] = nex_flux_out
        except:
            print ' Something went wrong with '+object_id+'. Possible nan values in spectra or interpolation problem.'
            if export_as_csv:
                 txt.write(','.join([str(f) for f in empty_data]) + '\n')
            else:
                spectra_array[i_r, :] = empty_data
        i_r += 1
        # os.chdir('..')
    if export_as_csv:
         txt.close()
    else:
        save_pkl_spectra(spectra_array, data_target_dir+out_txt_file+suffix+'.pkl')
