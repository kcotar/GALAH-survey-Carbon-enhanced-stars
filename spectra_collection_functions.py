import numpy as np

from scipy.stats import loggamma
from scipy.optimize import curve_fit
from sklearn.externals import joblib


# --------------------------------------------------------
# ---------------- Global variables ----------------------
# --------------------------------------------------------
SWAN_C2_CENTER = 4737.
SWAN_C2_FIT_MIN = 4728.
SWAN_C2_FIT_MAX = 4742.
SWAN_C2_INTEG_MIN = 4730.
SWAN_C2_INTEG_MAX = 4738.

# --------------------------------------------------------
# ---------------- Save and read collection dumps --------
# --------------------------------------------------------
def read_pkl_spectra(file_path, read_rows=None, read_cols=None):
    data_read = joblib.load(file_path)
    if read_cols is not None:
        data_read = data_read[:, read_cols]
    if read_rows is not None:
        data_read = data_read[read_rows, :]
    return data_read


def save_pkl_spectra(data, file_path):
    joblib.dump(data, file_path)


# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
def integrate_swan_spectra(spectra_data, wvl_data, offset=1.):
    # integrate part of the spectrum representing SWAN bands
    idx_spectra_integrate = np.logical_and(np.logical_and(wvl_data > SWAN_C2_INTEG_MIN,
                                                          wvl_data < SWAN_C2_INTEG_MAX),
                                           np.isfinite(spectra_data))
    integral = np.trapz(offset - spectra_data[idx_spectra_integrate],  # values corrected for the continuum/offset level
                        wvl_data[idx_spectra_integrate])
    return integral


def swan_curve_fit(spectra_data, wvl_data, fit_steps=3, fit_std_outlier=2.):
    # try to fit log norm function to the spectra_div as it has more distinct peak around the C2 swan band head
    idx_fit_range = np.logical_and(np.logical_and(wvl_data >= SWAN_C2_FIT_MIN,
                                                  wvl_data <= SWAN_C2_FIT_MAX),
                                   np.isfinite(spectra_data))
    final_fit = [0.5, 0.2, 1., SWAN_C2_CENTER]  # amp, sig, offset, wvl
    try:
        for i_f_s in range(fit_steps):
            final_fit, cov = curve_fit(swan_curve_model, wvl_data[idx_fit_range], spectra_data[idx_fit_range],
                                       p0=final_fit,  # initial fit guess
                                       bounds=([-np.inf, -np.inf, 0.5, SWAN_C2_CENTER - 2.],  # lower fit bounds
                                               [np.inf, np.inf, 1.5, SWAN_C2_CENTER + 2.]))  # upper fit bounds
            spectra_div_fitted = swan_curve_model(wvl_data, final_fit[0], final_fit[1], final_fit[2], final_fit[3])
            # remove outlying points
            spec_diff = spectra_div_fitted - spectra_data
            spec_diff_std = np.nanstd(spec_diff[idx_fit_range])
            idx_fit_range = np.logical_and(idx_fit_range, np.abs(spec_diff) < fit_std_outlier*spec_diff_std)
        return final_fit
    except RuntimeError:
        print ' Fitting problem - curve'
        return np.repeat(np.nan, len(final_fit))


def swan_lin_fit(spectra_data, wvl_data, fit_steps=2, fit_std_outlier=2.):
    # try to fit log norm function to the spectra_div as it has more distinct peak around the C2 swan band head
    mean_wvl = np.nanmean(wvl_data)
    idx_fit_range = np.isfinite(spectra_data)
    final_fit = [0., 1., mean_wvl]  # amp, sig, offset, wvl
    try:
        for i_f_s in range(fit_steps):
            final_fit, cov = curve_fit(swan_lin_model, wvl_data[idx_fit_range], spectra_data[idx_fit_range],
                                       p0=final_fit,  # initial fit guess
                                       bounds=([-np.inf, -2, mean_wvl - 0.1],  # lower fit bounds
                                               [np.inf, 2, mean_wvl + 0.1]))  # upper fit bounds
            spectra_dif_fitted = swan_lin_model(wvl_data, final_fit[0], final_fit[1], final_fit[2])
            # remove outlying points
            spec_diff = spectra_dif_fitted - spectra_data
            spec_diff_std = np.nanstd(spec_diff[idx_fit_range])
            idx_fit_range = np.logical_and(idx_fit_range, np.abs(spec_diff) < fit_std_outlier * spec_diff_std)
        return final_fit
    except RuntimeError:
        print ' Fitting problem - linear'
        return np.repeat(np.nan, len(final_fit))


def swan_curve_model(x, amp, sig, offset, wvl):
    # shift wvl offset
    # x_zero = x - wvl
    # func = offset - amp / (sig * x_zero * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (np.log(x_zero) / sig) ** 2)
    # return func
    # return offset - skewnorm.pdf(x, sig, wvl, amp)
    # return offset - lognorm.pdf(x, sig, wvl, amp)
    return offset - loggamma.pdf(x, sig, wvl, amp)


def swan_lin_model(x, amp, offset, wvl):
    return amp * (x - wvl) + offset


class CollectionParameters:
    def __init__(self, filename):
        self.filename_full = filename
        self.filename_split = '.'.join(filename.split('.')[:-1]).split('_')

    def __get_str_pos__(self, search_str):
        try:
            idx = self.filename_split.index(search_str)
        except ValueError:
            idx = None
        return idx

    def __get_value__(self, search_str):
        idx_value = self.__get_str_pos__(search_str)
        if idx_value is not None:
            value_str = self.filename_split[idx_value + 1]
            try:
                return int(value_str)
            except ValueError:
                return float(value_str)
        else:
            return None

    def get_ccd(self):
        return self.filename_split[2][3]  # the fourth character is the number of ccd

    def get_wvl_range(self):
        wvl_start = float(self.filename_split[3])
        wvl_end = float(self.filename_split[4])
        return wvl_start, wvl_end

    def get_wvl_values(self):
        min, max = self.get_wvl_range()
        return np.arange(min, max, self.get_wvl_step())

    def get_interp_method(self):
        return self.filename_split[5]

    def get_wvl_step(self):
        return self.__get_value__('wvlstep')

    def get_snr_limit(self):
        return self.__get_value__('snr')

    def get_teff_step(self):
        return self.__get_value__('teff')

    def get_logg_step(self):
        return self.__get_value__('logg')

    def get_feh_step(self):
        return self.__get_value__('feh')

