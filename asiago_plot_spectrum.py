import matplotlib as mpl
from astropy.io import fits
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from helper_functions import *
from os import chdir

# set matplotlib properties for better looking plot
mpl.rcParams['font.size'] = 11
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
font = FontProperties().copy()
font.set_style('italic')
font.set_weight('bold')

# define Galah spectra directory
spectra_dir_3 = '/data4/cotar/dr5.3/'

# read Asiago spectrum
s_file = 'EC60089_1D_vh_norm.0001.fits'

data = fits.open(s_file)
header = data[0].header

flx = data[0].data
wvl = header.get('CRVAL1') + np.float64(range(0, len(flx))) * header.get('CDELT1')
# shift spectrum to rest frame
rv_shift = 126.6
wvl *= (1 - rv_shift / 299792.458)

# read equivalent Galah spectrum
f2, w2 = get_spectra_dr52(str(150409005101291), bands=[1, 2, 3, 4], root=spectra_dir_3,
                          individual=False, extension=4, read_sigma=False)
# combine Galah arms, nan inserted to break arms when plotting the data
flx_g = np.hstack((np.nan, f2[0], np.nan, f2[1], np.nan, f2[2], np.nan, f2[3], np.nan))
wvl_g = np.hstack((np.nan, w2[0], np.nan, w2[1], np.nan, w2[2], np.nan, w2[3], np.nan))

fig, ax = plt.subplots(6, 1, figsize=(4,9.5))
plt.subplots_adjust(bottom=0.055, right=0.95, top=0.99, left=0.15, hspace=0.32)

for i_ax in range(len(ax)):
    ax[i_ax].plot(wvl_g, flx_g, lw=0.5, c='C3')

# mark different carbon features in the observed spectra
ax[0].axvline(4312.5, ls='--', alpha=0.6, color='C2')
ax[0].axvline(4323.5, ls='--', alpha=0.6, color='C2')
ax[0].axvline(4338, ls='--', alpha=0.6, color='C2')
ax[0].plot(wvl, flx, lw=0.5, c='black')
ax[0].text(4314, 0.3, 'CH', color='C2', fontproperties=font, alpha=0.9)
ax[0].set(xlim=(4280,4350), ylim=(0.2, 1.2), ylabel=' ')

ax[1].axvline(4715, ls='--', alpha=0.6, color='C2')
ax[1].axvline(4737, ls='--', alpha=0.6, color='C2')
ax[1].plot(wvl, flx, lw=0.5, c='black')
ax[1].text(4716, 0.65, r'C$_{2}$', color='C2', fontproperties=font, alpha=0.9)
ax[1].set(xlim=(4710,4750), ylim=(0.6, 1.2), ylabel=' ')

ax[2].axvline(5165, ls='--', alpha=0.6, color='C2')
ax[2].plot(wvl, flx, lw=0.5, c='black')
ax[2].text(5162, 0.65, r'C$_{2}$', color='C2', fontproperties=font, alpha=0.9)
ax[2].set(xlim=(5150,5190), ylim=(0.6, 1.2), ylabel='  ')

ax[3].axvline(5658.8, ls='--', alpha=0.6, color='C2')
ax[3].axvline(5663, ls='--', alpha=0.6, color='C2')
ax[3].plot(wvl, flx, lw=0.5, c='black')
ax[3].text(5660, 0.65, r'C$_{2}$', color='C2', fontproperties=font, alpha=0.9)
ax[3].set(xlim=(5640,5680), ylim=(0.6, 1.2), ylabel=' Normalized flux ')

ax[4].axvline(6162.30, ls='--', alpha=0.6, color='C2')
ax[4].axvline(6191.5, ls='--', alpha=0.6, color='C2')
ax[4].plot(wvl, flx, lw=0.5, c='black')
ax[4].text(6164, 0.65, r'C$_{2}$', color='C2', fontproperties=font, alpha=0.9)
ax[4].set(xlim=(6154,6195), ylim=(0.6, 1.2), ylabel=' ')

ax[5].axvline(6496.8, ls='--', alpha=0.6, color='C2')
ax[5].plot(wvl, flx, lw=0.5, c='black')
ax[5].text(6498.5, 0.65, 'Ba II', color='C2', fontproperties=font, alpha=0.9)
ax[5].set(xlim=(6480,6520), ylim=(0.6, 1.2), xlabel=r'Wavelength [$\AA$]', ylabel=' ')

plt.savefig('asiago_cemp2_.pdf', dpi=300)
plt.close()

