import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from time import time
from astropy.table import Table, join

plt.rcParams['font.size'] = 15

data_dir = '/data4/cotar/'

# raed Gaia, Galah data and results of the analysis
print 'Read'
carbon_s = Table.read(data_dir + 'GALAH_carbon_cemp_dr53.fits')
galah = Table.read(data_dir + 'sobject_iraf_53_reduced_20180327.fits')
gaia = Table.read(data_dir + 'sobject_iraf_53_gaia.fits')

print 'feh < -1.5',np.sum(carbon_s['feh']<-1.5)
print 'cemops',np.sum(carbon_s['cemp_cand'])
print 'Usup, sup', np.sum(carbon_s['det_usup']),  np.sum(carbon_s['det_sup'])

print 'Join'
galah = join(galah, gaia, keys='sobject_id', join_type='left')

# orbit of a Solar-like object, used to normalise L_z
ts = np.linspace(0., 500, 5e4) * un.Myr
orbit_sun = Orbit(vxvv=[0. * un.deg,
                        0. * un.deg,
                        0.00001 * un.pc,
                        0. * un.mas / un.yr,
                        0. * un.mas / un.yr,
                        0. * un.km / un.s],
                  radec=True,
                  ro=8.2, vo=238., zo=0.025,
                  solarmotion=[-11., 10., 7.25])
orbit_sun.turn_physical_on()
L_ang_sun = orbit_sun.L()
print L_ang_sun

z_min = []
z_max = []
z_abs_max = []
L_ang = []
L_ang_z = []
vy = []
vx_vz = []
suffix = ''

# optional filters
# exclude data with large relative parallax error
# carbon_s = carbon_s[carbon_s['parallax_error']/carbon_s['parallax'] < 0.2]
# exclude far away stars
# carbon_s = carbon_s[1./carbon_s['parallax'] < 2.5]

# determine orbits for carbon-enhanced stars
for s_id in carbon_s['sobject_id']:
    print s_id
    star_data = galah[galah['sobject_id'] == s_id]

    orbit = Orbit(vxvv=[np.float64(star_data['ra_1']) * un.deg,
                        np.float64(star_data['dec_1']) * un.deg,
                        1e3 / np.float64(star_data['parallax']) * un.pc,
                        np.float64(star_data['pmra']) * un.mas / un.yr,
                        np.float64(star_data['pmdec']) * un.mas / un.yr,
                        np.float64(star_data['rv_guess']) * un.km / un.s],
                  radec=True,
                  ro=8.2, vo=238., zo=0.025,
                  solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing
    orbit.turn_physical_on()
    orbit.integrate(ts, MWPotential2014)
    orbit_xyz = [orbit.x(ts) * 1e3, orbit.y(ts) * 1e3, orbit.z(ts) * 1e3]
    print ' Z range:', np.min(orbit_xyz[2]), np.max(orbit_xyz[2]), ' pc'

    z_min.append(np.min(orbit_xyz[2]))
    z_max.append(np.max(orbit_xyz[2]))
    z_abs_max.append(np.max(np.abs(orbit_xyz[2])))

    L_ang.append(np.sqrt(np.sum(orbit.L()**2)))
    L_ang_z.append(orbit.L()[0][2])  # L_z component
    vy.append(orbit.vy())
    vx_vz.append(np.sqrt(orbit.vx()**2 + orbit.vz()**2))


pc_range = [0, 9]
n_bins = 70
idx_cemp = carbon_s['cemp_cand'] == True
idx_tsne = carbon_s['det_usup'] == True
z_max = np.array(z_abs_max)/1e3

idx_v = z_max > 5

xvals = np.linspace(0., 330, 50000)
yvals = np.sqrt(210**2 - (xvals - 238.)**2)

# plot Toomre diagram and mark CEMP candiates
plt.figure(figsize=(7,5.5))
plt.scatter(np.array(vy)[~idx_cemp], np.array(vx_vz)[~idx_cemp], c='black', lw=0, s=6, label='')
plt.scatter(np.array(vy)[idx_cemp], np.array(vx_vz)[idx_cemp], c='C3', lw=0, s=35, label='CEMP candidates', marker='*')
plt.plot(xvals, yvals, label='', c='C3', lw=1.5, alpha=0.5)
plt.xlim(-200, 320)
plt.ylim(0, 380)
plt.xlabel(r'Galactic $v_{y}$ [km s$^{-1}$]')
plt.ylabel(r'Galactic $\sqrt{v_{x}^2 + v_{z}^2}$ [km s$^{-1}$]')
plt.tight_layout()
plt.grid(ls='--', c='black', alpha=0.2)
plt.legend()
plt.savefig('carbon_orbits_vy_vxvz'+suffix+'.png', dpi=300)
# plt.show()
plt.close()

# plot angular momentum and maximmal z height of the stars
plt.figure(figsize=(7,5.5))
plt.scatter(np.array(L_ang_z/L_ang_sun[0][2])[~idx_cemp], z_max[~idx_cemp], c='black', lw=0, s=6, label='')
plt.scatter(np.array(L_ang_z/L_ang_sun[0][2])[idx_cemp], z_max[idx_cemp], c='C3', lw=0, s=35, label='CEMP candidates', marker='*')
plt.axvline(1000./L_ang_sun[0][2], c='C3', lw=1.5, alpha=0.5, label='', ls='--')
plt.xlim(-0.8, 1.5)
plt.ylim(0, 10)
plt.xlabel(r'Normalised angular momentum $L_{z}$/$L_{z\odot}$')
plt.ylabel(r'Maximal Galactic z height [kpc]')
plt.legend()
plt.tight_layout()
plt.grid(ls='--', c='black', alpha=0.2)
plt.savefig('carbon_orbits_zmax_lznorm'+suffix+'.png', dpi=300)
# plt.show()
plt.close()

# plot distribution maximmal heights
fig, ax = plt.subplots(2, 1,figsize=(7,5), sharex=True)
ax[0].hist(z_max, range=pc_range, bins=n_bins, alpha=0.3, color='black', label='Complete set')
ax[0].hist(z_max, range=pc_range, bins=n_bins, alpha=1., color='black', label='', histtype='step')
ax[0].legend()
ax[0].grid(ls='--', alpha=0.2, color='black')
ax[1].hist(z_max[idx_cemp], range=pc_range, bins=n_bins, alpha=0.3, color='C3', label='CEMP candidates')
ax[1].hist(z_max[idx_cemp], range=pc_range, bins=n_bins, alpha=1., color='C3', histtype='step', label='')
ax[1].legend()
ax[1].grid(ls='--', alpha=0.2, color='black')
ax[1].set(xlim=pc_range, ylim=(0,4.35), xlabel='Maximal Galactic z height [kpc]')#, ylabel='Number of stars')
fig.text(0.01, 0.5, '    Number of objects', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0., wspace=0., left=0.10, top=0.97, bottom=0.11, right=0.98)
plt.savefig('carbon_orbits_zmax_all'+suffix+'.png', dpi=300)
# plt.show()
plt.close()

print '> 5pc:', np.nanmean(carbon_s['feh'][z_max > 5])
print '< 5pc:', np.nanmean(carbon_s['feh'][z_max < 5])
