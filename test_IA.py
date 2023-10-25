import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glass.jax as jglass

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology import Cosmology

# GLASS modules: cosmology and everything in the glass namespace
import glass.shells
import glass.fields
import glass.shapes
import glass.lensing
import glass.observations

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
nside = lmax = 256

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# get the cosmology from CAMB
cosmo = Cosmology.from_camb(pars)

# %%
# Set up the matter sector.

# shells of 200 Mpc in comoving distance spacing
zb = glass.shells.distance_grid(cosmo, 0., 3., dx=200.)

# tophat window function for shells
zs, ws = glass.shells.tophat_windows(zb)

# compute the angular matter power spectra of the shells with CAMB
#cls = glass.camb.matter_cls(pars, lmax, zs, ws)
#np.save(cls)
cls = np.load('cls.npy')
# compute Gaussian cls for lognormal fields for 3 correlated shells
# putting nside here means that the HEALPix pixel window function is applied
gls = glass.fields.lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=3)

# generator for lognormal matter fields
matter = glass.fields.generate_lognormal(gls, nside, ncorr=3)
# np.save('cls.npy', cls)
# %%
# Set up the lensing sector.

# this will compute the convergence field iteratively
convergence = glass.lensing.MultiPlaneConvergence(cosmo)

# %%
# Set up the galaxies sector.

# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# true redshift distribution following a Smail distribution
z = np.arange(0.01, 3., 0.01)
dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
dndz *= n_arcmin2

# compute bin edges with equal density
nbins = 5
zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)

# photometric redshift error
sigma_z0 = 0.03

# split distribution by tomographic bin, assuming photometric redshift errors
tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# constant bias parameter for all shells
bias = 1.2

# ellipticity standard deviation as expected for a Stage-IV survey
sigma_e = 0.27

for i, delta_i in enumerate(matter):
        
    ia = jglass.intrinsic_alignments.get_IA(zb[i], delta_i, nside, A1=0.18, bTA=0.8, A2=0.1, model='NLA')

    hp.mollview(np.log10(2+delta_i))
    plt.show()

    hp.mollview(ia.real)
    plt.show()

    hp.mollview(ia.imag)
    plt.show()


    ia = jglass.intrinsic_alignments.get_IA(zb[i], delta_i, nside, A1=0.18, bTA=0.8, A2=0.1, model='TATT')

    hp.mollview(ia.real)
    plt.show()

    hp.mollview(ia.imag)
    plt.show()

