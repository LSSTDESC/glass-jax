import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

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
ws = glass.shells.tophat_windows(zb)

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
z = np.arange(0.1, 3., 0.01)
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
# %%

# %%
# Simulation
# ----------
# Simulate maps of lensing and clustering, without actually sampling
# from them

# Shear maps
kap_bar = np.stack([np.zeros(hp.nside2npix(nside)) for i in range(5)],axis=0)
gam1_bar = np.stack([np.zeros(hp.nside2npix(nside)) for i in range(5)],axis=0)
gam2_bar = np.stack([np.zeros(hp.nside2npix(nside)) for i in range(5)],axis=0)
gam1_ia_bar = np.stack([np.zeros(hp.nside2npix(nside)) for i in range(5)],axis=0)
gam2_ia_bar = np.stack([np.zeros(hp.nside2npix(nside)) for i in range(5)],axis=0)

# simulate the matter fields in the main loop, and build up the catalogue
for i, delta_i in enumerate(matter):

    # compute the lensing maps for this shell
    convergence.add_window(delta_i, ws[i])

    kappa_i = convergence.kappa
    gamma_i = glass.lensing.from_convergence(kappa_i, shear=True, discretized=False)

    #gamm1_i_jax, gamm2_i_jax = glass.shear_from_convergence.get_shear(kappa_i, nside)    

    hp.mollview(gamma_i[0].imag)
    plt.show()

    #hp.mollview(gamm1_i_jax)
    #plt.show()

    #hp.mollview(gamma_i.real - gamm1_i_jax)
    #plt.show()  
