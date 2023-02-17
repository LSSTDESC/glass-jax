import sys
sys.path.append("/Users/eleni/Desktop/desc/glass-jax/glass/")

import numpy as np


""" 
Here we check if new jax modules pass basic syntax checks,
formulated as simple calls to each jaxified function.
"""

import observations as jax_obs

jax_obs.vmap_galactic_ecliptic(nside=64)
jax_obs.gaussian_nz(z=np.ones(1), mean=np.ones(1), sigma=np.ones(1))
jax_obs.smail_nz(z=np.ones(1), z_mode=np.ones(1), alpha=1, beta=1)
jax_obs.fixed_zbins(zmin=0, zmax=1.5, nbins=2)
jax_obs.equal_dens_zbins(z=np.ones(10), nz=np.ones(10), nbins=2)
jax_obs.tomo_nz_gausserr(z=np.ones(10), nz=np.ones(10), sigma_0=1, zbins=[np.linspace(0,1,1), np.linspace(0,1,1)])

import math_utils as jax_mu

jax_mu.ndinterp(np.ones(10), np.linspace(0,1,10), np.linspace(0,1,10))
jax_mu.trapz_product((np.linspace(0,1,10), np.linspace(0,1,10)), (np.linspace(0,1,10), np.linspace(0,1,10)))
jax_mu.cumtrapz(np.linspace(0,1,10), np.linspace(0,1,10))

import galaxies as jax_gals

jax_gals.redshifts_from_nz(size=10, z=np.linspace(0,1,10), nz=np.linspace(0,1,10))
jax_gals.galaxy_shear(lon=np.arange(0,1,10), lat=np.arange(0,1,10), eps=np.arange(0,1,10), kappa=np.arange(0,1,10), gamma1=np.arange(0,1,10), gamma2=np.arange(0,1,10))

import points as jax_pts

jax_pts.effective_bias(np.linspace(0.,1,10), np.linspace(0.,1,10), np.linspace(0.,1,10), np.linspace(0.,1,10))
jax_pts.linear_bias(1,2)
jax_pts.loglinear_bias(np.ones(1), 2*np.ones(1))
jax_pts.positions_from_delta(np.ones(12*1**2),np.ones(12*1**2))
jax_pts.uniform_positions(0.00001)
