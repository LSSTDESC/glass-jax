import dataclasses
import jax.numpy as jnp
import jax.scipy as jsp
import jax_cosmo as jc
import glass.jax as jglass
import pyccl.ccllib as lib
import healpy as hp
import jax
import s2fft
import pylab as plt


rho_crit = lib.cvar.constants.RHO_CRITICAL

def tidal_tensor_jax(hpmap):

    delta = jnp.copy(hpmap)
    nside = hp.get_nside(hpmap)

    alm_E = s2fft.forward_jax(hpmap, 512, reality=True, sampling="healpix", nside=nside)
    print('s2fft', jnp.shape(alm_E))
    print(alm_E)

    alm_E_base = hp.map2alm(hpmap, lmax=512, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    print('healpy', jnp.shape(alm_E_base))
    print(alm_E_base)

    print("diff", alm_E - alm_E_base)

    plt.plot(alm_E)
    plt.plot(alm_E_base)
    plt.show()

    alm_B = jnp.zeros_like(alm_E)

    # smoothing if necessary here

    maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2, lmax=5000, mmax=None)

    tidal_tensor_sph = jnp.zeros((hp.nside2npix(nside), 3), dtype=jnp.float32)

    tidal_tensor_sph = jnp.concatenate([
      ((maps_QU[0] + delta) / 2.0 - 1.3 * hpmap)[:,jnp.newaxis],
      ((delta - maps_QU[0]) / 2.0 - 1.3 * hpmap)[:,jnp.newaxis],
      maps_QU[1][:,jnp.newaxis]], axis=1)

    return tidal_tensor_sph

def tidal_tensor(hpmap):

    delta = jnp.copy(hpmap)

    nside = hp.get_nside(hpmap)

    alm_E = hp.map2alm(hpmap, lmax=5000, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    alm_B = jnp.zeros_like(alm_E) # FIXME: jaxify

    # smoothing if necessary here

    maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2, lmax=5000, mmax=None)

    tidal_tensor_sph = jnp.zeros((hp.nside2npix(nside), 3), dtype=jnp.float32)

    tidal_tensor_sph = jnp.concatenate([
      ((maps_QU[0] + delta) / 2.0 - 1.3 * hpmap)[:,jnp.newaxis],
      ((delta - maps_QU[0]) / 2.0 - 1.3 * hpmap)[:,jnp.newaxis],
      maps_QU[1][:,jnp.newaxis]], axis=1)

    return tidal_tensor_sph

def Epsilon1_NLA(cosmo, z, sxx, syy, A1):
    gz = jc.background.growth_factor(cosmo, 1./(1+z))
    Fact = -1. * A1 * 5e-14 * rho_crit * cosmo.Omega_c / gz
    e1_NLA = Fact * (sxx - syy)
    return e1_NLA

def Epsilon2_NLA(cosmo, z, sxy, A1):
    gz = jc.background.growth_factor(cosmo, 1./(1+z))
    Fact = -1. * A1 * 5e-14 * rho_crit * cosmo.Omega_c / gz
    e2_NLA = 2 * Fact * sxy
    return e2_NLA

def Epsilon1_TT(cosmo, z, sxx, syy, A2):
    gz = jc.background.growth_factor(cosmo, 1./(1+z))
    Fact = 5 * A2 * 5e-14 * rho_crit * cosmo.Omega_c / gz**2
    e1_TT = Fact * (syy**2 - sxx**2)
    return e1_TT

def Epsilon2_TT(cosmo, z, sxy, sxx, syy, A2):
    gz = jc.background.growth_factor(cosmo, 1./(1+z))
    Fact = 5 * A2 * 5e-14 * rho_crit * cosmo.Omega_c / gz**2
    e2_TT = Fact * sxy * (sxx + syy)
    return e2_TT

def NLA(cosmo, z, A1, tidal_field):
    
    sxx = tidal_field[:,0]
    syy = tidal_field[:,1]
    sxy = tidal_field[:,2]

    e1_NLA = Epsilon1_NLA(cosmo, z, sxx, syy, A1)
    e2_NLA = Epsilon2_NLA(cosmo, z, sxy, A1)

    epsilon_NLA = jnp.empty(e1_NLA.shape, dtype=complex)
    epsilon_NLA = e1_NLA + 1j * e2_NLA

    return epsilon_NLA

def TATT(cosmo, z, A1, bTA, A2, tidal_field, delta):

    sxx = tidal_field[:,0]
    syy = tidal_field[:,1]
    sxy = tidal_field[:,2]

    e1_NLA = Epsilon1_NLA(cosmo, z, sxx, syy, A1)
    e2_NLA = Epsilon2_NLA(cosmo, z, sxy, A1)

    e1_TT = Epsilon1_TT(cosmo, z, sxx, syy, A2)
    e2_TT = Epsilon2_TT(cosmo, z, sxy, sxx, syy, A2)

    epsilon_NLA  = jnp.array(e1_NLA + 1j * e2_NLA)
    epsilon_TATT = jnp.array(e1_TT  + 1j * e2_TT)

    epsilon_IA_TATT = epsilon_NLA * (1. + bTA * delta) + epsilon_TATT

    return epsilon_IA_TATT

def get_IA(z, density_planes, A1=0.18, bTA=0.8, A2=0.1, model='NLA'):

    tidal_tensor_map = tidal_tensor_jax(density_planes)

    cosmo = jc.Planck15() #FIXME pass cosmo as argument (camb -> jax)
    redshift = jnp.atleast_1d(z)

    if(model == 'TATT'):
      return TATT(cosmo, redshift, A1, bTA, A2, tidal_tensor_map, density_planes)
    elif(model == 'NLA'):
      return NLA(cosmo, redshift, A1, tidal_tensor_map)
    else:
      raise ValueError(f'Unknown IA method')

