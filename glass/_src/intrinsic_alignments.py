import dataclasses
import jax.numpy as jnp
import jax.scipy as jsp
import jax_cosmo as jc

def computeIA(self, density_planes):
  """Computes IA from NLA or TATT model and outputs the 
  2 (real and imag) components"""

  if self.IA_method == 'NLA':
    return self._NLA(density_planes)
  elif self.IA_method == 'TATT':
    return self._TATT(density_planes)
  else:
    raise ValueError(f'Unknown IA method')

def tidal_tensor(hpmap):
    alm_E = hp.map2alm(hpmap, lmax=5000, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    alm_B = jnp.zeros_like(alm_E)

    # smoothing if necessary here

    nside = hp.get_nside(hpmap)
    maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2, lmax=5000, mmax=None)

    tidal_tensor_sph = jnp.zeros((hp.nside2npix(nside), 3), dtype=jnp.float32)

    tidal_tensor_sph[:,0] = (maps_QU[0] + delta) / 2.0 - 1.3 * hpmap
    tidal_tensor_sph[:,1] = (delta - maps_QU[0]) / 2.0 - 1.3 * hpmap
    tidal_tensor_sph[:,2] = maps_QU[1]

    return tidal_tensor_sph

def Epsilon1_NLA(cosmo, z, A1, rho_crit, sxx, syy):
    gz = jc.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    e1_NLA = Fact * (sxx - syy)
    return e1_NLA

def Epsilon2_NLA(cosmo, z, A1, rho_crit, sxy):
    gz = jc.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    e2_NLA = 2 * Fact * sxy
    return e2_NLA

def Epsilon1_TT(cosmo, z, A2, rho_crit, sxx, syy):
    gz = jc.growth_factor(cosmo, 1./(1+z))
    Fact = -5*A2*5e-14*rho_crit*cosmo['Omega_m']/gz**2
    e1_TT = Fact * (syy**2 - sxx**2)
    return e1_TT

def Epsilon2_TT(cosmo, z, A2, rho_crit, sxy, sxx, syy):
    gz = jc.growth_factor(cosmo, 1./(1+z))
    Fact = -5*A2*5e-14*rho_crit*cosmo['Omega_m']/gz**2
    e2_TT = Fact * sxy * (sxx + syy)
    return e2_TT

def NLA(cosmo, z, A_IA, Aia, rho_crit, s11, s12, s22):
    e1_NLA = Epsilon1_NLA(cosmo, z, Aia, rho_crit, s11, s22)
    e2_NLA = Epsilon2_NLA(cosmo, z, Aia, rho_crit, s12)

    epsilon_NLA = jnp.empty(e1_NLA.shape, dtype=complex)
    epsilon_NLA.real = A_IA * e1_NLA
    epsilon_NLA.imag = A_IA * e2_NLA

    return epsilon_NLA

def TATT(density_planes):

    e1_NLA = Epsilon1_NLA(cosmo, z, Aia, rho_crit, s11, s22)
    e2_NLA = Epsilon2_NLA(cosmo, z, Aia, rho_crit, s12)

    e1_TT = Epsilon1_TT(cosmo, z, C2, rho_crit, s11, s22)
    e2_TT = Epsilon2_TT(cosmo, z, C2, rho_crit, s12, s11, s22)

    epsilon_NLA = jnp.array(A_IA * e1_NLA + 1j * A_IA * e2_NLA)
    epsilon_TATT = jnp.array(A_IA * e1_TT + 1j * A_IA * e2_TT)

    epsilon_IA_TATT = epsilon_NLA * (1. + b_TA * delta) + epsilon_TATT

    return epsilon_IA_TATT
