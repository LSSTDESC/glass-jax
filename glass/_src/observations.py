import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import healpy as hp

from typing import Optional, Tuple, List
from numpy.typing import ArrayLike

from .math_utils import cumtrapz


def vmap_galactic_ecliptic(nside: int, galactic: Tuple[float, float] = (30, 90),
                           ecliptic: Tuple[float, float] = (20, 80)
                           ) -> jnp.ndarray:
    if jnp.ndim(galactic) != 1 or len(galactic) != 2:
        raise TypeError('galactic stripe must be a pair of numbers')
    if jnp.ndim(ecliptic) != 1 or len(ecliptic) != 2:
        raise TypeError('ecliptic stripe must be a pair of numbers')

    m = jnp.ones(hp.nside2npix(nside))

    m.at[hp.query_strip(nside, *galactic)].set(0)
    m = hp.Rotator(coord='GC').rotate_map_pixel(m)
    m.at[hp.query_strip(nside, *ecliptic)].set(0) 
    m = hp.Rotator(coord='CE').rotate_map_pixel(m)

    return m


def gaussian_nz(z: jnp.ndarray, mean: ArrayLike, sigma: ArrayLike, *,
                norm: Optional[ArrayLike] = None) -> jnp.ndarray:
    mean = jnp.reshape(mean, mean.shape + (1,)*(z.ndim-1))
    sigma = jnp.reshape(sigma, sigma.shape + (1,)*(z.ndim-1))

    nz = jax.scipy.stats.norm.pdf(z, loc=mean, scale=sigma)
    nz = jnp.exp(nz - jax.scipy.special.logsumexp(nz, axis=-1, keepdims=True))

    if norm is not None:
        nz *= norm

    return nz


def smail_nz(z: jnp.ndarray, z_mode: ArrayLike, alpha: ArrayLike,
             beta: ArrayLike, *, norm: Optional[ArrayLike] = None
             ) -> jnp.ndarray:
    z_mode = jnp.asarray(z_mode)[..., jnp.newaxis]
    alpha = jnp.asarray(alpha)[..., jnp.newaxis]
    beta = jnp.asarray(beta)[..., jnp.newaxis]

    pz = z**alpha*jnp.exp(-alpha/beta*(z/z_mode)**beta)
    pz /= jnp.trapz(pz, z, axis=-1)[..., jnp.newaxis]

    if norm is not None:
        pz *= norm

    return jnp.asarray(pz)


def fixed_zbins(zmin: float, zmax: float, nbins: int = None, dz: float = None
    ) -> List[Tuple[float, float]]:
    if nbins is not None and dz is None:
        zbinedges = jnp.linspace(zmin, zmax, nbins+1)
    elif nbins is None and dz is not None:
        zbinedges = jnp.arange(zmin, zmax, dz)
    else:
        raise ValueError('exactly one of nbins and dz must be given')

    return list(zip(zbinedges, zbinedges[1:]))


def equal_dens_zbins(z: jnp.ndarray, nz: jnp.ndarray, nbins: int
    ) -> List[Tuple[float, float]]:
    cuml_nz = cumtrapz(nz, z) 
    # CHANGE: Change cuml_nz[[-1]] to cuml_nz[-1]
    cuml_nz /= cuml_nz[-1]
    zbinedges = jnp.interp(jnp.linspace(0, 1, nbins+1), cuml_nz, z)

    return list(zip(zbinedges, zbinedges[1:]))


def tomo_nz_gausserr(z: jnp.ndarray, nz: jnp.ndarray, sigma_0: float, zbins: List[Tuple[float, float]]
    ) -> jnp.ndarray:
    zbins_arr = jnp.asarray(zbins)

    z_lower = zbins_arr[:, 0, jnp.newaxis]
    z_upper = zbins_arr[:, 1, jnp.newaxis]

    sz = jnp.sqrt(2) * sigma_0 * (1 + z)
    binned_nz = jsp.erf((z - z_lower) / sz)
    binned_nz -= jsp.erf((z - z_upper) / sz)
    binned_nz /= 1 + jsp.erf(z / sz)
    binned_nz *= nz

    return binned_nz
