import jax.numpy as jnp
from jax import lax 
from jax import random as jrandom
import healpix 

from typing import Optional, Tuple

from .math_utils import cumtrapz


def redshifts_from_nz(size: int, z: jnp.ndarray, nz: jnp.ndarray, *,
    rng: Optional[jrandom.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:

    if rng is None:
        rng = jrandom.PRNGKey(0)

    if jnp.ndim(nz) > 1:
        pop = list(lax.ndindex(nz.shape[:-1]))
    else:
        pop = None

    cdf = cumtrapz(nz, z)

    p = cdf[..., -1]/cdf[..., -1].sum(axis=None, keepdims=True)

    cdf /= cdf[..., -1:]

    if pop is not None:
        x = jrandom.choice(rng, len(pop), p=p, shape=(size,))
        gal_z = jrandom.uniform(rng, (size,))
        for i, j in enumerate(pop):
            s = (x == i)
            gal_z = jnp.where(s, jnp.interp(gal_z, cdf[j], z), gal_z)
        gal_pop = jnp.take(pop, x)
    else:
        gal_z = jnp.interp(jrandom.uniform(rng, (size,)), cdf, z)
        gal_pop = None

    return gal_z, gal_pop


def galaxy_shear(lon: jnp.ndarray, lat: jnp.ndarray, eps: jnp.ndarray,
                 kappa: jnp.ndarray, gamma1: jnp.ndarray, gamma2: jnp.ndarray, *,
                 reduced_shear: bool = True) -> jnp.ndarray:
    # CHANGE: Can we switch nside = healpix.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1]) to:
    # CHANGE in call: lax wants integer types for all arguments?
    nside = healpix.npix2nside(lax.broadcast(gamma1, gamma2).shape[-1])

    # CHANGE: Can we switch size = np.broadcast(lon, lat, eps).size to:
    size = lax.broadcast(lon, lat).size

    k = jnp.empty(size)
    g = jnp.empty(size, dtype=jnp.complex64)

    for i in range(0, size, 10000):
        s = slice(i, i+10000)
        ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
        k[s] = kappa[ipix]
        g.real[s] = gamma1[ipix]
        g.imag[s] = gamma2[ipix]

    if reduced_shear:
        g /= 1 - k
        g = (eps + g)/(1 + jnp.conj(g)*eps)
    else:
        g += eps

    return g


def gaussian_phz(z: jnp.ndarray, sigma_0: float,
                 rng: Optional[jrandom.PRNGKey] = None) -> jnp.ndarray:
    if rng is None:
        rng = jnp.random.default_rng()

    size = jnp.shape(z)
    z = jnp.reshape(z, (-1,))

    zphot = rng.normal(z, (1 + z)*sigma_0)

    trunc = jnp.where(zphot < 0)[0]
    while trunc.size:
        zphot = jnp.where(zphot < 0, rng.normal(z[trunc], (1 + z[trunc])*sigma_0), zphot)
        trunc = trunc[zphot[trunc] < 0]

    return zphot.reshape(size)
