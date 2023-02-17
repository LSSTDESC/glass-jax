import jax
import jax.numpy as jnp
import jax.random as random
import healpy as hp

from math_utils import ARCMIN2_SPHERE, trapz_product


def effective_bias(bias_z, bias_b, window_z, window_w):
    norm = jnp.trapz(window_w, window_z)
    return trapz_product((bias_z, bias_b), (window_z, window_w))/norm


def linear_bias(delta, b):
    return b*delta


def loglinear_bias(delta, b):
    delta_g = jnp.log1p(delta)
    delta_g *= b
    # Switch np.expm1(delta_g, out=delta_g) to:
    delta_g = jnp.expm1(delta_g)
    return delta_g


def positions_from_delta(ngal, delta, bias=None, vis=None,
                         bias_model=linear_bias, remove_monopole=False, rng=None):
    if rng is None:
        rng = random.PRNGKey(0)

    if not callable(bias_model):
        raise ValueError('bias_model must be callable')

    if bias is None:
        n = jnp.copy(delta)
    else:
        n = bias_model(delta, bias)

    if remove_monopole:
        n -= jnp.mean(n, keepdims=True)

    n += 1
    n *= ARCMIN2_SPHERE/n.size*ngal

    if vis is not None:
        n *= vis

    # Switch jnp.clip(n, 0, None, out=n) to:
    n = jnp.clip(n, 0, None)
    
    n = random.poisson(rng, n)

    ntot = n.sum()

    npix = n.shape[-1]
    nside = hp.npix2nside(npix)

    lon = jnp.empty(ntot)
    lat = jnp.empty(ntot)

    batch = 10_000
    ncur = 0
    for i in range(0, npix, batch):
        k = n[i:i+batch]
        bpix = jnp.repeat(jnp.arange(i, i+k.size), k)
        blon, blat = hp.pix2ang(nside, bpix, lonlat=True)

        # CHANGE: From lon[ncur:ncur+blon.size] = blon to:
        lon.at[jnp.index_exp[ncur:ncur+blon.size]].set(blon)
        lat.at[jnp.index_exp[ncur:ncur+blon.size]].set(blat)

        ncur += bpix.size

    assert ncur == ntot, 'internal error in sampling'

    return lon, lat


def uniform_positions(ngal, *, rng=None):
    if rng is None:
        rng = random.PRNGKey(0)

    ntot = random.poisson(rng, ARCMIN2_SPHERE*ngal)
    print(ntot)
    lon = random.uniform(rng, shape=(ntot, ), minval=-180, maxval=180)
    lat = jnp.rad2deg(jnp.arcsin(random.uniform(rng, shape=(ntot, ), minval=-1, maxval=1)))

    return lon, lat
