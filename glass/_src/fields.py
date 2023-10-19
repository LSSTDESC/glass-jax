import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import s2fft
from jax import jit, vmap
from mcfit import C2w, w2C

# typing
from typing import Any, Union, Tuple, Generator, Optional, Sequence, Callable, Iterable

Array = jnp.ndarray
Cl = Array
Alms = np.ndarray


def generate_gaussian(gls, nside, L, seed):
    sampling = "healpix"
    mean_pixel_area = 4 * np.pi / hp.nside2npix(nside)
    scaling_factor = np.sqrt(1 / mean_pixel_area)
    keys = jax.random.split(seed, len(gls))

    for i, cl in enumerate(gls):
        # Sample gaussian map
        z = (
            jax.random.normal(key=keys[i], shape=(hp.nside2npix(nside),))
            * scaling_factor
        )

        # Convert to alms
        alm = s2fft.forward_jax(z, L, reality=True, sampling=sampling, nside=nside)

        # Multiply alms by power spectrum
        alm = alm * jnp.sqrt(cl.reshape([L, 1]))

        # Inverse SHT
        yield s2fft.inverse_jax(alm, L, reality=True, sampling=sampling, nside=nside)


def generate_lognormal(gls, nside, shift, L, seed):
    # gls have length number of bins, it is just the auto spectra of the matter shells
    # we will just do individual Hankel transforms on each bin since they are not correlated
    # TODO: L max for Hankel transform should be much larger than the target ell range
    # we will just do this approximately here
    # here we just assume C2w and w2C carries information of the ell ranegs of gls

    nshell = len(gls)
    nell_gls = gls[0].shape[-1]
    lmax_gls = nell_gls - 1
    print("nshell", nshell)
    print("nell", nell_gls)

    # gls along integer ells
    ell = jnp.arange(nell_gls)
    # we need log spaced gls'
    ell_log = jnp.logspace(0, jnp.log10(lmax_gls), 400)
    gls_log = jnp.array(
        [(jnp.interp(ell_log, ell, gls[i][0])).reshape([1, -1]) for i in range(nshell)]
    )

    # Hankel transforms
    # broadcasting to the last axis
    C2w_f = C2w(ell_log, backend="jax")  # C -> w function
    theta = C2w_f.y
    w2C_f = w2C(theta, backend="jax")  # w -> C function

    # ell_query defined by lmax = L (not inclusive)
    ell_query = jnp.arange(L)

    # TODO: the Hankel transform can already be done in batch, will rewrite this part later
    gls_log_gauss = jnp.zeros((nshell, 1, L))
    for i, (C_ln, a) in enumerate(zip(gls_log, shift)):
        C_ln = C_ln[0]  # cl here has shape (1, nells)
        C_gauss = _calc_C_gauss(
            a=a, C_ln=C_ln, C2w_f=C2w_f, w2C_f=w2C_f, ell_query=ell_query
        )
        C_ln = C_gauss.reshape([1, -1])

        gls_log_gauss = gls_log_gauss.at[i].set(C_ln)

    for i, map_g in enumerate(generate_gaussian(gls_log_gauss, nside, L, seed)):
        mag_ln = (jnp.exp(map_g - jnp.var(map_g) / 2) - 1) * shift[i]
        yield mag_ln


def _interp_farray(x_query, x, f_x):
    f_axis0 = vmap(jnp.interp, in_axes=(None, None, 0))
    f_axis1 = vmap(f_axis0, in_axes=(None, None, 0))
    return f_axis1(x_query, x, f_x)


def _calc_C_gauss(a, C_ln, C2w_f, w2C_f, ell_query):
    # calculate a single auto power spectra for a lognormal field
    # a scalar
    # C_ln 1-d array (nells,)
    # this routine is much faster than the full cov counter part since no eigh is needed

    _theta, w_ln = C2w_f(C_ln)  # _theta logspaced

    w_gauss = 1 + w_ln / a / a
    w_gauss = jnp.where(w_gauss <= 0, 1e-18, w_gauss)
    w_gauss = jnp.log(w_gauss)

    # Hankel transform, broadcasted to the last axis
    _ell, C_gauss = w2C_f(w_gauss)

    # interpolate
    C_gauss_interp = jnp.interp(x=ell_query, xp=_ell, fp=C_gauss)

    return C_gauss_interp


def _calc_C_gauss_cov(a, C_ln, C2w_f, w2C_f, ell_query):
    # calculate the power spectra covariance of the correlated lognormal field
    # a 1-d array (nbins)
    # C_ln 3-d array (nbins, nbins, nells,)

    _theta, w_ln = C2w_f(C_ln)  # _theta logspaced

    w_gauss = 1 + w_ln / (jnp.outer(a, a)[:, :, None])
    w_gauss = jnp.where(w_gauss <= 0, 1e-18, w_gauss)
    w_gauss = jnp.log(w_gauss)

    # Hankel transform, broadcasted to the last axis
    _ell, C_gauss = w2C_f(w_gauss)

    # interpolate
    C_gauss_interp = _interp_farray(x_query=ell_query, x=_ell, f_x=C_gauss)

    C_gauss_interp_blk = jnp.swapaxes(C_gauss_interp, 2, 0)
    blk_L, blk_U = jnp.linalg.eigh(C_gauss_interp_blk)
    blk_L = jnp.where(blk_L <= 0, 1e-10, blk_L)
    blk_L = jnp.array([jnp.diag(L) for L in blk_L])
    tmp = jnp.einsum("lrs,lts -> lrt", blk_L, blk_U)
    C_gauss_interp_mod_blk = jnp.einsum("lrs,lst -> lrt", blk_U, tmp)

    C_gauss_mod = jnp.swapaxes(C_gauss_interp_mod_blk, 0, 2)

    return C_gauss_mod
