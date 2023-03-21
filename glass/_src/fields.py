import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import s2fft


# typing
from typing import (Any, Union, Tuple, Generator, Optional, Sequence, Callable,
                    Iterable)

Array = jnp.ndarray
Cl = Array
Alms = np.ndarray

def generate_gaussian(gls, 
                      nside,
                      L,
                      seed):
    sampling= "healpix"
    mean_pixel_area = 4 * np.pi / hp.nside2npix(nside)
    scaling_factor = np.sqrt(1 / mean_pixel_area)
    keys = jax.random.split(seed, len(gls))

    for i, cl in enumerate(gls):

        # Sample gaussian map
        z = jax.random.normal(key=keys[i], shape=(hp.nside2npix(nside),)) * scaling_factor

        # Convert to alms
        alm = s2fft.forward_jax(z, L, 
                                reality=True, 
                                sampling=sampling, 
                                nside=nside)

        # Multiply alms by power spectrum
        alm =  alm * jnp.sqrt(cl.reshape([L, 1]))

        # Inverse SHT
        yield s2fft.inverse_jax(alm, L, 
                                reality=True, 
                                sampling=sampling, 
                                nside=nside)
        
