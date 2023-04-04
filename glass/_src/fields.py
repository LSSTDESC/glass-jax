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

def multalm(alm: Alms, bl: Array, inplace: bool = False) -> Alms:
    '''multiply alm by bl'''
    n = len(bl)
    if inplace:
        out = jnp.asanyarray(alm)
    else:
        out = jnp.copy(alm)
    for m in range(n):
        out = out.at[m*n-m*(m-1)//2:(m+1)*n-m*(m+1)//2].multiply(bl[m:])
    return out

def generate_gaussian(gls: Cl, 
                      nside: int, *,
                      L: Optional[int],
                      rng: Optional[np.random.Generator] = None):

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()
    
    sampling= "healpix"

    for i, cl in enumerate(gls):

        # Sample gaussian map
        z = rng.standard_gamma(hp.nside2npix(nside))
        
        # Convert to alms
        alm = s2fft.forward_jax(z, L, 
                                reality=True, 
                                sampling=sampling, 
                                nside=nside)

        # Multiply alms by power spectrum
        alm = multalm(alm, cl)

        # Inverse SHT
        yield s2fft.inverse_jax(alm, L, 
                                reality=True, 
                                sampling=sampling, 
                                nside=nside)