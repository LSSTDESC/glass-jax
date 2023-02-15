# import jax
# import jax.numpy as jnp
# import numpyro
# import healpy as hp

# # typing
# from typing import (Any, Union, Tuple, Generator, Optional, Sequence, Callable,
#                     Iterable)

# Array = jnp.ndarray
# Cl = Array

# def generate_gaussian(gl: Cl, 
#                       nside: int):
    
#     # Sample random normal map
#     gaussian_map = numpyro.sample('gaussian_map',
#                                   numpyro.distributions.Normal(jnp.zeros(hp.nside2npix(nside)), 1))

#     # Convert to alms, multiply by gl, convert back to map
#     gaussian_map = hp.alm2map(hp.map2alm(gaussian_map) * gl, nside)
#     return gaussian_map
