import jax.numpy as jnp
import jax_cosmo as jc

import warnings

# type checking
from typing import (Union, Sequence, List, Tuple, Optional, Callable,
                    TYPE_CHECKING)
# types
ArrayLike1D = Union[Sequence[float], jnp.ndarray]
WeightFunc = Callable[[ArrayLike1D], jnp.ndarray]


def tophat_windows(zbins: ArrayLike1D, dz: float = 1e-3,
                   wfunc: Optional[WeightFunc] = None
                   ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    '''Tophat window functions from the given redshift bin edges.

    Uses the *N+1* given redshifts as bin edges to construct *N* tophat
    window functions.  The redshifts of the windows have linear spacing
    approximately equal to ``dz``.

    An optional weight function :math:`w(z)` can be given using
    ``wfunc``; it is applied to the tophat windows.

    Parameters
    ----------
    zbins : (N+1,) array_like
        Redshift bin edges for the tophat window functions.
    dz : float, optional
        Approximate spacing of the redshift grid.
    wfunc : callable, optional
        If given, a weight function to be applied to the window
        functions.

    Returns
    -------
    zs, ws : (N,) list of array_like
        List of window functions.

    '''
    if len(zbins) < 2:
        raise ValueError('zbins must have at least two entries')
    if zbins[0] != 0:
        warnings.warn('first tophat window does not start at redshift zero')

    wf: WeightFunc
    if wfunc is not None:
        wf = wfunc
    else:
        wf = jnp.ones_like

    zs, ws = [], []
    for zmin, zmax in zip(zbins, zbins[1:]):
        n = max(round((zmax - zmin)/dz), 2)
        z = jnp.linspace(zmin, zmax, n)
        zs.append(z)
        ws.append(wf(z))
    return zs, ws


def distance_grid(cosmo, zmin, zmax, *, dx=None, num=None):
    '''Redshift grid with uniform spacing in comoving distance.'''
    xmin = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(zmin))
    xmax = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(zmax))
    if dx is not None and num is None:
        x = jnp.arange(xmin, jnp.nextafter(xmax+dx, xmax), dx)
    elif dx is None and num is not None:
        x = jnp.linspace(xmin, xmax, num+1)
    else:
        raise ValueError('exactly one of "dx" or "num" must be given')
    return jc.utils.a2z(jc.background.a_of_chi(cosmo, x))
