import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import warnings

from .math_utils import ndinterp

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
    # if zbins[0] != 0:
    #     warnings.warn('first tophat window does not start at redshift zero')

    wf: WeightFunc
    if wfunc is not None:
        wf = wfunc
    else:
        wf = jnp.ones_like

    zs, ws = [], []
    for zmin, zmax in zip(zbins, zbins[1:]):
        # THIS WILL NOT JIT
        # n = max(round((zmax - zmin)/dz), 2)
        # z = jnp.linspace(zmin, zmax, n)
        z = np.linspace(0,4,int(4./dz))
        zs.append(z)
        ws.append(wf(z))
    return zs, ws

def restrict(z: ArrayLike1D, f: ArrayLike1D, w: 'RadialWindow'
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''Restrict a function to a redshift window.
    Multiply the function :math:`f(z)` by a window function :math:`w(z)`
    to produce :math:`w(z) f(z)` over the support of :math:`w`.
    The function :math:`f(z)` is given by redshifts ``z`` of shape
    *(N,)* and function values ``f`` of shape *(..., N)*, with any
    number of leading axes allowed.
    The window function :math:`w(z)` is given by ``w``, which must be a
    :class:`RadialWindow` instance or compatible with it.
    The restriction has redshifts that are the union of the redshifts of
    the function and window over the support of the window.
    Intermediate function values are found by linear interpolation
    Parameters
    ----------
    z, f : array_like
        The function to be restricted.
    w : :class:`RadialWindow`
        The window function for the restriction.
    Returns
    -------
    zr, fr : array
        The restricted function.
    '''
    #TODO: use functions instead of tabulated values
    # z_ = jnp.compress(jnp.greater(z, w.za[0]) & jnp.less(z, w.za[-1]), z)
    zr = z #jnp.union1d(w.za, z)
    fr = f * jnp.interp(z, w.za, w.wa)
    return zr, fr


def distance_grid(cosmo, zmin, zmax, *, dx=None, num=None):
    '''Redshift grid with uniform spacing in comoving distance.'''
    xmin = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(zmin))/cosmo.h
    xmax = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(zmax))/cosmo.h
    print(xmin[0], xmax[0])
    if dx is not None and num is None:
        x = jnp.arange(xmin[0], jnp.nextafter(xmax[0]+dx, xmax[0]), dx)
    elif dx is None and num is not None:
        x = jnp.linspace(xmin[0], xmax[0], num+1)
    else:
        raise ValueError('exactly one of "dx" or "num" must be given')
    return jc.utils.a2z(jc.background.a_of_chi(cosmo, x*cosmo.h))
