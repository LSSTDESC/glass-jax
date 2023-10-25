import healpy as hp
import jax.numpy as jnp
import s2fft
import numpy as np
import jax
from s2fft.sampling import s2_samples as samples


def get_shear(kappa, nside, lmax=None, discretized=True):

    if lmax is None:
        lmax = 3*nside - 1

    L = 2*nside

    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)
    alm_m2 = s2fft.forward_jax(kappa, L, spin=-2, reality=True, sampling="healpix", nside=nside)
    alm_p2 = s2fft.forward_jax(kappa, L, spin=2, reality=True, sampling="healpix", nside=nside)

    alm_E = -0.5 * (alm_p2 + alm_m2)     # replace this with the kaiser squires
    alm_B = 0.5 * 1j * (alm_p2 - alm_m2) # replace this with the kaiser squires

    blm = jnp.zeros_like(alm_E)

    l = jnp.arange(0.,lmax+1, 1)
    fl = ((l+2)*(l+1)*l*(l-1))**0.5
    fl /= jnp.clip(l*(l+1), 1, None)
    fl *= -1

    #if discretized:
    #    pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
    #    fl *= pw2/pw0

    #hp.almxfl(alm, fl, inplace=True)


    f_E = s2fft.inverse_jax(alm_E, L, reality=True, sampling="healpix", nside=nside)
    f_B = s2fft.inverse_jax(alm_B, L, reality=True, sampling="healpix", nside=nside)     

    return f_E, f_B


def from_convergence(kappa, lmax = None, *,
                     potential = False,
                     deflection = False,
                     shear = False,
                     discretized = True
                     ):
    r'''Compute other weak lensing maps from the convergence.

    Takes a weak lensing convergence map and returns one or more of
    deflection potential, deflection, and shear maps.  The maps are
    computed via spherical harmonic transforms.

    Parameters
    ----------
    kappa : array_like
        HEALPix map of the convergence field.
    lmax : int, optional
        Maximum angular mode number to use in the transform.
    potential, deflection, shear : bool, optional
        Which lensing maps to return.

    Returns
    -------
    psi : array_like
        Map of the deflection potential.  Only returned if ``potential``
        is true.
    alpha : array_like
        Map of the deflection (complex).  Only returned if ``deflection``
        if true.
    gamma : array_like
        Map of the shear (complex).  Only returned if ``shear`` is true.

    Notes
    -----
    The weak lensing fields are computed from the convergence or
    deflection potential in the following way. [1]_

    Define the spin-raising and spin-lowering operators of the
    spin-weighted spherical harmonics as

    .. math::

        \eth {}_sY_{lm}
        = +\sqrt{(l-s)(l+s+1)} \, {}_{s+1}Y_{lm} \;, \\
        \bar{\eth} {}_sY_{lm}
        = -\sqrt{(l+s)(l-s+1)} \, {}_{s-1}Y_{lm} \;.

    The convergence field :math:`\kappa` is related to the deflection
    potential field :math:`\psi` by the Poisson equation,

    .. math::

        2 \kappa
        = \eth\bar{\eth} \, \psi
        = \bar{\eth}\eth \, \psi \;.

    The convergence modes :math:`\kappa_{lm}` are hence related to the
    deflection potential modes :math:`\psi_{lm}` as

    .. math::

        2 \kappa_{lm}
        = -l \, (l+1) \, \psi_{lm} \;.

    The :term:`deflection` :math:`\alpha` is the gradient of the
    deflection potential :math:`\psi`.  On the sphere, this is

    .. math::

        \alpha
        = \eth \, \psi \;.

    The deflection field has spin weight :math:`1` in the HEALPix
    convention, in order for points to be deflected towards regions of
    positive convergence.  The modes :math:`\alpha_{lm}` of the
    deflection field are hence

    .. math::

        \alpha_{lm}
        = \sqrt{l \, (l+1)} \, \psi_{lm} \;.

    The shear field :math:`\gamma` is related to the deflection
    potential :math:`\psi` and deflection :math:`\alpha` as

    .. math::

        2 \gamma
        = \eth\eth \, \psi
        = \eth \, \alpha \;,

    and thus has spin weight :math:`2`.  The shear modes
    :math:`\gamma_{lm}` are related to the deflection potential modes as

    .. math::

        2 \gamma_{lm}
        = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \psi_{lm} \;.

    References
    ----------
    .. [1] Tessore N., et al., OJAp, 6, 11 (2023).
           doi:10.21105/astro.2302.01942

    '''

    # no output means no computation, return empty tuple
    if not (potential or deflection or shear):
        return ()

    # get the NSIDE parameter
    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3*nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # mode number; all conversions are factors of this
    l = np.arange(lmax+1)

    # this tuple will be returned
    results = ()

    # convert convergence to potential
    fl = np.divide(-2, l*(l+1), where=(l > 0), out=np.zeros(lmax+1))
    hp.almxfl(alm, fl, inplace=True)

    # if potential is requested, compute map and add to output
    if potential:
        psi = hp.alm2map(alm, nside, lmax=lmax)
        results += (psi,)

    # if no spin-weighted maps are requested, stop here
    if not (deflection or shear):
        return results

    # zero B-modes for spin-weighted maps
    blm = np.zeros_like(alm)

    # compute deflection alms in place
    fl = np.sqrt(l*(l+1))
    # TODO: missing spin-1 pixel window function here
    hp.almxfl(alm, fl, inplace=True)

    # if deflection is requested, compute spin-1 maps and add to output
    if deflection:
        alpha = hp.alm2map_spin([alm, blm], nside, 1, lmax)
        alpha = alpha[0] + 1j*alpha[1]
        results += (alpha,)

    # if no shear is requested, stop here
    if not shear:
        return results

    # compute shear alms in place
    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    fl = np.sqrt((l-1)*(l+2), where=(l > 0), out=np.zeros(lmax+1))
    fl /= 2
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    gamma = hp.alm2map_spin([alm, blm], nside, 2, lmax)
    gamma = gamma[0] + 1j*gamma[1]
    results += (gamma,)

    # all done
    return gamma
