import jax.numpy as jnp
from typing import Optional
from jax import random


def triaxial_axis_ratio(zeta, xi, size=None, *, rng=None):
    # default RNG if not provided
    if rng is None:
        rng = random.PRNGKey(0)

    # get size from inputs if not explicitly provided
    if size is None:
        size = jnp.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    cos2_theta = random.uniform(rng, shape=size, minval=-1., maxval=1.)**2
    sin2_theta = 1 - cos2_theta
    cos2_phi = jnp.cos(random.uniform(rng, shape=size, minval=0., maxval=2*jnp.pi))**2
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = jnp.square(zeta)
    z2m1 -= 1
    x2 = jnp.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1*sin2_phi)*cos2_theta + x2*sin2_theta
    B2 = 4*z2m1**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 + z2m1*cos2_phi

    # eq. (12)
    q = jnp.sqrt((A+C-jnp.sqrt((A-C)**2+B2))/(A+C+jnp.sqrt((A-C)**2+B2)))

    return q


def ellipticity_ryden04(mu, sigma, gamma, sigma_gamma, size=None, *, rng=None):
    # default RNG if not provided
    if rng is None:
        rng = random.PRNGKey(0)

    # get size from inputs if not explicitly provided
    if size is None:
        size = (mu.shape if jnp.shape(mu) == jnp.shape(sigma) == jnp.shape(gamma) == jnp.shape(sigma_gamma) else ())

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    # first sample unbounded normal, then rejection sample truncation
    eps = norm.rvs(rng, mu, sigma, size=size)
    bad = (eps > 0)
    while jnp.any(bad):
        eps = norm.rvs(rng, mu, sigma, size=eps[bad].shape)
        bad = (eps > 0)
    gam = norm.rvs(rng, gamma, sigma_gamma, size=size)
    bad = (gam < 0) | (gam > 1)
    while jnp.any(bad):
        gam = norm.rvs(rng, gamma, sigma_gamma, size=gam[bad].shape)
        bad = (gam < 0) | (gam > 1)

    # compute triaxial axis ratios zeta = B/A, xi = C/A
    zeta = -jnp.expm1(eps)
    xi = (1 - gam) * zeta

    # random projection of random triaxial ellipsoid
    q = triaxial_axis_ratio(zeta, xi, size=size, rng=rng)

    # assemble ellipticity with random complex phase
    e = jnp.exp(1j * random.uniform(rng, shape=size) * 2 * jnp.pi)
    e *= (1 - q) / (1 + q)

    # return the ellipticity
    return e


def ellipticity_gaussian(size, sigma, *, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # sample complex ellipticities
    # reject those where abs(e) > 0
    e = jax.random.normal(rng, shape=(size, 2)).astype(jnp.complex128)
    e *= sigma
    i = jnp.where(jnp.abs(e) > 1)[0]
    while len(i) > 0:
        e = jax.ops.index_update(e, i, jax.random.normal(rng, shape=(len(i), 2)).astype(jnp.complex128))
        e *= sigma
        i = i[jnp.abs(e[i]) > 1]

    return e


def ellipticity_intnorm(size, sigma, *, rng=None):
    if not 0 <= sigma < 0.5**0.5:
        raise ValueError('sigma must be between 0 and sqrt(0.5)')

    # default RNG if not provided
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # convert to sigma_eta using fit
    sigma_eta = sigma*((8 + 5*sigma**2)/(2 - 4*sigma**2))**0.5

    # sample complex ellipticities
    e = jax.random.normal(rng, (size, 2), dtype=jnp.float64).view(jnp.complex128)
    e *= sigma_eta
    r = jnp.hypot(e.real, e.imag)
    e = jnp.where(r > 0, e * (jnp.tanh(r/2) / r), e)

    return e