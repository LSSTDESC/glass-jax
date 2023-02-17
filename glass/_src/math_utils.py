import jax.numpy as jnp
import jax.ops
import jax.lax

DEGREE2_SPHERE = 60**4//100/jnp.pi
ARCMIN2_SPHERE = 60**6//100/jnp.pi
ARCSEC2_SPHERE = 60**8//100/jnp.pi

def ndinterp(x, xp, fp, axis=-1, left=None, right=None, period=None):
    return jnp.apply_along_axis(jnp.interp, axis, fp, x, xp, left=left, right=right, period=period)

def trapz_product(f, *ff, axis=-1):
    x, _ = f
    for x_, _ in ff:
        x = jnp.union1d(x, x_[(x_ > x[0]) & (x_ < x[-1])])
    y = jnp.interp(x, *f)
    for f_ in ff:
        y *= jnp.interp(x, *f_)
    return jnp.trapz(y, x, axis=axis)

def cumtrapz(f, x, out=None):
    if out is None:
        out = jnp.empty_like(f)

    # CHANGE: Careful.
    out = jnp.cumsum((f[..., 1:] + f[..., :-1])/2*jnp.diff(x), axis=-1)
    out = out.at[..., 0].set(0)
    return out
