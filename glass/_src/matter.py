import jax.numpy as jnp
import jax_cosmo as jc


def matter_cls(cosmo, lmax, zs, ws):
    """ Uses jax-cosmo to compute matter angular cls for the 
    given redshift bins.
    """
    #TODO: Replace this by proper top hat window function
    nzs = [jc.redshift.kde_nz(z, w, bw=(z[1]-z[0]), zmax=jnp.max(z)) for z, w in zip(zs, ws)]
    ell = jnp.arange(lmax)
    #CAVEAT: we only compute the autopowerspectra because we don't have the non-limber integral necessary for the cross anyway.
    return [jc.angular_cl.angular_cl(cosmo, ell, [jc.probes.NumberCounts([nz], jc.bias.constant_linear_bias(1.))]) for nz in nzs]
