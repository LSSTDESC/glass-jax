import jax.numpy as jnp
import jax_cosmo as jc


def matter_cls(cosmo, lmax, zs, ws):
    """ Uses jax-cosmo to compute matter angular cls for the 
    given redshift bins.
    """
    #TODO: Replace this by proper top hat window function
    nzs = [jc.redshift.kde_nz(z, w, bw=(z[1]-z[0]), zmax=jnp.max(z)) for z, w in zip(zs, ws)]
    probes = [jc.probes.NumberCounts(nzs, jc.bias.constant_linear_bias(1.))]
    ell = jnp.arange(lmax+1)
    return jc.angular_cl.angular_cl(cosmo, ell, probes)
