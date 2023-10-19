import jax.numpy as jnp
import jax_cosmo as jc


def matter_cls(cosmo, lmax, win):
    """ Uses jax-cosmo to compute matter angular cls for the 
    given redshift bins.
    """
    #TODO: Replace this by proper top hat window function
    nzs = [jc.redshift.kde_nz(w.za, w.wa, bw=(w.za[1]-w.za[0]), zmax=w.za.max()) for w in win]
    ell = jnp.arange(lmax)
    #CAVEAT: we only compute the autopowerspectra because we don't have the non-limber integral necessary for the cross anyway.
    return [jc.angular_cl.angular_cl(cosmo, ell, [jc.probes.NumberCounts([nz], jc.bias.constant_linear_bias(1.))]) for nz in nzs]
