from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import jax.numpy as jnp
import jax_cosmo as jc

Cosmology = jc.Cosmology
    
class MultiPlaneConvergence:
    '''Compute convergence fields iteratively from multiple matter planes.'''

    def __init__(self, cosmo: 'Cosmology') -> None:
        '''Create a new instance to iteratively compute the convergence.'''
        self.cosmo = cosmo

        # set up initial values of variables
        self.z2: float = 0.
        self.z3: float = 0.
        self.x3: float = 0.
        self.w3: float = 0.
        self.r23: float = 1.
        self.delta3: jnp.ndarray = jnp.array(0.)
        self.kappa2: Optional[jnp.ndarray] = None
        self.kappa3: Optional[jnp.ndarray] = None

    def add_window(self, delta: jnp.ndarray, w: 'RadialWindow') -> None:
        '''Add a mass plane from a window function to the convergence.

        The source plane redshift can be given using ``zsrc``.
        Otherwise, the mean redshift of the window is used.

        '''

        zsrc = w.zeff
        lens_weight = jnp.trapz(w.wa, w.za)/jnp.interp(zsrc, w.za, w.wa)

        self.add_plane(delta, zsrc, lens_weight)

    def add_plane(self, delta: jnp.ndarray, zsrc: float, wlens: float = 1.
                  ) -> None:
        '''Add a mass plane at redshift ``zsrc`` to the convergence.'''

        if zsrc <= self.z3:
            raise ValueError('source redshift must be increasing')

        # cycle mass plane, ...
        delta2, self.delta3 = self.delta3, delta

        # redshifts of source planes, ...
        z1, self.z2, self.z3 = self.z2, self.z3, zsrc

        # and weights of mass plane
        w2, self.w3 = self.w3, wlens

        # extrapolation law
        x2, self.x3 = self.x3, jc.background.transverse_comoving_distance(self.cosmo, jc.utils.z2a(self.z3))
        r12 = self.r23
        r13 = jnp.abs(jc.background.transverse_comoving_distance(self.cosmo, jc.utils.z2a(z1)) - self.x3)/self.x3
        self.r23 = jnp.abs(jc.background.transverse_comoving_distance(self.cosmo, jc.utils.z2a(self.z2)) - self.x3) /self.x3
        t = r13/r12

        # lensing weight of mass plane to be added
        f = 3*self.cosmo.Omega_m/2 
        f *= x2*self.r23 / jc.constants.rh
        f *= (1 + self.z2)/jnp.sqrt(jc.background.Esqr(self.cosmo, jc.utils.z2a(self.z2)))
        f *= w2

        # create kappa planes on first iteration
        if self.kappa2 is None:
            self.kappa2 = jnp.zeros_like(delta)
            self.kappa3 = jnp.zeros_like(delta)

        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        self.kappa2, self.kappa3 = self.kappa3, self.kappa2

        # compute next convergence plane in place of last
        self.kappa3 *= 1 - t
        self.kappa3 += t*self.kappa2
        self.kappa3 += f*delta2

    @property
    def zsrc(self) -> float:
        '''The redshift of the current convergence plane.'''
        return self.z3

    @property
    def kappa(self) -> Optional[jnp.ndarray]:
        '''The current convergence plane.'''
        return self.kappa3

    @property
    def delta(self) -> jnp.ndarray:
        '''The current matter plane.'''
        return self.delta3

    @property
    def wlens(self) -> float:
        '''The weight of the current matter plane.'''
        return self.w3
