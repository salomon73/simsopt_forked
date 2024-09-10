from .._core.optimizable import Optimizable
from .biotsavart import BiotSavart
from .._core.derivative import derivative_dec, Derivative
from jax import grad, vmap
import jax.numpy as jnp
from ..geo.jit import jit
from ..geo.curve import incremental_arclength_pure

def action_pure(gamma, gammadash,  bs):
    bs.set_point(gamma)
    return 1 # jnp.mean(jnp.dot(bs.A().reshape(-1,3), incremental_arclength_pure(gammadash)))

class MagneticAction(Optimizable):
    def __init__(self, curve, bs):
        self.curve = curve
        self.gamma = curve.gamma()
        self.gammadash = curve.gammadash()
        self.bs = bs


        self.J_jax = jit(lambda gamma, gammadash, bs: action_pure(gamma, gammadash, bs))
        self.thisgrad0 = jit(lambda gamma, gammadash, bs: grad(self.J_Jax,argnums=0)(gamma, gammadash,  bs))
        self.thisgrad1 = jit(lambda gamma, gammadash, bs: grad(self.J_Jax,argnums=1)(gamma, gammadash,  bs))

        super().__init__( depends_on = [curve])

    def J(self):
        return self.J_jax(self.gamma, self.gammadash, self.bs)
    
    @derivative_dec
    def dJ(self):
        thisgrad0 = self.thsigrad0(self.gamma, self.gammadash,  self.bs)
        thisgrad1 = self.thsigrad1(self.gamma, self.gammadash,  self.bs)
        return (self.curve.dgamma_by_dcoeff_vjp(thisgrad0) +
                self.curve.dgammadash_by_dcoeff_vjp(thisgrad1))
    
    
    class IotaFromAction:
        def __init__(self, action):
            self.action = action
