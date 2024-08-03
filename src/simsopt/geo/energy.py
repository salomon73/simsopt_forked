from deprecated import deprecated

import numpy as np
from jax import grad
import jax.numpy as jnp
from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec
from ..field.selffield import self_ind_accurate, mutual_inductance, self_ind
import simsoptpp as sopp
from scipy import constants


# This function is not jitted due to slow compilation times from the two levels of forloop
def energy_pure(array_gamma, array_gammadash, array_current, array_quadpoints, regularization):
    r"""
    This function computes the energy as the sum over the self and mutual inductances of a set of coils
    whose curves are provided in the arrays passed as args.
    """
    E = jnp.zeros(1)
    n = array_gamma.shape[0]
    for i in range(n):
        E += array_current[i]**2 * self_ind(array_gamma[i,:,:], array_gammadash[i,:,:], array_quadpoints[i,:], regularization)
        g_i = jnp.delete(array_gamma, i, axis = 0) 
        gdash_i = jnp.delete(array_gammadash, i, axis = 0)
        I_i = jnp.delete(array_current, i)
        for j in range(n-1):
            E += array_current[i]*I_i[j] * mutual_inductance(array_gamma[i,:,:], array_gammadash[i,:,:], g_i[j], gdash_i[j])
    return jnp.array(0.5 * E)[0]


# Self energy targets only the self-inductance terms from the energy. Runs faster. 
def self_energy(array_gamma, array_gammadash, array_current, array_quadpoints, regularization):
    r"""
    This function computes the energy as the sum over the dominant terms (self inductances) 
    for a given set of coils
    """
    E = jnp.zeros(1)
    for i in range(array_gamma.shape[0]):
        gamma1 = array_gamma[i,:,:]
        gammadash1 = array_gammadash[i,:,:]
        quadpoints1 = array_quadpoints[i,:]
        I1 = array_current[i]
        E += I1**2 * self_ind(gamma1, gammadash1, quadpoints1, regularization)
    
    return jnp.array(0.5 * E)[0]


class Energy(Optimizable):
    "Class that handles the vacuum-field energy"
    def __init__(self, coils, ncoils, regularization):
        self.coils = coils
        self.ncoils = ncoils 
        self.regularization = regularization

        self.J_Jax = lambda array_gamma, array_gammadash, array_current, array_quadpoints, regularization : \
            energy_pure(array_gamma, array_gammadash, array_current, array_quadpoints, regularization)
        
        self.thisgrad = [lambda array_gamma, array_gammadash, array_current, array_quadpoints, regularization: \
            grad(self.J_Jax, argnums=i)(array_gamma, array_gammadash, array_current, array_quadpoints, regularization) for i in range(3)]
        
        super().__init__( depends_on = coils)

    def J(self):
        gamma_array = jnp.array([coil.curve.gamma() for coil in self.coils])
        gammadash_array = jnp.array([coil.curve.gammadash() for coil in self.coils])
        current_array = jnp.array([coil.current.get_value() for coil in self.coils])
        quadpoints_array = jnp.array([coil.curve.quadpoints for coil in self.coils])
        regularization = self.regularization
        
        return self.J_Jax(gamma_array, gammadash_array, current_array, quadpoints_array, regularization)
    
    @derivative_dec
    def dJ(self):
        gamma_array = jnp.array([coil.curve.gamma() for coil in self.coils])
        gammadash_array = jnp.array([coil.curve.gammadash() for coil in self.coils])
        current_array = jnp.array([coil.current.get_value() for coil in self.coils])
        quadpoints_array = jnp.array([coil.curve.quadpoints for coil in self.coils])
        regularization = self.regularization
        thisgrad = self.thisgrad

        res = [ self.coils[i].curve.dgamma_by_dcoeff_vjp(thisgrad[0](gamma_array, gammadash_array, current_array, quadpoints_array, regularization)[i])  
              + self.coils[i].curve.dgammadash_by_dcoeff_vjp(thisgrad[1](gamma_array, gammadash_array, current_array, quadpoints_array, regularization)[i])
              + self.coils[i].current.vjp(thisgrad[2](gamma_array, gammadash_array, current_array, quadpoints_array, regularization)[i]) 
                for i in range(self.ncoils) ]
        return sum(res)
    
    return_fn_map = {'J': J, 'dJ': dJ}




def coil_energy_pure(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization):
    r"""
    This function computes the energy contribution of a given coil:
    self energy + mutual energies terms
    """
    E = jnp.zeros(1)
    # self energy term
    E += I**2 * self_ind(gamma, gammadash, quadpoints, regularization)
    # mutual energy terms
    for i in range(array_g.shape[0]): 
        E += I*array_current[i] * mutual_inductance(gamma, gammadash, array_g[i,:,:], array_gdash[i,:,:])
    return jnp.array(0.5 * E)[0]


class CoilEnergy(Optimizable):
    "Class that handles the energy of one coil surrounded by a set of external coils"
    def __init__(self, coil, coils, regularization):
        self.coil = coil
        self.coils = coils 
        self.regularization = regularization

        self.J_Jax = lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
            coil_energy_pure(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
    
        self.thisgrad = [lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
            grad(self.J_Jax, argnums=i)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization) for i in range(3)]
        
        super().__init__(depends_on = [coil])

    def J(self):

        # coil parameters
        I = self.coil.current.get_value()
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        quadpoints = self.coil.curve.quadpoints

        # external coils parameters
        array_g = jnp.array([c.curve.gamma() for c in self.coils])
        array_gdash = jnp.array([c.curve.gammadash() for c in self.coils])
        array_current = jnp.array([c.current.get_value() for c in self.coils])
        regularization = self.regularization

        return self.J_Jax(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
    
    @derivative_dec
    def dJ(self):
        thisgrad = self.thisgrad

        # coil parameters
        I = self.coil.current.get_value()
        gamma = self.coil.curve.gamma()
        gammadash = self.coil.curve.gammadash()
        quadpoints = self.coil.curve.quadpoints

        # external coils parameters
        array_g = jnp.array([c.curve.gamma() for c in self.coils])
        array_gdash = jnp.array([c.curve.gammadash() for c in self.coils])
        array_current = jnp.array([c.current.get_value() for c in self.coils])
        regularization = self.regularization

        return self.coil.curve.dgamma_by_dcoeff_vjp(thisgrad[1](I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)) \
            + self.coil.curve.dgammadash_by_dcoeff_vjp(thisgrad[2](I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)) 
        
        # This issue needs to be fixed but at this time the current is kept constant
        #    + self.coil.current.vjp(thisgrad[0](I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)) 

            
    return_fn_map = {'J': J, 'dJ': dJ}