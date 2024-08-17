from deprecated import deprecated

import numpy as np
import jax
from jax import grad, vmap
import jax.numpy as jnp
from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec
from ..field.selffield import self_ind_accurate, mutual_inductance, mutual_inductance_vec, self_ind, self_ind_vec
import simsoptpp as sopp
from scipy import constants

__all__ = ['Energy', 'CoilEnergy' ]
# This function is not jitted due to slow compilation times from the two levels of forloop
def energy_pure(array_gamma, array_gammadash, array_current, array_quadpoints, regularization):
    r"""
    This function computes the energy as the sum over the self and mutual inductances of a set of coils
    whose curves are provided in the arrays passed as args.
    """
    E = jnp.zeros(1)
    n = array_gamma.shape[0]
    for i in range(n):
        E += array_current[i]**2 * self_ind_vec(array_gamma[i,:,:], array_gammadash[i,:,:], array_quadpoints[i,:], regularization)
        g_i = jnp.delete(array_gamma, i, axis = 0) 
        gdash_i = jnp.delete(array_gammadash, i, axis = 0)
        I_i = jnp.delete(array_current, i)
        for j in range(n-1):
            E += array_current[i]*I_i[j] * mutual_inductance_vec(array_gamma[i,:,:], array_gammadash[i,:,:], g_i[j], gdash_i[j])
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



def coil_energy_pure_vec(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization):
    """
    Computes the energy contribution of a given coil: 
    self-energy + mutual energies terms, optimized version.
    """
    self_energy =  I**2 * self_ind_vec(gamma, gammadash, quadpoints, regularization)

    mutual_energies = 0.0
    if len(array_g) > 0:
        mutual_energies = jnp.sum(I * array_current * vmap(mutual_inductance_vec, in_axes=(None, None, 0, 0))(gamma, gammadash, array_g, array_gdash))
        
    E = 0.5 * jnp.sum(self_energy + mutual_energies)
    return E


class CoilEnergy(Optimizable):
    "Class that handles the energy of one coil surrounded by a set of external coils"
    def __init__(self, coil, other_coils, regularization):

        self.coil = coil
        self.other_coils = other_coils 
        self.regularization = regularization

        self.J_Jax = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
            coil_energy_pure_vec(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
    
        # Gradients with respect to the considered coil for which it is intended to compute the energy
        self.thisgrad0 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
                        grad(self.J_Jax, argnums=0)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad1 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
                        grad(self.J_Jax, argnums=1)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad2 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
                        grad(self.J_Jax, argnums=2)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        
        # Gradients with respect to the other coils' parameters
        self.thisgrad4 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization: 
            grad(self.J_Jax, argnums=4)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad5 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization: 
            grad(self.J_Jax, argnums=5)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad6 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization: 
            grad(self.J_Jax, argnums=6)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))

        super().__init__(depends_on = [coil, *other_coils])

    def J(self):

        # coil parameters
        I =         jnp.array(self.coil.current.get_value())
        gamma =     jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        # external coils parameters
        array_g =       jnp.array([jnp.array(c.curve.gamma()) for c in self.other_coils])
        array_gdash =   jnp.array([jnp.array(c.curve.gammadash()) for c in self.other_coils])
        array_current = jnp.array([jnp.array(c.current.get_value()) for c in self.other_coils])
        regularization = jnp.array(self.regularization)
        
        return self.J_Jax(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
    
    @derivative_dec
    def dJ(self):

        # coil parameters
        I =         jnp.array(self.coil.current.get_value())
        gamma =     jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        # external coils parameters
        array_g =       jnp.array([jnp.array(c.curve.gamma()) for c in self.other_coils])
        array_gdash =   jnp.array([jnp.array(c.curve.gammadash()) for c in self.other_coils])
        array_current = jnp.array([jnp.array(c.current.get_value()) for c in self.other_coils])
        regularization = jnp.array(self.regularization)
        
        thisgrad0 = self.thisgrad0(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad1 = self.thisgrad1(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad2 = self.thisgrad2(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad4 = self.thisgrad4(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad5 = self.thisgrad5(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad6 = self.thisgrad6(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)    

        # Gradient with respect to the DOFS of the 'self' coil
        primary_grad =  self.coil.curve.dgamma_by_dcoeff_vjp(thisgrad1) \
            + self.coil.curve.dgammadash_by_dcoeff_vjp(thisgrad2) \
            + self.coil.current.vjp(jnp.asarray([thisgrad0]))

        # Gradients with respect to the DOFS of the external coil
        other_coils_grad = Derivative({})
        for i,c  in enumerate(self.other_coils):
            other_coils_grad += (
                c.curve.dgamma_by_dcoeff_vjp(thisgrad4[i,:,:]) +
                c.curve.dgammadash_by_dcoeff_vjp(thisgrad5[i,:,:]) +
                c.current.vjp(jnp.asarray([thisgrad6[i]]))
            ) 

        return primary_grad + other_coils_grad

    return_fn_map = {'J': J, 'dJ': dJ}