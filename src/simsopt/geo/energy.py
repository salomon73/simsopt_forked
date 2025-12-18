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

__all__ = ['CoilEnergy','SelfEnergy', 'MutualEnergy' ]


def coil_energy_pure(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization):
    """
    Computes the energy contribution of a given coil: 
    self-energy + mutual energies terms, vectorized version.
    """
    self_energy =  I**2 * self_ind_vec(gamma, gammadash, quadpoints, regularization)

    mutual_energies = 0.0
    if len(array_g) > 0:
        mutual_energies = jnp.sum(I * array_current * vmap(mutual_inductance_vec, in_axes=(None, None, 0, 0))(gamma, gammadash, array_g, array_gdash))
        
    E = 0.5 * jnp.sum(self_energy) + mutual_energies # mutual calculated only on unique pairs (hence no 1/2)
    return E


class CoilEnergy(Optimizable):
    "Class that handles the energy of one coil surrounded by a set of external coils"
    def __init__(self, coil, other_coils, regularization):

        self.coil = coil
        self.other_coils = other_coils 
        self.regularization = regularization

        self.J_Jax = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization : \
            coil_energy_pure(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
    
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

        # Gradients with respect to the DOFS of the external coils
        other_coils_grad = Derivative({})
        for i,c  in enumerate(self.other_coils):
            other_coils_grad += (
                c.curve.dgamma_by_dcoeff_vjp(thisgrad4[i,:,:]) +
                c.curve.dgammadash_by_dcoeff_vjp(thisgrad5[i,:,:]) +
                c.current.vjp(jnp.asarray([thisgrad6[i]]))
            ) 

        return primary_grad + other_coils_grad

    return_fn_map = {'J': J, 'dJ': dJ}

# The classes below provide a separation of the self energy and the mutual energy terms to study their influence
# on the optimized sets of coils. 

def self_energy_pure(I, gamma, gammadash, quadpoints, regularization):
    """
    Compute the self-energy contribution for a single coil (vectorized).
    This matches the per-coil self-term in original `coil_energy_pure`
    which used 0.5*(I**2 * L).
    """
    L = self_ind_vec(gamma, gammadash, quadpoints, regularization)
    Eself = 0.5 * jnp.sum(I**2 * L)
    return Eself


def mutual_energy_pure(I, gamma, gammadash, quadpoints,
                       array_g, array_gdash, array_current):
    """
    Compute the mutual-energy contribution for a single primary coil
    interacting with a list/array of external coils (vectorized).
    This matches the per-coil mutual-term in original `coil_energy_pure`
    which used 0.5 * sum(I * I_j * M_ij).
    Note: array_* may be empty.
    Note: There is no factor 1/2 because of the convention chosen in computing the energy
          from a list of coils with unique pairs.
    """
    if len(array_g) == 0:
        return 0.0
    Mvec = vmap(mutual_inductance_vec, in_axes=(None, None, 0, 0))(gamma, gammadash, array_g, array_gdash)
    Emut = jnp.sum(I * array_current * Mvec)
    return Emut


class SelfEnergy(Optimizable):
    """
    Self-energy functional for a single coil:
    E_self = 0.5 * I^2 * L(gamma)
    """
    def __init__(self, coil, regularization):
        self.coil = coil
        self.regularization = regularization

        # JAX-compiled scalar functional (arguments kept same shape/signature as other classes)
        self.J_Jax = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                         self_energy_pure(I, gamma, gammadash, quadpoints, regularization))

        # Gradients w.r.t. primary coil params (argnums 0,1,2)
        self.thisgrad0 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=0)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad1 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=1)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad2 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=2)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))

        # No other_coils for SelfEnergy, so we don't define argnums gradients for 4,5,6.
        super().__init__(depends_on=[coil])

    def J(self):
        I = jnp.array(self.coil.current.get_value())
        gamma = jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        # keep same placeholders for compatibility with the J_Jax signature
        array_g = jnp.array([])            # unused
        array_gdash = jnp.array([])        # unused
        array_current = jnp.array([])      # unused
        regularization = jnp.array(self.regularization)

        return self.J_Jax(I, gamma, gammadash, quadpoints,
                          array_g, array_gdash, array_current, regularization)

    @derivative_dec
    def dJ(self):
        I = jnp.array(self.coil.current.get_value())
        gamma = jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        array_g = jnp.array([])
        array_gdash = jnp.array([])
        array_current = jnp.array([])
        regularization = jnp.array(self.regularization)

        thisgrad0 = self.thisgrad0(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad1 = self.thisgrad1(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad2 = self.thisgrad2(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)

        # Gradient with respect to the DOFs of the 'self' coil
        primary_grad = (self.coil.curve.dgamma_by_dcoeff_vjp(thisgrad1) +
                        self.coil.curve.dgammadash_by_dcoeff_vjp(thisgrad2) +
                        self.coil.current.vjp(jnp.asarray([thisgrad0])))

        return primary_grad

    return_fn_map = {'J': J, 'dJ': dJ}


class MutualEnergy(Optimizable):
    """
    Mutual-energy functional for a single primary coil interacting with a list of external coils.
    For consistency with original implementation, the returned value is:
      E_mut_per_primary = sum_j ( I_i * I_j * M_ij )
    where `other_coils` should be the list of coils with index > i (so each mutual pair is counted once).
    Note: No factor 1/2 because the mutual contribution is computed from a list of unique pairs. 
    """
    def __init__(self, coil, other_coils, regularization):
        self.coil = coil
        self.other_coils = other_coils
        self.regularization = regularization

        # JAX-compiled mutual functional (note regularization kept in signature for compatibility)
        self.J_Jax = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                         mutual_energy_pure(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current))

        # Gradients w.r.t. considered coil
        self.thisgrad0 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=0)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad1 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=1)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad2 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=2)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))

        # Gradients w.r.t. other coils' parameters (these exist because mutual depends on array_g, array_gdash, array_current)
        self.thisgrad4 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=4)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad5 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=5)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))
        self.thisgrad6 = jit(lambda I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization:
                              grad(self.J_Jax, argnums=6)(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization))

        super().__init__(depends_on=[coil, *other_coils])

    def J(self):
        # primary coil params
        I = jnp.array(self.coil.current.get_value())
        gamma = jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        # external coils params
        array_g = jnp.array([jnp.array(c.curve.gamma()) for c in self.other_coils])
        array_gdash = jnp.array([jnp.array(c.curve.gammadash()) for c in self.other_coils])
        array_current = jnp.array([jnp.array(c.current.get_value()) for c in self.other_coils])
        regularization = jnp.array(self.regularization)

        return self.J_Jax(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)

    @derivative_dec
    def dJ(self):
        # primary coil params
        I = jnp.array(self.coil.current.get_value())
        gamma = jnp.array(self.coil.curve.gamma())
        gammadash = jnp.array(self.coil.curve.gammadash())
        quadpoints = jnp.array(self.coil.curve.quadpoints)

        # external coils params
        array_g = jnp.array([jnp.array(c.curve.gamma()) for c in self.other_coils])
        array_gdash = jnp.array([jnp.array(c.curve.gammadash()) for c in self.other_coils])
        array_current = jnp.array([jnp.array(c.current.get_value()) for c in self.other_coils])
        regularization = jnp.array(self.regularization)

        thisgrad0 = self.thisgrad0(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad1 = self.thisgrad1(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad2 = self.thisgrad2(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad4 = self.thisgrad4(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad5 = self.thisgrad5(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)
        thisgrad6 = self.thisgrad6(I, gamma, gammadash, quadpoints, array_g, array_gdash, array_current, regularization)

        # Gradient with respect to the DOFs of the self coil
        primary_grad = (self.coil.curve.dgamma_by_dcoeff_vjp(thisgrad1) +
                        self.coil.curve.dgammadash_by_dcoeff_vjp(thisgrad2) +
                        self.coil.current.vjp(jnp.asarray([thisgrad0])))

        # Gradients with respect to the DOFs of the external coils
        other_coils_grad = Derivative({})
        for i, c in enumerate(self.other_coils):
            other_coils_grad += (
                c.curve.dgamma_by_dcoeff_vjp(thisgrad4[i, :, :]) +
                c.curve.dgammadash_by_dcoeff_vjp(thisgrad5[i, :, :]) +
                c.current.vjp(jnp.asarray([thisgrad6[i]]))
            )

        return primary_grad + other_coils_grad

    return_fn_map = {'J': J, 'dJ': dJ}
