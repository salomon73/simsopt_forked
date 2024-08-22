#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + ENERGY_WEIGHT * (sum CoilEnergy)

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty, SquaredRootPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, Energy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, B_regularized, regularization_circ, regularization_rect

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 8


# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)


# Initialize the boundary magnetic surface:
nphi = 128
ntheta = 64
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Initialize the coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e4) for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))


# Initialize the functional to test 
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i ], regularization_rect(0.1, 0.1)) for i in range(ncoils) ] 
JF =  sum(Jenergy)

# Wrapper function for scipy.minimize
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    return J, grad

# Parameters for the Taylor test
verbose = False
n_tests=40

print("""
################################################################################
################           PERFORMING A TAYLOR TEST            #################
################################################################################
""")
print("Average computed over")
print(n_tests, " tests")
f = fun
dofs = JF.x

np.random.seed(1)
eps_vec = np.flip(np.logspace(-12,-3,40))
err_vec = np.zeros((n_tests, 40))
for i in range(n_tests):
    h = np.random.uniform(size=dofs.shape)
    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    for j in range(len(eps_vec)):
        J1, _ = f(dofs + eps_vec[j]*h)
        J2, _ = f(dofs - eps_vec[j]*h)
        if verbose:
            print('diff ', (J1-J2)/(2*eps_vec[j]))
            print('autodiff', dJh)
            print('------------')
            print("err", (J1-J2)/(2*eps_vec[j]) - dJh)
            print("------------")
            print('J1', J1)
            print('J2', J2)
        err_vec[i,j]= np.abs((J1-J2)/(2*eps_vec[j]) - dJh)/np.abs(dJh)
avg_err = np.sum(err_vec, axis=0)

# Plot the Taylor test result
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
ax.tick_params(axis='y')
ax.loglog(eps_vec, avg_err)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.set_xlabel('$\epsilon$',fontsize=20)
ax.set_ylabel(r'$\frac{|(J(\mathbf{x}+\epsilon\mathbf{h})-J(\mathbf{x}-\epsilon\mathbf{h}))|}{|2\epsilon dJ|}$', fontsize=20)
plt.savefig('Taylortest.pdf', dpi=300)
plt.show()
