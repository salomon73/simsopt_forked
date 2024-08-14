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
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty, SquaredRootPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, Energy
from simsopt.field.selffield import regularization_rect, B_regularized_pure, regularization_circ

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 5


FLUX_WEIGHT = Weight(1)
LENGTH_WEIGHT = Weight(1e-6)  #1e-5 id, 8e-7 quadratic, 3.5e-5 squaredroot
ENERGY_WEIGHT   = Weight(1e-12)  


# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 800

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

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

# Save initial coils and surface
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)


Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jenergy = []
for i in range(ncoils):
    coils_i = coils.copy()
    coils_i.pop(i)
    Jenergy.append(CoilEnergy(coils[i], coils_i, regularization_rect(0.1,0.1)))


JF = FLUX_WEIGHT* Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + ENERGY_WEIGHT * sum(Jenergy[i] for i in range(ncoils))


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    E_string = ", ".join(f"{J.J():1e}" for J in Jenergy)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", E = sum([{E_string}])={sum(J.J() for J in Jenergy):1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)


# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller B·n on the surface.
""" dofs = res.x
LENGTH_WEIGHT *= 0.1
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt_long")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_long", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
 """