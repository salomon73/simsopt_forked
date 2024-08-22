import os, time, logging
from pathlib import Path
import numpy as np
import simsopt
import scipy.constants as constants
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty, SquaredRootPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, Energy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print
from simsopt.configs.zoo import get_w7x_data


# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.W7-X_standard_configuration'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# save forces on coils as points with magnitude of force
def pointData_forces(coils):
    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization_rect(0.015, 0.015)), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    point_data = {"F": forces}
    return point_data

# Initialize the boundary magnetic surface and the coils 
nphi = 128 #128
ntheta = 64 #64
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
nfp = s.nfp
curves, currents, ma = get_w7x_data()
coils = coils_via_symmetries(curves, currents, 5, True)
bs = BiotSavart(coils)
ncoils = len(coils)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

# Save initial coils and surface
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "W7X_curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "W7X_surf_init", extra_data=pointData)


print("Coils' currents: "+ f"{[current.get_value()*1e-6 for current in currents]}"+ "MA")
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i ], regularization_rect(0.015, 0.015)) for i in range(ncoils)] 
print("Total energy stored in W7X coils: " + f"{sum(Je.J()*1e-6/np.e/np.e for Je in Jenergy):2e}" + " MJ")    
curves_to_vtk(curves, OUT_DIR + f"W7X_curves_w_forces", extra_data=pointData_forces(coils))
