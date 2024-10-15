#!/usr/bin/env python


import os, csv
from pathlib import Path
import numpy as np
import simsopt
import scipy as scp
from scipy.optimize import minimize
from simsopt.field import ( BiotSavart, Current, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation, CurveCurveDistance, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty, SquaredRootPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, Energy
from simsopt.field.force import coil_force_pure, coil_force, LpCurveForce
from simsopt.field.selffield import regularization_rect
from simsopt.util import in_github_actions

# Constants
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 6

# Fixed Weights
FLUX_WEIGHT = Weight(1)
ARCLENGTH_WEIGHT = Weight(1e-4)

# Scan ranges for weights + regularization
energy_weight_range = np.logspace(-13, -9, 40)  
length_weight_range = [0] 
regularization = [1e-3, 1e-2, 1e-1]

# Number of iterations
MAXITER = 50 if in_github_actions else 800

# Bools
run_opt = True
compute_forces = True
make_p2 = False  

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output_scan/"
os.makedirs(OUT_DIR, exist_ok=True)

# Define optimization function
def fun(dofs, JF, Jf, Jls, Jenergy, Ja):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    E_string = ", ".join(f"{J.J():1e}" for J in Jenergy)
    arcString = ", ".join(f"{J.J():1e}" for J in Ja)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", E = sum([{E_string}])={sum(J.J() for J in Jenergy):1e}"
    outstr += f", arc = [{arcString}]"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

# Scan over energy regularization
results = []

for ew in energy_weight_range:
    for reg in regularization:
        print(f"Running optimization for ENERGY_WEIGHT={ew}, REGULARIZATION={reg}")
        ENERGY_WEIGHT = Weight(ew)
        LENGTH_WEIGHT = Weight(0.0)
        # Initialize the boundary magnetic surface
        nphi = 128
        ntheta = 64
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        nfp = s.nfp

        # Initialize the coils
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
        base_currents = [Current(1e5) for _ in range(ncoils)]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))

        # Save initial coils and surface
        curves = [c.curve for c in coils]

        # Define the objective function
        Jf =     SquaredFlux(s, bs)
        Jls =   [CurveLength(c) for c in base_curves]
        Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
        Ja =    [ArclengthVariation(c) for c in base_curves]
        Jenergy = [CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j > i], regularization_rect(reg, reg)) for i in range(ncoils)]
        Jccdist = CurveCurveDistance(curves, 0.0, num_basecurves=ncoils)
        Jcsdist = CurveSurfaceDistance(curves, s, 0.0)

        JF = FLUX_WEIGHT * Jf \
            + ARCLENGTH_WEIGHT * sum(Ja) \
            + LENGTH_WEIGHT * sum(Jls) \
            + ENERGY_WEIGHT * sum(Jenergy) 

        # Run optimization 
        dofs = JF.x
        res = minimize(fun, dofs, args=(JF, Jf, Jls, Jenergy, Ja), jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
        
        # Save results 
        result = {
            'energy_weight': ew,
            'length_weight': 0.0,
            'reg': reg,
            'total_energy': sum(J.J() for J in Jenergy),
            'total_length': sum(J.J() for J in Jls),
            'total_flux': Jf.J(),
            'BdotN': np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))),
            'avg_B': np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
            'rel_B': np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))) / np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
            'C-S-Sep': Jcsdist.shortest_distance(),
            'C-C-Sep': Jccdist.shortest_distance(),
            'max_force': [np.max(np.linalg.norm(coil_force(c, coils, regularization_rect(reg, reg)), axis=1)) for c in coils[0:ncoils]],
            'int_force': [1/CurveLength(c.curve).J() \
                           * np.mean(np.linalg.norm(coil_force(c, coils, regularization_rect(reg, reg)), axis=1) \
                           * np.linalg.norm(c.curve.gammadash(), axis=1)) for c in coils[0:ncoils]],
            'arclength': sum(J.J() for J in Ja)
        }
        results.append(result)

# Save Pareto front in CSV file
res_file = OUT_DIR + "results_scan_reg.csv"

with open(res_file, 'w', newline='') as csvfile:
    fieldnames = ['energy_weight', 'length_weight', 'reg', 'total_energy', 'total_length', 'total_flux', 'BdotN', 'avg_B', 'rel_B', 'C-S-Sep', 'C-C-Sep', 'max_force', 'int_force', 'arclength']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)
csvfile.close()
print(f"All scan results saved to {res_file}")