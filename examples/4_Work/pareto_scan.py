#!/usr/bin/env python


import os, time, logging, csv
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
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print

# Constants
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 6

# Fixed Weights
FLUX_WEIGHT = Weight(1)
ARCLENGTH_WEIGHT = Weight(1e-4)

# Scan ranges for weights
energy_weight_range = np.logspace(-13, -8, 100)  # Adjust the range as needed
length_weight_range = [0] #np.logspace(-7, -5, 3)   # Adjust the range as needed

# Number of iterations
MAXITER = 50 if in_github_actions else 800

# Bools
run_opt = True
compute_forces = True
make_p2 = False  # Set to False for scanning, could slow down the scan if True

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

# Scan over energy and length weights
results = []

for ew in energy_weight_range:
    for lw in length_weight_range:
        print(f"Running optimization for ENERGY_WEIGHT={ew}, LENGTH_WEIGHT={lw}")
        ENERGY_WEIGHT = Weight(ew)
        LENGTH_WEIGHT = Weight(lw)

        # Initialize the boundary magnetic surface
        nphi = 128
        ntheta = 64
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        nfp = s.nfp

        # Initialize the coils
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
        base_currents = [Current(1e4) for _ in range(ncoils)]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))

        # Save initial coils and surface
        curves = [c.curve for c in coils]

        # Define the objective function
        Jf = SquaredFlux(s, bs)
        Jls = [CurveLength(c) for c in base_curves]
        Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
        Ja = [ArclengthVariation(c) for c in base_curves]
        Jenergy = [CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j > i], regularization_rect(0.1, 0.1)) for i in range(ncoils)]
        Jccdist = CurveCurveDistance(curves, 0.0, num_basecurves=ncoils)
        Jcsdist = CurveSurfaceDistance(curves, s, 0.0)

        JF = FLUX_WEIGHT * Jf \
            + ARCLENGTH_WEIGHT * sum(Ja) \
            + LENGTH_WEIGHT * sum(Jls) \
            + ENERGY_WEIGHT * sum(Jenergy) 

        # Run optimization 
        dofs = JF.x
        res = minimize(fun, dofs, args=(JF, Jf, Jls, Jenergy, Ja), jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
        
        # Save results for Pareto analysis
        result = {
            'energy_weight': ew,
            'length_weight': lw,
            'total_energy': sum(J.J() for J in Jenergy),
            'total_length': sum(J.J() for J in Jls),
            'total_flux': Jf.J(),
            'BdotN':  np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))),
            'avg_B': np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
            'rel_B': np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))/np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
            'C-S-Sep': Jcsdist.shortest_distance(),
            'C-C-Sep': Jccdist.shortest_distance(),
            'max_force': np.max([np.max(np.linalg.norm(coil_force(c, coils, regularization_rect(0.1, 0.1)), axis=1)) for c in coils]),
            'arclength': sum(J.J() for J in Ja)
        }
        results.append(result)


# Generate Pareto front based on max_force, total_energy, and rel_B
pareto_front = []
for result in results:
    dominated = False
    for other in results:
        if (other['total_energy'] <= result['total_energy'] and
            other['max_force'] <= result['max_force'] and
            other['rel_B'] <= result['rel_B'] and
            other['arclength'] <= result['arclength'] and
            (other['total_energy'] < result['total_energy'] or
             other['max_force'] < result['max_force'] or
             other['rel_B'] < result['rel_B'] or
             other['arclength'] < result['arclength'])):
            dominated = True
            break
    if not dominated:
        pareto_front.append(result)

# Print the number of Pareto front points found
print(f"Pareto front found with {len(pareto_front)} points.")

# Save Pareto front in CSV file
res_file = OUT_DIR + "results.csv"
pareto_file = OUT_DIR + "pareto_front.csv"
with open(pareto_file, 'w', newline='') as csvfile:
    fieldnames = ['energy_weight', 'length_weight', 'total_energy', 'total_length', 'total_flux', 'BdotN', 'avg_B', 'rel_B', 'C-S-Sep', 'C-C-Sep', 'max_force', 'arclength']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in pareto_front:
        writer.writerow(result)
csvfile.close()
print(f"Pareto front saved to {pareto_file}")

with open(res_file, 'w', newline='') as csvfile:
    fieldnames = ['energy_weight', 'length_weight', 'total_energy', 'total_length', 'total_flux', 'BdotN', 'avg_B', 'rel_B', 'C-S-Sep', 'C-C-Sep', 'max_force', 'arclength']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)
csvfile.close()
print(f"All scan results saved to {res_file}")