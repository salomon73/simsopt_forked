#!/usr/bin/env python

import os
import csv
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves, CurveLength,
    MeanSquaredCurvature, ArclengthVariation, CurveCurveDistance,
    CurveSurfaceDistance, LinkingNumber
)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy
from simsopt.field.force import coil_force
from simsopt.field.selffield import regularization_rect

# Constants
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 6
reg = regularization_rect(0.015, 0.015)

# Scan ranges for weights
energy_weight_range = np.logspace(-13, -8, 50)  # Adjust the range as needed
length_weight_range = np.append(0,np.logpsace(-10,-4, 12))  # Adjust the range as needed
cs_weight_range = [0, 10]  # Range for CurveSurfaceDistance
cc_weight_range = [0, 1000]  # Range for CurveCurveDistance
link_weight_range = [0] # Range for LinkingNumber
flux_weight_range = [1, 0.5]
arclength_weight_range = [1e-7, 1e-4]  # Range for ArclengthVariation

# Number of iterations
MAXITER = 50 if in_github_actions else 800

# Thresholds
CS_THRESHOLD = 0.0  # Define appropriate thresholds
CC_THRESHOLD = 0.0
LINK_THRESHOLD = 0.0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output_scan/"
os.makedirs(OUT_DIR, exist_ok=True)

# Define optimization function
def fun(dofs, JF, Jf, Jls, Jenergy, Ja, Jcsdist, Jccdist, Jlink):
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
    outstr += f", CSdist={Jcsdist.J():.1e}"
    outstr += f", CCdist={Jccdist.J():.1e}"
    outstr += f", Link={Jlink.J():.1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

# Scan over all weight ranges
results = []

for ew in energy_weight_range:
    for lw in length_weight_range:
        for csw in cs_weight_range:
            for ccw in cc_weight_range:
                for lwk in link_weight_range:
                    for fw in flux_weight_range:
                        for aw in arclength_weight_range:
                            print(f"Running optimization for ENERGY_WEIGHT={ew}, LENGTH_WEIGHT={lw}, CS_WEIGHT={csw}, CC_WEIGHT={ccw}, LINK_WEIGHT={lwk}, FLUX_WEIGHT={fw}, ARCLENGTH_WEIGHT={aw}")
                            
                            # Set weights
                            ENERGY_WEIGHT = Weight(ew)
                            LENGTH_WEIGHT = Weight(lw)
                            CS_WEIGHT = Weight(csw)
                            CC_WEIGHT = Weight(ccw)
                            LINK_WEIGHT = Weight(lwk)
                            FLUX_WEIGHT = Weight(fw)
                            ARCLENGTH_WEIGHT = Weight(aw)

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
                            Jenergy = [CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j > i], regularization_rect(reg, reg)) for i in range(ncoils)]

                            Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
                            Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
                            Jlink = LinkingNumber(curves)

                            JF = FLUX_WEIGHT * Jf \
                                + ARCLENGTH_WEIGHT * sum(Ja) \
                                + ENERGY_WEIGHT * sum(Jenergy) \
                                + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 15.8, "max") \
                                + CS_WEIGHT * Jcsdist \
                                + CC_WEIGHT * Jccdist \
                                + LINK_WEIGHT * QuadraticPenalty(Jlink, LINK_THRESHOLD, "max")

                            # Run optimization 
                            dofs = JF.x
                            res = minimize(fun, dofs, args=(JF, Jf, Jls, Jenergy, Ja, Jcsdist, Jccdist, Jlink), 
                                           jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
                            
                            # Save results for Pareto analysis
                            result = {
                                'energy_weight': ew,
                                'length_weight': lw,
                                'cs_weight': csw,
                                'cc_weight': ccw,
                                'link_weight': lwk,
                                'flux_weight': fw,
                                'arclength_weight': aw,
                                'total_energy': sum(J.J() for J in Jenergy),
                                'total_length': sum(J.J() for J in Jls),
                                'total_flux': Jf.J(),
                                'BdotN':  np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))),
                                'avg_B': np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
                                'rel_B': np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))/np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)),
                                'C-S-Sep': Jcsdist.shortest_distance(),
                                'C-C-Sep': Jccdist.shortest_distance(),
                                'max_force': np.max([np.max(np.linalg.norm(coil_force(c, coils, regularization_rect(0.1, 0.1)), axis=1)) for c in coils]),
                                'arclength': sum(J.J() for J in Ja),
                                'link_number': Jlink.J()
                            }
                            results.append(result)

# Generate Pareto front based on all relevant objectives
pareto_front = []
for result in results:
    dominated = False
    for other in results:
        if (other['total_energy'] <= result['total_energy'] and
            other['max_force'] <= result['max_force'] and
            other['rel_B'] <= result['rel_B'] and
            other['C-S-Sep'] <= result['C-S-Sep'] and
            other['C-C-Sep'] <= result['C-C-Sep'] and
            other['link_number'] <= result['link_number'] and
            other['arclength'] <= result['arclength'] and
            (other['total_energy'] < result['total_energy'] or
             other['max_force'] < result['max_force'] or
             other['rel_B'] < result['rel_B'] or
             other['C-S-Sep'] < result['C-S-Sep'] or
             other['C-C-Sep'] < result['C-C-Sep'] or
             other['link_number'] < result['link_number'] or
             other['arclength'] < result['arclength'])):
            dominated = True
            break
    if not dominated:
        pareto_front.append(result)

# Print the number of Pareto front points found
print(f"Pareto front found with {len(pareto_front)} points.")

# Save Pareto front in CSV file
pareto_file = OUT_DIR + "pareto_front.csv"
with open(pareto_file, 'w', newline='') as csvfile:
    fieldnames = ['energy_weight', 'length_weight', 'cs_weight', 'cc_weight', 'link_weight', 'flux_weight', 'arclength_weight', 'total_energy', 'total_length', 'total_flux', 'BdotN', 'avg_B', 'rel_B', 'C-S-Sep', 'C-C-Sep', 'max_force', 'arclength', 'link_number']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in pareto_front:
        writer.writerow(result)
csvfile.close()
print(f"Pareto front saved to {pareto_file}")

# Save all results in CSV file
res_file = OUT_DIR + "results.csv"
with open(res_file, 'w', newline='') as csvfile:
    fieldnames = ['energy_weight', 'length_weight', 'cs_weight', 'cc_weight', 'link_weight', 'flux_weight', 'arclength_weight', 'total_energy', 'total_length', 'total_flux', 'BdotN', 'avg_B', 'rel_B', 'C-S-Sep', 'C-C-Sep', 'max_force', 'arclength', 'link_number']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
csvfile.close()
print(f"All scan results saved to {res_file}")

