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

import os, time, logging
from pathlib import Path
import numpy as np
import simsopt
import scipy as scp
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, LinkingNumber)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty, SquaredRootPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, Energy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 6

# Regularization parameters
regularization = regularization_rect(0.015,0.015)

# Weights
FLUX_WEIGHT = Weight(1)
LENGTH_WEIGHT = Weight(0.0) 
ENERGY_WEIGHT   = Weight(1e-11)  
ARCLENGTH_WEIGHT = Weight(1e-6)
CC_WEIGHT = Weight(0.0)
CS_WEIGHT = Weight(0.0)
LINK_WEIGHT = Weight(0.0)
CURVATURE_WEIGHT = Weight(0.0)
MSC_WEIGHT = Weight(0.0)

# Thresholds
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.1
LINK_THRESHOLD = 0.1
CURVATURE_THRESHOLD = 5.
MSC_THRESHOLD = 5
CL_THRESHOLD  = 18.1

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 800

# Bools 
run_opt = True
compute_forces = True
make_p2 = False
init_guess = False

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
nphi = 128 #128
ntheta = 64 #64
s = SurfaceRZFourier.from_vmec_input(filename, range="", nphi=nphi, ntheta=ntheta)
nfp = s.nfp

# Initialize the coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for _ in range(ncoils)]
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
Ja = [ArclengthVariation(c) for c in base_curves]
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i], regularization) for i in range(ncoils)] 
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]

#if CC_WEIGHT.value!=0:
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
#if CS_WEIGHT.value!=0:
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
if LINK_WEIGHT.value!=0:
    Jlink = LinkingNumber(curves) 


JF = FLUX_WEIGHT* Jf \
    + ARCLENGTH_WEIGHT * sum(Ja) \
    + ENERGY_WEIGHT * sum(Jenergy)  \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls),CL_THRESHOLD, "max") 

if CS_WEIGHT.value!=0:
    JF += CS_WEIGHT * Jcsdist 
if CC_WEIGHT.value!=0:
    JF += CC_WEIGHT * Jccdist 
if LINK_WEIGHT.value!=0:
    JF += LINK_WEIGHT * QuadraticPenalty(Jlink, LINK_THRESHOLD, "max")
if CURVATURE_WEIGHT!=0:
    JF += CURVATURE_WEIGHT*sum(Jmscs)
if MSC_WEIGHT.value !=0:
    JF += MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)  
    
# Wrapper function for scipy.minimize
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    arcString = ", ".join(f"{J.J():1e}" for J in Ja)
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", E = {sum(J.J() for J in Jenergy):1e}"
    outstr += f", arc = [{arcString}]"
    #if CC_WEIGHT.value!=0:
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    #if CS_WEIGHT.value!=0:
    outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
    if LINK_WEIGHT.value!=0:
        outstr += f", Link=[{Jlink.J()}]"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

# save forces on coils as points with magnitude of force
def pointData_forces(coils):
    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    point_data = {"F": forces}
    return point_data


if run_opt:
    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
    curves_to_vtk(curves, OUT_DIR + "curves_opt_short_ew=" + f"{ENERGY_WEIGHT.value}", close=True, extra_data=pointData_forces(coils))
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/ np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)[:, :, None], \
                 "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_short_ew=" + f"{ENERGY_WEIGHT.value}", extra_data=pointData)
    bs.save(OUT_DIR + "biot_savart_opt.json")
    print("Final coils characteristics:")
    dofs = res.x
    Jfinal, gradfinal = f(dofs)
    print('currents: ' + f"{[coil.current.get_value() for coil in coils[0:ncoils]]}")

if compute_forces:
    f_int_max = np.zeros((2,ncoils))

    for i in range(ncoils):
        f_int_max[0,i] = np.max(np.linalg.norm(coil_force(coils[i], coils, regularization), axis=1))
        arc_length = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
        f_int_max[1,i] = 1/CurveLength(coils[i].curve).J() * np.mean(np.linalg.norm(coil_force(coils[i], coils, regularization), axis=1) * arc_length)

    print("Forces on the obtained coils")
    print(["coil " + f'{i+1}' for i in range(ncoils)])
    print('max force:' +f"{f_int_max[0,:]}")
    print('integrated force:' +f"{f_int_max[1,:]}")


if make_p2:
    proc0_print("Tracing the Poincaré plot for the obtained field lines")
    proc0_print("======================================================")

    logging.basicConfig()
    logger = logging.getLogger('simsopt.field.tracing')
    logger.setLevel(1)

    # If we're in the CI, make the run a bit cheaper:
    nfieldlines = 3 if in_github_actions else 10 
    tmax_fl = 10000 if in_github_actions else 20000
    degree = 2 if in_github_actions else 4

    # Load in the optimized coils from stage_two_optimization.py:
    coils_filename = OUT_DIR + "biot_savart_opt.json"
    bs = simsopt.load(coils_filename)

    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    nplot_fieldlines = 4
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        # Set initial grid of points for field line tracing, going from
        # the magnetic axis to the surface. The actual plasma boundary is
        # at R=1.300425, but the outermost initial point is a bit inward
        # from that, R = 1.295, so the SurfaceClassifier does not think we
        # have exited the surface
        R0 = np.linspace(1.2125346, 1.295, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i/nplot_fieldlines)*(2*np.pi/nfp) for i in range(nplot_fieldlines)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.pdf', dpi=300, surf=s, marker='.')
            
    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 5 
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), n//2)


    def skip(rs, phis, zs):
        # The RegularGrindInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if *all* of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip


    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
    )
    proc0_print('Done initializing InterpolatedField.')

    bsh.set_points(s.gamma().reshape((-1, 3)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))

    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))

    proc0_print('Beginning field line tracing')
    trace_fieldlines(bsh, 'bsh')

    proc0_print("Poincaré plot finalized")
    proc0_print("========================================")


# New optimization with initial guess 
if init_guess:
    # Use result as initial guess for reduced length penalty. Slightly longer coils but smaller `B·n` on the surface.
    dofs = res.x
    ENERGY_WEIGHT = Weight(1e-9)
    print("Starting second optimization with reduced penalty w_L = " + f"{LENGTH_WEIGHT.__float__()}") 
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

    # Save results second opti
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_long_indiv", extra_data=pointData_forces(coils))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_long_indiv", extra_data=pointData)
    bs.save(OUT_DIR + "biot_savart_opt_indiv.json")

    f_int_max = np.zeros((2,ncoils))

    for i in range(ncoils):
        f_int_max[0,i] = np.max(np.linalg.norm(coil_force(coils[i], coils, regularization), axis=1))
        arc_length = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
        f_int_max[1,i] = 1/CurveLength(coils[i].curve).J() * np.mean(np.linalg.norm(coil_force(coils[i], coils, regularization), axis=1) * arc_length)

    print("Forces on the obtained coils")
    print(["coil " + f'{i+1}' for i in range(ncoils)])
    print('max force:' +f"{f_int_max[0,:]}")
    print('integrated force:' +f"{f_int_max[1,:]}")