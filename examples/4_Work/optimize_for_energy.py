#!/usr/bin/env python
r"""
FOCUS like Stage II coil optimisation problem: find coils that 
generate a specific target Bn on a given surface. In this particular 
case we consider a vacuum field, so the target is just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + ENERGY_WEIGHT * Energy

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. 
The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os, time, logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mayavi.mlab import *
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.field.force import coil_force_pure
from simsopt.field.force import MeanSquaredForce, coil_force, LpCurveForce
from simsopt.field.selffield import regularization_rect, B_regularized_pure, regularization_circ
from simsopt.geo import (SurfaceRZFourier, CurveLength, MeanSquaredCurvature, curves_to_vtk, create_equally_spaced_curves, plot)
from simsopt.geo.energy import CoilEnergy
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions, comm_world, proc0_print


# INPUT PARAMETERS #
#######################################################

ncoils = 3 # times 2 for stellsym * 2 for nfp 

# coils radii
R0 = 1.0
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 4
                                                                            # LINEAR PENALTY WEIGHTS #
                                                                            # L: 1e-5, 1e-5
                                                                            # E: 1e-8, 1e-10
# Weights
LENGTH_WEIGHT   = Weight(1e-5)    #1e-5 #1e-5
ENERGY_WEIGHT   = Weight(1e-12)    #1e-8 #1e-10
FORCE_WEIGHT    = Weight(0.0)   #1e-26
MSC_WEIGHT      = Weight(0)
MSC_THRESHOLD   = 5


# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA' #'input.0604380'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)
 
# Make Poincaré plot
make_p2 = False

# Compute forces on coils
compute_forces = False
forces = False

# run second optim with first one as initial guess
init_guess = False

# Penalize energy with a quadratic penalty
quad_pen = True
#######################################################


# Initialize the boundary magnetic surface:
nphi = 200   #32
ntheta = 512 #32
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
nfp = s.nfp

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all() # Fix one current
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_coils = coils[:ncoils]

# Field from these coils
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

Jenergy = []
for i in range(ncoils):
    coils_i = coils.copy()
    coils_i.pop(i)
    Jenergy.append(CoilEnergy(coils[i], coils_i, regularization_rect(0.1,0.1)))

# Total objective function 
JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + MSC_WEIGHT    * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) 
if quad_pen:
    JF += ENERGY_WEIGHT * sum(QuadraticPenalty(Jenergy[i]) for i in range(ncoils))
else: 
    JF +=  ENERGY_WEIGHT * sum(Jenergy) 
if forces:
    Jforce = [LpCurveForce(c, coils, regularization_circ(0.05), p=2) for c in base_coils]
    JF +=  FORCE_WEIGHT  * sum(Jforce)

# Wrapper function for scipy.optimize.minimize
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    E_string = ", ".join(f"{J.J():1e}" for J in Jenergy)
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}]"
    outstr += f", E = sum([{E_string}])={sum(J.J() for J in Jenergy):1e}"
    if forces:
        force_string = ", ".join(f"{J.J():1e}" for J in Jforce)
        outstr += f", Fmax = sum([{force_string}])={sum(J.J() for J in Jforce):1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

def pointData_forces(coils):
    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    point_data = {"F": forces}
    return point_data


print("""
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
t = time.time()
J0, dJ0 = f(dofs)
elapsed = time.time() - t
print(elapsed, 's elapsed for compilation')
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)



print("""
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)


# Save results first opti
curves_to_vtk(curves, OUT_DIR + f"curves_opt_short_indiv", extra_data=pointData_forces(coils))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short_indiv", extra_data=pointData)

if init_guess:
    # Use result as initial guess for reduced length penalty. Slightly longer coils but smaller `B·n` on the surface.
    dofs = res.x
    LENGTH_WEIGHT *= 0.1
    print("Starting second optimization with reduced penalty w_L = " + f"{LENGTH_WEIGHT.__float__()}") 
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

    # Save results second opti
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_long_indiv", extra_data=pointData_forces(coils))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_long_indiv", extra_data=pointData)
    bs.save(OUT_DIR + "biot_savart_opt_indiv.json")


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

    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2) #h=0.03
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    def trace_fieldlines(bfield, label):
        t1 = time.time()
        # Set initial grid of points for field line tracing, going from
        # the magnetic axis to the surface. The actual plasma boundary is
        # at R=1.300425, but the outermost initial point is a bit inward
        # from that, @ R = 1.295, so the SurfaceClassifier does not think we
        # have exited the surface
        R0 = np.linspace(1.2125346, 1.295, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_indiv_{label}.pdf', dpi=300, surf=s)

            
    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 20 
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

if compute_forces:
    fmax = np.zeros(ncoils)
    forces = []

    for i in range(ncoils):
        coils_ext = coils.copy()
        coils_ext.pop(i)
        bs = BiotSavart(coils_ext)
        bs.set_points(coils[i].curve.gamma())
        Bext = bs.B()
        B_self = B_regularized_pure(
            coils[i].curve.gamma(), coils[i].curve.gammadash(),coils[i].curve.gammadashdash(), \
            coils[i].curve.quadpoints, coils[i].current.get_value(), regularization_rect(0.1,0.1))
        B_tot = B_self + Bext
        frenet_frame = coils[i].curve.frenet_frame()
        t = frenet_frame[0][:,:]
        forces.append(coil_force_pure(B_tot, coils[i].current.get_value(),t))

        fmax[i] = force_opt_pure(
            coils[i].curve.gamma(), 
            coils[i].curve.gammadash(), 
            coils[i].curve.gammadashdash(), 
            coils[i].current.get_value(), 
            2*np.pi*coils[i].curve.quadpoints,
            Bext, 
            regularization_rect(0.1,0.1)
            )
        
    print("Forces on the obtained coils")
    print(["coil " + f'{i+1}' for i in range(ncoils)])
    print(fmax)
    array_forces = np.array(forces)

    import pyvista as pv
    pos = np.concatenate([coils[k].curve.gamma() for k in range(ncoils)], axis=0)
    vect = np.concatenate(array_forces, axis=0)

    polydat = pv.PolyData(pos)
    polydat['forces'] = vect

    polydat.save(OUT_DIR+ 'forces_opt_indiv.vtk')

    #_________________________________________
    #          PLOT THE FINAL COILS
    # ________________________________________
    plt.figure
    for k in range(ncoils):
        coil = coils[k]
        curve = coil.curve
        Gamma = curve.gamma()
        x = Gamma[:,0]
        y = Gamma[:,1]
        z = Gamma[:,2]
        obj = quiver3d(x,y,z, array_forces[k,:,0], array_forces[k,:,1], array_forces[k,:,2], line_width=3, colormap='jet') 
    colorbar(object=obj, title=None, orientation=None, nb_labels=None, nb_colors=None, label_fmt=None)
    plot(coils[:] + [s], engine="mayavi", close=True)
