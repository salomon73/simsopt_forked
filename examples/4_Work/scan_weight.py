import os, time, logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp
from mayavi.mlab import *
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import regularization_rect, B_regularized_pure
from simsopt.geo import (SurfaceRZFourier, CurveLength, LinkingNumber, MeanSquaredCurvature, curves_to_vtk, create_equally_spaced_curves, plot, ArclengthVariation)
from simsopt.geo.energy import CoilEnergy
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions, comm_world, proc0_print


# INPUT PARAMETERS #
#######################################################

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)
 
ncoils = 4 # times 2 for stellsym * 2 for nfp 

# coils radii
R0 = 1.0
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weights
FLUX_WEIGHT = Weight(1.)
LENGTH_WEIGHT = Weight(1e-5)
ARCLENGTH_WEIGHT = Weight(1e-4)
array_energy_weights = np.logspace(-12,-7,40)

#######################################################

for i in range(array_energy_weights.shape[0]):

    ENERGY_WEIGHT = Weight(array_energy_weights[i])   
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

    Jf =     SquaredFlux(s, bs)
    Jls =   [CurveLength(c) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Ja =    [ArclengthVariation(c) for c in base_curves]
    Jenergy = []
    for i in range(ncoils):
        coils_i = coils.copy()
        coils_i.pop(i)
        Jenergy.append(CoilEnergy(coils[i], coils_i, regularization_rect(0.1,0.1)))


    JF = FLUX_WEIGHT* Jf \
        + ENERGY_WEIGHT * sum(Jenergy[i] for i in range(ncoils)) \
        + ARCLENGTH_WEIGHT * sum(Ja)
    #   + LENGTH_WEIGHT * sum(Jls) \


    # Wrapper function for scipy.minimize
    def fun(dofs):
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
        return J, grad, outstr

    # save forces on coils as points with magnitude of force
    def pointData_forces(coils):
        forces = []
        for c in coils:
            force = np.linalg.norm(coil_force(c, coils, regularization_rect(0.1,0.1)), axis=1)
            force = np.append(force, force[0])
            forces = np.concatenate([forces, force])
        point_data = {"F": forces}
        return point_data


    print("""
    ################################################################################
    ### Perform a Taylor test ######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    J0, dJ0, _ = f(dofs)
    dJh = sum(dJ0 * h)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        J1, _, _ = f(dofs + eps*h)
        J2, _, _ = f(dofs - eps*h)
        print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
    final = f(res.x)
    Jf, gradf, outstrf = final
    curves_to_vtk(curves, OUT_DIR + "curves_opt_short", close=True, extra_data=pointData_forces(coils))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

    ind_f = 1
    f_int_max = np.zeros((2,ncoils))
    forces = []

    for i in range(ncoils):
        coils_ext = coils.copy()
        coils_ext.pop(i)
        bs = BiotSavart(coils_ext)
        bs.set_points(coils[i].curve.gamma()[ind_f-1:-ind_f])
        Bext = bs.B()
        B_self = B_regularized_pure(
            coils[i].curve.gamma()[ind_f-1:-ind_f], coils[i].curve.gammadash()[ind_f-1:-ind_f],coils[i].curve.gammadashdash()[ind_f-1:-ind_f], \
            2*np.pi*coils[i].curve.quadpoints[ind_f-1:-ind_f], coils[i].current.get_value(), regularization_rect(0.1,0.1))
        B_tot = B_self + Bext
        frenet_frame = coils[i].curve.frenet_frame()
        t = frenet_frame[0][:,:]
        force_pure = coil_force_pure(B_tot, coils[i].current.get_value(),t[ind_f-1:-ind_f])
        forces.append(force_pure)
        f_int_max[0,i] = np.max(np.linalg.norm(force_pure, axis=1)) 
        dphi = 2*np.pi/coils[i].curve.quadpoints.shape[0]
        f_int_max[1,i] = 1/CurveLength(coils[i].curve).J()*scp.integrate.trapezoid(np.linalg.norm(force_pure,ord=2, axis=1), dx=dphi)

    print("Forces on the obtained coils")
    print(["coil " + f'{i+1}' for i in range(ncoils)])
    print(f_int_max[0,:])
    print(f_int_max[1,:])
    array_forces = np.array(forces)

    print("Forces on the obtained coils")
    print(["coil " + f'{i+1}' for i in range(ncoils)])
    print(f_int_max[0,:])
    print(f_int_max[1,:])
    array_forces = np.array(forces)
        
    with open(OUT_DIR +f"Scan_weight_energy_Nf={order}_Nc={ncoils}_wL=5e{6}_long_linkNum.txt", 'a') as f:
        print(f" {float(LENGTH_WEIGHT):e}, " + f" {float(ENERGY_WEIGHT):e}, " + outstrf + f", {f_int_max[0,0]}, " + f" {f_int_max[0,1]}, " + f" {f_int_max[0,2]}, " + f" {f_int_max[0,3]}, " + f" {f_int_max[1,0]}, " + f" {f_int_max[1,1]}, " + f" {f_int_max[1,2]}, "+ f" {f_int_max[1,3]}", file=f)
    f.close()
