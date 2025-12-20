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
import gvec
from pathlib import Path
import numpy as np
import simsopt
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_line, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, LinkingNumber)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, SelfEnergy, MutualEnergy
from simsopt.field.force import coil_force 
from simsopt.field.selffield import regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print
from simsopt.geo.curvetools import coils_to_rectangular_vtk

def taylor_test():
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

# save forces on coils as points with magnitude of force

def pointData_forces(coils, regularization):
    forces = []
    for c in coils:
        f = np.linalg.norm(
            coil_force(c, coils, regularization),
            axis=1
        )
        forces.append(f)   # NO concatenation
    return {"F": forces}

def flatten_point_data(curves, point_data, close=False):
    """
    Convert per-curve point data (list of arrays) into
    a single flat NumPy array per key, compatible with curves_to_vtk.
    """
    flat = {}

    for key, values in point_data.items():
        chunks = []
        for i, c in enumerate(curves):
            arr = np.asarray(values[i])
            if close:
                arr = np.concatenate([arr, [arr[0]]])
            chunks.append(arr)
        flat[key] = np.concatenate(chunks)

    return flat

def surf_to_xyzfourier(xyz_surf, 
                         nfp,
                         nphi=101,
                         ntheta=102, 
                         M=None,
                         N=None, 
                         stellsym=True, 
                         tol=1e-8, 
                         sign_rot=1,
                         range='full torus'
):
    """
    Converts a surface in cartesian coordinates with shape [0:nzeta*nfp,0:ntheta,0:2] 
    to a SurfaceXYZFourier Simsopt object.

    arguments:
        - xyz_surf: Cartesian surface points [0:nzeta*nfp,0:ntheta,0:2] (full torus, excluding end point)
        - nphi: nb of points in toroidal direction
        - ntheta: nb of points in poloidal direction
        - nfp: Number of field periods
        - M: if None, minimal maximum poloidal mode number s.t the error is below the tolerance tol
        - N: if None, minimal maximum toroidal mode number s.t the error is below the tolerance tol
        - stellsym: Stellarator symmetry if True
        - tol: Tolerance for the error associated to the minimal maximum mode numbers (M, N)
        - sign_rot: direction of zeta for the rotation into hat coordinates, +1 or -1
        - range: Range of which points are evaluated (e.g 'full torus' or 'half period')

    returns:
        - surf_xyzF_simsopt: SurfaceXYZFourier Simsopt object 
    """

    nz = xyz_surf.shape[0] // nfp
    zeta = np.linspace(0, 2*np.pi / nfp, nz, endpoint=False)
    xhat, yhat, zhat = gframe.xyz_to_xyz_hat(xyz_surf[0:nz,:,:], zeta, sign_rot=sign_rot)

    # Defining poloidal & toroidal mode numbers
    if (M==None and N==None):
        Mmax, Nmax = gframe.minimal_modes(xhat.T,yhat.T,zhat.T,tolerance=tol)
    else:
        Mmax, Nmax = M, N

    # Fourier transform
    xhat_c, xhat_s = fourier.fft2d(xhat.T)
    yhat_c, yhat_s = fourier.fft2d(yhat.T)
    zhat_c, zhat_s = fourier.fft2d(zhat.T)
    Mgrid, Ngrid = fourier.fft2d_modes(Mmax,Nmax,grid=True)

    print(f"Surface mode numbers: M={Mmax}, N={Nmax}")

    # Get Fourier modes
    xhat_c = fourier.scale_modes2d(xhat_c, Mmax, Nmax)
    xhat_s = fourier.scale_modes2d(xhat_s, Mmax, Nmax)
    yhat_c = fourier.scale_modes2d(yhat_c, Mmax, Nmax)
    yhat_s = fourier.scale_modes2d(yhat_s, Mmax, Nmax)
    zhat_c = fourier.scale_modes2d(zhat_c, Mmax, Nmax)
    zhat_s = fourier.scale_modes2d(zhat_s, Mmax, Nmax)

    # Create SurfaceXYZFourier Simsopt object
    surf_xyzF_simsopt = SurfaceXYZFourier.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, 
                                                           nfp=nfp, stellsym=stellsym, mpol=Mmax, ntor=Nmax)

    # Associate modes as dofs
    for m,n,xc,ys,zs in zip(Mgrid.flatten(),Ngrid.flatten(),xhat_c.flatten(),yhat_s.flatten(),zhat_s.flatten()):
        if not(m==0 and n<0):
            surf_xyzF_simsopt.set(f"xc({m},{n})",xc)
            if not(m==0 and n==0):
                surf_xyzF_simsopt.set(f"ys({m},{n})",ys)
                surf_xyzF_simsopt.set(f"zs({m},{n})",zs)

    # if not stellarator symmetric
    if stellsym==False:
        for m,n,xs,yc,zc in zip(Mgrid.flatten(),Ngrid.flatten(),xhat_s.flatten(),yhat_c.flatten(),zhat_c.flatten()):
            if not(m==0 and n<0):
                surf_xyzF_simsopt.set(f"yc({m},{n})",ys)
                surf_xyzF_simsopt.set(f"zc({m},{n})",zs)
                if not(m==0 and n==0):
                    surf_xyzF_simsopt.set(f"xs({m},{n})",xc)

    return surf_xyzF_simsopt
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp, ntotal = ncoils*2*nfp to get the total number of coils.)
ncoils = 4 # 4 QA
R0 = 14    # 1.0
R1 = 6     # 0.5
order = 6

# Regularization parameters
dim = 0.01
regularization = regularization_rect(dim,dim)

# Weights
FLUX_WEIGHT = Weight(10)
LENGTH_WEIGHT = Weight(0) # wl 4e-5 we = 1e-11 
ENERGY_WEIGHT   = Weight(3e-10)  #1e-11
WEIGHT_SELF = Weight(0.0)
WEIGHT_MUTUAL = Weight(0.0)

ARCLENGTH_WEIGHT = Weight(1e-2) # 1e-6
CC_WEIGHT = Weight(10)
CS_WEIGHT = Weight(100)
LINK_WEIGHT = Weight(10)
CURVATURE_WEIGHT = Weight(0.0)
MSC_WEIGHT = Weight(0.0)

# Thresholds
CC_THRESHOLD = 1.10
CS_THRESHOLD = 1.6
LINK_THRESHOLD = 0.1
CURVATURE_THRESHOLD = 5.
MSC_THRESHOLD = 5
CL_THRESHOLD  = 35

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 10
SAVE_EVERY = 1

# Bools 
taylor = False
callback = True
run_opt = True
compute_forces = False
make_p2 = False
init_guess = False

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres' #'input.LandremanPaul2021_QA'
#filename = "../2_Intermediate/inputs/input.LandremanPaul2021_QH"
# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Loading state Figure-8
state = gvec.State('parameter_NS_final.ini', 'NS_State_final.dat')

# Evaluating the position at boundary (rho=1.0) 
ev = state.evaluate('pos', 'mod_B', rho=1.0, theta=np.linspace(0, 2 * np.pi, 128), zeta=np.linspace(0, 2*np.pi, 20*state.nfp))

# Loading the boundary geometry
ev_bnd_2D = ev.sel(rho=1.0)
bnd = ev_bnd_2D.pos

# Transpose it so that if fits the shape for surf_to_xyzfourier
bnd = bnd.transpose('tor', 'pol', 'xyz')

# Converting into Simsopt object
xyz_simsopt = surf_to_xyzfourier(bnd, state.nfp, range='full torus', tol=1e-8)
xyz_simsopt.to_vtk('figure_8shape')

base_curves = create(state=state, ncoils=7, nfp=state.nfp, order=25, scale_fact=1.3)
print(len(base_curves))
#base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)

# Creates initial current:
base_currents = [Current(1e5) for i in range(7)]
base_currents[0].fix_all()

# Generate all coils via stellarator and field-period symmetry
coils = coils_via_symmetries(base_curves, base_currents, xyz_simsopt.nfp, True)
# Initialize the boundary magnetic surface:
nphi = 128  
ntheta = 128
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
nfp = s.nfp

# Initialize the coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1,
                                            order=order, numquadpoints=25*order)
base_currents = [Current(1e5) for _ in range(ncoils)]
[base_currents[i].fix_all() for i in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

# Save initial coils and surface
curves = [c.curve for c in coils]
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Save initial coils with actual dimensions (not filaments for paraview)
force_dict = pointData_forces(coils, regularization)
extra_data = {
    **force_dict,
    "coil_id": [i * np.ones(len(coils[i].curve.gamma())) for i in range(len(coils))]
}

# Now save coils with actual dimensions (not filaments for paraview)
curves_to_vtk(curves[0:ncoils], OUT_DIR + "curves_init", close=True, 
              extra_data=flatten_point_data(curves[:ncoils], force_dict, close=True))
coils_to_rectangular_vtk(coils, OUT_DIR + "curves_init_rect.vtp", w=0.4, h=0.4, 
                            planar = True, extra_data=extra_data, use_close_rmf=True, verbose=False)
#for animation (initial coils)
fname_vid = OUT_DIR + f"coils_iter_00000.vtp"

coils_to_rectangular_vtk(coils, fname_vid, w=0.4, h=0.4, 
                             planar = True, extra_data=extra_data, use_close_rmf=True, verbose=False)

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Ja = [ArclengthVariation(c) for c in base_curves]
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i], regularization) for i in range(ncoils)] 
# New self-energy list: one per coil
Jself = [ SelfEnergy(coils[i], regularization) for i in range(ncoils) ]
# New mutual-energy list: one per pair i<j (count each pair exactly once)
Jmutual = [ MutualEnergy(coils[i],[coils[j] for j in range(ncoils) if j > i],
                          regularization) for i in range(ncoils)]
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
if LINK_WEIGHT.value!=0:
    Jlink = LinkingNumber(curves) 

# compute numerical values
E_self = sum(J.J() for J in Jself)
E_mutual = sum(J.J() for J in Jmutual)
E_total = sum(J.J() for J in Jenergy)  # original total

print(f"E_self   = {E_self:.12e}")
print(f"E_mutual = {E_mutual:.12e}")
print(f"E_self + E_mutual = {E_self + E_mutual:.12e}")
print(f"E_total (Jenergy) = {E_total:.12e}")

# numeric check — change tolerances if needed
if not np.isclose(E_self + E_mutual, E_total, atol=1e-10, rtol=1e-8):
    print("Warning: decomposition does not match original CoilEnergy total (not close).")
else:
    print("OK: Self + Mutual matches CoilEnergy total (within tolerance).")
    

JF = FLUX_WEIGHT* Jf \
    + ARCLENGTH_WEIGHT * sum(Ja) \
    + WEIGHT_SELF * sum(Jself) \
    + WEIGHT_MUTUAL * sum(Jmutual) \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls),CL_THRESHOLD, "max") \
    + ENERGY_WEIGHT * sum(Jenergy)  \

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
    BdotNoverB = BdotN/ np.mean(np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)[:, :, None])
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/B={BdotNoverB:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    arcString = ", ".join(f"{J.J():1e}" for J in Ja)
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", E = {sum(J.J() for J in Jenergy):1e}"
    outstr += f", selfE = {sum(J.J() for J in Jself):1e}"
    outstr += f", mutualE = {sum(J.J() for J in Jmutual):1e}"
    outstr += f", arc = [{arcString}]"
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
    if LINK_WEIGHT.value!=0:
        outstr += f", Link=[{Jlink.J()}]"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

if taylor:
    taylor_test()

def save_coils(iter_counter):
    fname = OUT_DIR + f"coils_iter_{iter_counter:05d}.vtp"

    force_dict = pointData_forces(coils, regularization)

    extra_data = {
        **force_dict,
        "coil_id": [
            i * np.ones(len(coils[i].curve.gamma()))
            for i in range(len(coils))
        ],
    }
    if iter_counter ==0:
        plan = True
    else: plan = False
    coils_to_rectangular_vtk(
        coils,
        fname,
        w=0.4,
        h=0.4,
        planar=plan,
        extra_data=extra_data,
        use_close_rmf=True,
        verbose=False,
    )

def optimization_callback(xk):
    global iter_counter

    JF.x = xk
    if iter_counter % SAVE_EVERY == 0:
        save_coils(iter_counter)
    iter_counter += 1

def write_pvd( directory, filename="coils.pvd",pattern="coils_iter_", ext=".vtp", stride=1,):
    files = sorted(
        f for f in os.listdir(directory)
        if f.startswith(pattern) and f.endswith(ext)
    )

    with open(os.path.join(directory, filename), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')

        for i, name in enumerate(files[::stride]):
            f.write(f'    <DataSet timestep="{i}" file="{name}"/>\n')

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


if run_opt:
    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)

    # Save initial state (circular coils, before any step)
    iter_counter = 0
    dofs0 = JF.x.copy()
    JF.x = dofs0
    save_coils(iter_counter)
    iter_counter += 1

    # Run optimizer
    res = minimize(
        fun,
        dofs0,
        jac=True,
        method="L-BFGS-B",
        callback=optimization_callback,
        options={
            "maxiter": MAXITER,
            "maxcor": 300,
            "disp": False,
            "gtol": 1e-4,
        },
        tol=1e-15,
    )

    print(res.success, res.message)
    print("Final gradient norm:", np.linalg.norm(res.jac))
    # Write PVD after optimization
    write_pvd(OUT_DIR)

    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/ np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)[:, :, None], \
                 "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_short.vts", extra_data=pointData)
    print("Surf saved to file: " +  OUT_DIR + "surf_opt_short.vts")
    bs.save(OUT_DIR + "biot_savart_opt.json")
    print("Biot Savart saved to file: " + OUT_DIR + "biot_savart_opt.json")
    print("Final coils characteristics: ")
    dofs = res.x
    Jfinal, gradfinal = fun(dofs)
    print('currents: ' + f"{[coil.current.get_value() for coil in coils[0:ncoils]]}")
    force_dict = pointData_forces(coils, regularization)
    extra_data = {
        **force_dict,
        "coil_id": [i * np.ones(len(coils[i].curve.gamma())) for i in range(len(coils))]
    }
    curves_to_vtk(curves[0:ncoils], OUT_DIR + "curves_opt_filam", close=True, 
              extra_data=flatten_point_data(curves[:ncoils], force_dict, close=True))
    # Now save coils with actual dimensions (not filaments for paraview)
    coils_to_rectangular_vtk(coils, OUT_DIR + "curves_opt_rect.vtp", w=0.4, h=0.4, 
                             planar = False, extra_data=extra_data, use_close_rmf=True, verbose=False)
    print("coils saved to files" + "curves_opt_filam" + " and " + "curves_opt_rect.vtp")
    write_pvd(OUT_DIR)

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
    nfieldlines = 3 if in_github_actions else 12 
    tmax_fl = 10000 if in_github_actions else 30000
    degree = 2 if in_github_actions else 4

    # Load in the optimized coils from stage_two_optimization.py:
    coils_filename = OUT_DIR + "biot_savart_opt.json"
    bs = simsopt.load(coils_filename)

    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)
    sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)

    nplot_fieldlines = 3
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        # Set initial grid of points for field line tracing, going from
        # the magnetic axis to the surface. The actual plasma boundary is
        # at R=1.300425, but the outermost initial point is a bit inward
        # from that, R = 1.295, so the SurfaceClassifier does not think we
        # have exited the surface
        R0 = np.linspace(1.2125346, 1.295, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [0,np.pi/4, np.pi/2] #[(i/nplot_fieldlines)*(2*np.pi/nfp) for i in range(nplot_fieldlines)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
            plot_poincare_line(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.pdf', dpi=300, surf=s, marker='.')
            
    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 4 
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
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_long_indiv", close=True, extra_data=pointData_forces(coils))
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