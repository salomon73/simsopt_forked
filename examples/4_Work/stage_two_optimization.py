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
import vtk
import simsopt
import scipy as scp
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, plot_poincare_line, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, LinkingNumber)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy, SelfEnergy, MutualEnergy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print

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

## Methods to save coils with rect cross sections. 

# save forces on coils as points with magnitude of force
def pointData_forces(coils):
    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    point_data = {"F": forces}
    return point_data

def wrap(data):
    return np.concatenate([data, [data[0]]])

## Rotation Minimizing Frame helpers

# tangent
def unit_tangents(gamma):
    g = np.asarray(gamma, float)
    d = np.gradient(g, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d

# normal
def choose_initial_normal(t0):
    # pick a stable normal not parallel to t0
    v = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v, t0)) > 0.9:
        v = np.array([0.0, 1.0, 0.0])
    n0 = v - np.dot(v, t0) * t0
    return n0 / np.linalg.norm(n0)

def best_fit_plane_normal(gamma):
    """
    Compute the unit normal of the best-fit plane through a space curve.

    This function finds the least-squares plane that best approximates the
    given curve points and returns its unit normal vector. It is particularly
    useful for planar or nearly-planar curves (e.g. circular coils), where
    the plane normal provides a natural geometric reference direction.
    """
    g = np.asarray(gamma)
    g0 = g - g.mean(axis=0)
    _, _, vh = np.linalg.svd(g0, full_matrices=False)
    n_plane = vh[-1]
    return n_plane / np.linalg.norm(n_plane)

def choose_n0_planar(gamma, t0):
    b0 = best_fit_plane_normal(gamma)
    # ensure b0 not parallel to t0
    if abs(np.dot(b0, t0)) > 0.9:
        # fallback: any perpendicular direction
        v = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(v, t0)) > 0.9:
            v = np.array([0.0, 1.0, 0.0])
        n0 = v - np.dot(v, t0) * t0
        return n0 / np.linalg.norm(n0)

    n0 = np.cross(b0, t0)
    return n0 / np.linalg.norm(n0)

# Rotation Minimizing Frame
def rmf_parallel_transport(gamma):
    """
    Rotation-minimizing frame (Bishop/parallel transport).
    Returns unit tangents t, normals n, binormals b for each point.
    """
    g = np.asarray(gamma, float)
    t = unit_tangents(g)

    n = np.zeros_like(t)
    b = np.zeros_like(t)

    n[0] = choose_initial_normal(t[0])
    b[0] = np.cross(t[0], n[0])

    for i in range(1, len(g)):
        # rotate n[i-1] minimally to remain perpendicular to new tangent t[i]
        v = np.cross(t[i-1], t[i])
        s = np.linalg.norm(v)
        if s < 1e-12:
            n[i] = n[i-1]
        else:
            c = np.dot(t[i-1], t[i])
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
            n[i] = R @ n[i-1]
        n[i] -= np.dot(n[i], t[i]) * t[i]
        n[i] /= np.linalg.norm(n[i])
        b[i] = np.cross(t[i], n[i])

    return t, n, b

# Rotation Minimizing Frame (For planar coils)
def rmf_parallel_transport_planar(gamma):
    """
    Rotation-minimizing frame (Bishop/parallel transport).
    Returns unit tangents t, normals n, binormals b for each point.
    """
    g = np.asarray(gamma, float)
    t = unit_tangents(g)

    n = np.zeros_like(t)
    b = np.zeros_like(t)

    n[0] = choose_n0_planar(g, t[0])
    b[0] = np.cross(t[0], n[0])

    for i in range(1, len(g)):
        # rotate n[i-1] minimally to remain perpendicular to new tangent t[i]
        v = np.cross(t[i-1], t[i])
        s = np.linalg.norm(v)
        if s < 1e-12:
            n[i] = n[i-1]
        else:
            c = np.dot(t[i-1], t[i])
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
            n[i] = R @ n[i-1]
        n[i] -= np.dot(n[i], t[i]) * t[i]
        n[i] /= np.linalg.norm(n[i])
        b[i] = np.cross(t[i], n[i])

    return t, n, b

# close RMF in a periodic way (only if tangent match at endpoints)
def close_rmf(t, n, b):
    """
    Make RMF periodic by distributing the end-frame mismatch.
    """
    n0, b0 = n[0], b[0]
    n1, b1 = n[-1], b[-1]
    x = np.dot(n0, n1) + np.dot(b0, b1)
    y = np.dot(n0, b1) - np.dot(b0, n1)
    theta = np.arctan2(y, x)

    m = len(n)
    n2 = np.empty_like(n)
    b2 = np.empty_like(b)
    for i in range(m):
        a = -theta * i / (m - 1)
        ca, sa = np.cos(a), np.sin(a)
        n2[i] = ca * n[i] + sa * b[i]
        b2[i] = -sa * n[i] + ca * b[i]

    return t, n2, b2

# Sweep rectangle along the curve such that sides are aligned with RMF
def sweep_rectangle_rmf(gamma, n, b, w=0.5, h=0.5, extra_data=None):
    rect = np.array([[-w/2, -h/2],
                     [ w/2, -h/2],
                     [ w/2,  h/2],
                     [-w/2,  h/2]], dtype=float)
    nc = 4
    N = len(gamma)

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    polys = vtk.vtkCellArray()

    # points
    for i in range(N):
        p = gamma[i]
        ni = n[i]
        bi = b[i]
        for (u, v) in rect:
            pt = p + u * ni + v * bi
            points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))

    def pid(i, j):
        return i * nc + j

    # quads for i -> i+1
    for i in range(N - 1):
        for j in range(nc):
            j1 = (j + 1) % nc
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, pid(i, j))
            quad.GetPointIds().SetId(1, pid(i, j1))
            quad.GetPointIds().SetId(2, pid(i + 1, j1))
            quad.GetPointIds().SetId(3, pid(i + 1, j))
            polys.InsertNextCell(quad)

    # close: last -> first
    i = N - 1
    for j in range(nc):
        j1 = (j + 1) % nc
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, pid(i, j))
        quad.GetPointIds().SetId(1, pid(i, j1))
        quad.GetPointIds().SetId(2, pid(0, j1))
        quad.GetPointIds().SetId(3, pid(0, j))
        polys.InsertNextCell(quad)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polys)

    # point data: must be N*4 (NOT wrapped)
    if extra_data is not None:
        for name, arr in extra_data.items():
            arr = np.asarray(arr)
            if arr.shape[0] != poly.GetNumberOfPoints():
                raise ValueError(f"{name}: len {arr.shape[0]} != {poly.GetNumberOfPoints()}")
            vtk_arr = vtk.vtkDoubleArray()
            vtk_arr.SetName(str(name))
            vtk_arr.SetNumberOfComponents(1)
            vtk_arr.SetNumberOfTuples(poly.GetNumberOfPoints())
            for k, val in enumerate(arr):
                vtk_arr.SetValue(k, float(val))
            poly.GetPointData().AddArray(vtk_arr)

    poly.Modified()
    return poly

# save the coils
def coils_to_rectangular_vtk(
    coils,
    regularization,
    filename,
    w=0.3,
    h=0.3,
    planar = True,
    use_close_rmf=True,
    verbose=True,
):
    """
    Export coils as swept rectangular solids using a rotation-minimizing frame.

    This function:
      - extracts curves from the given coils
      - wraps curves for plotting (C⁰ closure)
      - constructs a rotation-minimizing frame (RMF)
      - optionally applies close_rmf (only meaningful if curve is C¹-periodic)
      - sweeps a rectangular cross-section of size (w, h)
      - attaches per-point force magnitude as point data ("F")
      - writes a single VTK PolyData file

    Parameters
    ----------
    coils : list
        List of simsopt Coil objects.
    regularization : object
        Regularization object passed to coil_force.
    filename : str
        Output VTK filename (e.g. "curves_final_rect.vtp").
    w, h : float
        Width and height of the rectangular coil cross-section.
    use_close_rmf : bool, optional
        Whether to apply close_rmf to the RMF.
        Has no visual effect if the curve is not C¹-periodic.
    verbose : bool, optional
        If True, print tangent mismatch diagnostics.

    Notes
    -----
    This function is intended for *visualization only*.
    The use of wrap() introduces a tangent mismatch at the seam, which is
    acceptable for plotting but should not be used for geometry-sensitive
    computations.
    """
    import vtk
    import numpy as np

    append = vtk.vtkAppendPolyData()

    def wrap(data):
        return np.concatenate([data, [data[0]]])

    curves = [coil.curve for coil in coils]

    for i, c in enumerate(curves[:len(coils)]):
        gamma = c.gamma() #wrap(c.gamma())

        # RMF construction
        if planar:
            t, n, b = rmf_parallel_transport_planar(gamma)
        else:
            t, n, b = rmf_parallel_transport(gamma)
        if use_close_rmf:
            t, n, b = close_rmf(t, n, b)

        # Tangent mismatch diagnostic (purely informational)
        if verbose:
            t0 = gamma[1] - gamma[0]
            t0 /= np.linalg.norm(t0)
            t1 = gamma[-1] - gamma[-2]
            t1 /= np.linalg.norm(t1)
            angle = np.degrees(
                np.arccos(np.clip(np.dot(t0, t1), -1.0, 1.0))
            )
            print(f"coil {i}: tangent angle mismatch (deg): {angle:.6f}")

        # Forces (per curve point, wrapped)
        force = np.linalg.norm(
            coil_force(coils[i], coils, regularization),
            axis=1
        )
        #force = np.append(force, force[0])
        surface_force = np.repeat(force, 4)
        coil_id = i * np.ones(4 * len(gamma), dtype=int)
        # Sweep surface
        poly = sweep_rectangle_rmf(
            gamma,
            n,
            b,
            w=w,
            h=h,
            extra_data={"F": surface_force,"id": coil_id},
        )

        poly.Modified()
        append.AddInputData(poly)

    append.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(append.GetOutput())
    writer.Write()

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4 # 4 QA
R0 = 14    # 1.0
R1 = 6     # 0.5
order = 10

# Regularization parameters
dim = 0.01
regularization = regularization_rect(dim,dim)

# Weights
FLUX_WEIGHT = Weight(10)
LENGTH_WEIGHT = Weight(0.0) # wl 4e-5 we = 1e-11 
ENERGY_WEIGHT   = Weight(0.0)  #1e-11
WEIGHT_SELF = Weight(1e-11)
WEIGHT_MUTUAL = Weight(1e-11)

ARCLENGTH_WEIGHT = Weight(1e-6) # 1e-6
CC_WEIGHT = Weight(0)
CS_WEIGHT = Weight(0)
LINK_WEIGHT = Weight(0.0)
CURVATURE_WEIGHT = Weight(0.0)
MSC_WEIGHT = Weight(0.0)

# Thresholds
CC_THRESHOLD = 1.10
CS_THRESHOLD = 1.6
LINK_THRESHOLD = 0.1
CURVATURE_THRESHOLD = 5.
MSC_THRESHOLD = 5
CL_THRESHOLD  = 18.1

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 800

# Bools 
taylor = True
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

# Initialize the boundary magnetic surface:
nphi = 128  
ntheta = 128
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
nfp = s.nfp

# Initialize the coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for _ in range(ncoils)]
# [base_currents[i].fix_all() for i in range(ncoils)]
#base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

# Save initial coils and surface
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Save initial coils with actual dimensions (not filaments for paraview)
coils_to_rectangular_vtk(coils, regularization,
    OUT_DIR + "curves_init_rect.vtp", w=0.3, h=0.5, planar = True, use_close_rmf=True, verbose=True)

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Ja = [ArclengthVariation(c) for c in base_curves]
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i], regularization) for i in range(ncoils)] 
# New self-energy list: one per coil
Jself = [ SelfEnergy(coils[i], regularization) for i in range(ncoils) ]
# New mutual-energy list: one per pair i<j (count each pair exactly once)
Jmutual = [
    MutualEnergy(coils[i],
                 [coils[j] for j in range(ncoils) if j > i],
                 regularization)
    for i in range(ncoils)
]
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]

#if CC_WEIGHT.value!=0:
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
#if CS_WEIGHT.value!=0:
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

if run_opt:
    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': False}, tol=1e-15)
    curves_to_vtk(curves[0:ncoils], OUT_DIR + "curves_opt_short_ew=" + f"{ENERGY_WEIGHT.value}", close=True, extra_data=pointData_forces(coils))
    print("coils saved to file" + OUT_DIR + "curves_opt_short_ew=" + f"{ENERGY_WEIGHT.value}")
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]/ np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)[:, :, None], \
                 "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_short_ew=" + f"{ENERGY_WEIGHT.value}", extra_data=pointData)
    print("Surf saved to file: " +  OUT_DIR + "surf_opt_short_ew=" + f"{ENERGY_WEIGHT.value}")
    bs.save(OUT_DIR + "biot_savart_opt.json")
    print("Biot Savart saved to file: " + OUT_DIR + "biot_savart_opt.json")
    print("Final coils characteristics:")
    dofs = res.x
    Jfinal, gradfinal = f(dofs)
    print('currents: ' + f"{[coil.current.get_value() for coil in coils[0:ncoils]]}")

    # Now save coils with actual dimensions (not filaments for paraview)
    coils_to_rectangular_vtk(coils, regularization,
    OUT_DIR + "curves_opt_rect.vtp", w=0.5, h=0.3, planar = False, use_close_rmf=True, verbose=True)

    
# Check Energy again
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