#!/usr/bin/env python
r"""
This coil optimization script is similar to stage_two_optimization.py. However
in this version, the coils are constrained to be planar, by using the curve type
CurvePlanarFourier. Also the LinkingNumber objective is used to prevent coils
from becoming topologically linked with each other.

In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + LinkingNumber

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os, vtk
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, curves_to_vtk, create_equally_spaced_planar_curves,
)
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
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 6

# Regularization parameters
dim = 0.015
regularization = regularization_rect(dim,dim)

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(0)
ENERGY_WEIGHT   = Weight(1e-11)  #1e-11

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.08
CC_WEIGHT = Weight(1000)

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.12
CS_WEIGHT = Weight(10)

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 10.
CURVATURE_WEIGHT = Weight(1e-6)

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 10
MSC_WEIGHT = Weight(1e-6)

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 400

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
ntheta = 128
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_planar_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
linkNum = LinkingNumber(curves)
Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i], regularization) for i in range(ncoils)] 


# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 2.6*ncoils) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + linkNum \
    + ENERGY_WEIGHT * sum(Jenergy)  \

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(bs.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", E = {sum(J.J() for J in Jenergy):1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
    outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
    outstr += f", Link Number = {linkNum.J()}"
    print(outstr)
    return J, grad



print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
f = fun
dofs = JF.x
# Now save coils with actual dimensions (not filaments for paraview)
coils_to_rectangular_vtk(coils, regularization,
    OUT_DIR + "curves_init_rect.vtp", w=0.1, h=0.1, planar = True, use_close_rmf=True, verbose=True)
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)
# Now save coils with actual dimensions (not filaments for paraview)
coils_to_rectangular_vtk(coils, regularization,
OUT_DIR + "curves_opt_rect_short.vtp", w=0.1, h=0.1, planar = True, use_close_rmf=True, verbose=True)


# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
dofs = res.x
LENGTH_WEIGHT *= 0.1
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt_long")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_long", extra_data=pointData)

# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR + "biot_savart_opt.json")
# Now save coils with actual dimensions (not filaments for paraview)
coils_to_rectangular_vtk(coils, regularization,
OUT_DIR + "curves_opt_rect_long.vtp", w=0.5, h=0.3, planar = True, use_close_rmf=True, verbose=True)