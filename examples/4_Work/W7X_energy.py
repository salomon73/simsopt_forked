import os, time, logging, vtk
from pathlib import Path
import numpy as np
import simsopt
import scipy.constants as constants
from scipy.optimize import minimize
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature,ArclengthVariation)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo.energy import CoilEnergy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.util import in_github_actions, comm_world, proc0_print
from simsopt.configs.zoo import get_w7x_data



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

import numpy as np

def best_fit_plane_normal(points):
    # points: (N,3)
    p = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(p, full_matrices=False)
    n = vh[-1]
    return n / np.linalg.norm(n)

def planarity_score(points):
    """
    Returns RMS distance to best-fit plane (meters).
    Smaller => more planar.
    """
    n = best_fit_plane_normal(points)
    p0 = points.mean(axis=0)
    d = (points - p0) @ n
    return np.sqrt(np.mean(d*d))

def split_planar_nonplanar_indices(coils, tol=1e-4, use_gammadash=False):
    """
    tol: planarity RMS threshold in meters.
         Start with 1e-4..1e-3 depending on your coil data scale/noise.
    """
    planar = []
    nonplanar = []
    scores = []
    for i, coil in enumerate(coils):
        g = coil.curve.gamma()
        s = planarity_score(g)
        scores.append(s)
        if s < tol:
            planar.append(i)
        else:
            nonplanar.append(i)
    return planar, nonplanar, np.array(scores)




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
#curves, currents, ma = get_w7x_data()
curves, currents, ma = get_w7x_data(ppp=5)
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

planar_indices = [
     0,  1,  7,  8,
    14, 15, 21, 22,
    28, 29, 35, 36,
    42, 43, 49, 50,
    56, 57, 63, 64
]
nonplanar_indices = [i for i in range(ncoils) if i not in planar_indices]

Jenergy = [ CoilEnergy(coils[i], [coils[j] for j in range(ncoils) if j>i ], regularization_rect(0.1, 0.1)) for i in range(ncoils)] 
#print("Total energy stored in W7X coils: " + f"{sum(Je.J()*1e-6/np.e for Je in Jenergy):2e}" + " MJ") 
print("Total energy stored in W7X coils: " + f"{sum(Je.J()*1e-6 for Je in Jenergy):2e}" + " MJ")  
planar_idx, nonplanar_idx, scores = split_planar_nonplanar_indices(coils, tol=5e-4)

print("Detected planar:", len(planar_idx), "nonplanar:", len(nonplanar_idx))
print("Smallest scores:", np.sort(scores)[:10])
print("Largest scores:",  np.sort(scores)[-10:])

curves_to_vtk(curves, OUT_DIR + f"W7X_curves_w_forces", extra_data=pointData_forces(coils))
coils_to_rectangular_vtk(
    [coils[i] for i in planar_idx],
    regularization_rect(0.1, 0.1),
    OUT_DIR + "curves_W7X_planar_rect.vtp",
    w=0.3, h=0.3, planar=True, use_close_rmf=True, verbose=True
)
coils_to_rectangular_vtk(
    [coils[i] for i in nonplanar_idx],
    regularization_rect(0.1, 0.1),
    OUT_DIR + "curves_W7X_nonplanar_rect.vtp",
    w=0.3, h=0.3, planar=False, use_close_rmf=True, verbose=True
)


