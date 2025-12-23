import numpy as np
from simsopt.geo.framedcurve import FramedCurveCentroid
import vtk

__all__ = ["coils_to_rectangular_vtk"]

def wrap(data):
    """
    function that wrap any dataset by concatenating the first datapoint at the end.
    """
    return np.concatenate([data, [data[0]]])

def unit_tangents(gamma):
    """
    Function that returns the unit tangents to a curve gamma. 
    """
    g = np.asarray(gamma, float)
    d = np.gradient(g, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d

def choose_initial_normal(t0):
    """
    Find a suitable inital normal vector
    to the curve whose tangent at a given point is the input
    t0
    """
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
    """
    Choose an appropriate normal vector 
    to a planar curve (for Bishop frame only).
    """
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

def rmf_parallel_transport_planar(gamma):
    """
    Rotation-minimizing frame (Bishop/parallel transport).
    
    Args: 
        gamma: curve endpoints
    Returns:
        unit tangent t, normal n, and binormal b at each point.
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


def close_rmf(t, n, b):
    """
    Make RMF periodic by distributing the end-frame mismatch along the curve. 
    Introduce a non-physical rotation, for visualization only. 

    Args: 
    t,n,b: frame tangent, normal and binormal vectors respectively. 
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


def sweep_rectangle_centroid(gamma, n, b, w=0.5, h=0.5, extra_data=None):
    """
    This function returns a swept rectangular section of custom width and height along 
    a given frame of a curve. The frame is specified by n and b, and can be RMF or Centroid.

    Args:
        gamma: the curve quadpoints.
        n,b: normal and binormal vectors, respectively, of the frame.
        w,h: width and height of the rectangular cross section.
        extra_data (Optional): extra data to add to the curve 
                                to visualize in Paraview. 

    Returns:
        A set of polyData for Paraview
    """
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


def coils_to_rectangular_vtk(
    coils,
    filename,
    w=0.3,
    h=0.3,
    planar = False,
    extra_data = None,
    useRMF=False,
    verbose=False,
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
    planar : bool, optional 
        Wether the coil is planar or not (only relevant for RMF)
    useRMF : bool, optional
        Whether to choose the RMF or not (RMF is faster than centroid frame).
    verbose : bool, optional
        If True, print tangent mismatch diagnostics.

    Notes
    -----
    This function is intended for *visualization only*.
    """

    append = vtk.vtkAppendPolyData()

    curves = [coil.curve for coil in coils]

    for i, c in enumerate(curves[:len(coils)]):
        fc = FramedCurveCentroid(c)
        gamma = c.gamma() 

        # RMF construction
        if useRMF:
            if planar:
                t, n, b = rmf_parallel_transport_planar(gamma)
            else: 
                t, n, b = rmf_parallel_transport(gamma)
        else:
            t, n, b = fc.rotated_frame() 

        # Tangent mismatch diagnostic (purely informational)
        if verbose:
            t0 = t[0]
            t1 = t[-1]
            angle = np.degrees(
                np.arccos(np.clip(np.dot(t0, t1), -1.0, 1.0))
            )
            print(f"coil {i}: tangent angle mismatch (deg): {angle:.6f}")

        # Handle extra data
        surface_extra_data = {}
        if extra_data is not None:
            for name, values in extra_data.items():
                if values is None:
                    continue
                data_curve = values[i]
                if len(data_curve) != len(gamma):
                    raise ValueError(
                        f"extra_data['{name}'][{i}] has wrong length"
                    )
                # Each curve point generates 4 surface points
                surface_extra_data[name] = np.repeat(data_curve, 4)

        poly = sweep_rectangle_centroid(
            gamma,
            n,
            b,
            w=w,
            h=h,
            extra_data=surface_extra_data
        )
        poly.Modified()
        append.AddInputData(poly)

    append.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(append.GetOutput())
    writer.Write()
