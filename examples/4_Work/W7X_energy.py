import os, vtk
from pathlib import Path
import numpy as np
from simsopt.field import (InterpolatedField, SurfaceClassifier, LevelsetStoppingCriterion, BiotSavart, Current,
                            particles_to_vtk, compute_fieldlines, plot_poincare_data, coils_via_symmetries)
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk)
from simsopt.geo.energy import CoilEnergy
from simsopt.field.force import coil_force_pure, coil_force 
from simsopt.field.selffield import B_regularized_pure, regularization_circ, regularization_rect
from simsopt.configs.zoo import get_w7x_data
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

## Methods to save coils with rect cross sections. 

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

def subset_point_data(point_data, indices):
    """
    Extract per-coil point data for a subset of coils.
    """
    sub = {}
    for key, values in point_data.items():
        sub[key] = [values[i] for i in indices]
    return sub

def wrap(data):
    return np.concatenate([data, [data[0]]])


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

force_dict = pointData_forces(coils, regularization_rect(0.1, 0.1))
extra_data = {
    **force_dict,
    "coil_id": [i * np.ones(len(coils[i].curve.gamma())) for i in range(len(coils))]
}

planar_coils = [coils[i] for i in planar_idx]

planar_extra_data = subset_point_data(force_dict, planar_idx)
planar_extra_data["coil_id"] = [
    np.full(len(planar_coils[i].curve.gamma()), i)
    for i in range(len(planar_coils))
]
nonplanar_coils = [coils[i] for i in nonplanar_idx]

nonplanar_extra_data = subset_point_data(force_dict, nonplanar_idx)
nonplanar_extra_data["coil_id"] = [
    np.full(len(nonplanar_coils[i].curve.gamma()), i)
    for i in range(len(nonplanar_coils))
]

curves_to_vtk(curves, OUT_DIR + "W7X_curves_w_forces", close=True, 
              extra_data=flatten_point_data(curves[:ncoils], force_dict, close=True))
coils_to_rectangular_vtk([coils[i] for i in planar_idx], OUT_DIR + "curves_W7X_planar_rect.vtp", w=0.3, h=0.3, 
                            planar = True, extra_data=planar_extra_data, use_close_rmf=True, verbose=True)
coils_to_rectangular_vtk([coils[i] for i in nonplanar_idx], OUT_DIR + "curves_W7X_nonplanar_rect.vtp", w=0.3, h=0.3, 
                            planar = False, extra_data=nonplanar_extra_data, use_close_rmf=True, verbose=True)




