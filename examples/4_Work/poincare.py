#!/usr/bin/env python
import os, time, logging
import numpy as np
from mpi4py import MPI
from pathlib import Path

from simsopt._core import load
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field import (
    InterpolatedField,
    SurfaceClassifier,
    LevelsetStoppingCriterion,
    compute_fieldlines,
    plot_poincare_data,
)

# ==============================================================================
# MPI
# ==============================================================================
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def proc0_print(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)

# ==============================================================================
# User inputs
# ==============================================================================
OUT_DIR = Path("./poincare_out")
OUT_DIR.mkdir(exist_ok=True)

VMEC_INPUT = "input.LandremanPaul2021_QH_reactorScale_lowres"
BIOT_FILE  = "./output/biot_savart_opt.json"

# Fieldline / Poincaré parameters
nfieldlines = 16
tmax        = 1e7
tol         = 1e-10
phis        = [0.0, 0.25*np.pi]

# Interpolant resolution (strong shaping + torsion)
degree = 4
nR     = 24
nZ     = 24
nPhi   = 96

# Surface sampling for geometry
nphi_geom   = 256
ntheta_geom = 128

# ==============================================================================
# Logging (simsopt internal)
# ==============================================================================
logging.basicConfig()
logger = logging.getLogger("simsopt.field.tracing")
logger.setLevel(logging.INFO)

# ==============================================================================
# 1. Dense surface for geometry
# ==============================================================================
proc0_print("Loading VMEC surface")

phis_geom   = np.linspace(0, 1, nphi_geom, endpoint=False)
thetas_geom = np.linspace(0, 1, ntheta_geom, endpoint=False)

s = SurfaceRZFourier.from_vmec_input(
    VMEC_INPUT,
    quadpoints_phi=phis_geom,
    quadpoints_theta=thetas_geom,
)
nfp = s.nfp

# ==============================================================================
# 2. Inflate surface to define safe domain (Wechsung pattern)
# ==============================================================================
proc0_print("Building inflated domain surface")

s2 = SurfaceRZFourier.from_vmec_input(
    VMEC_INPUT,
    quadpoints_phi=phis_geom,
    quadpoints_theta=thetas_geom,
)

s2.extend_via_normal(0.01)
s2.scale(1.4)
s2.extend_via_normal(0.3)

if rank == 0:
    s.to_vtk(OUT_DIR / "surface_nominal")
    s2.to_vtk(OUT_DIR / "surface_inflated")

# ==============================================================================
# 3. SurfaceClassifier ONLY as a stopping criterion
# ==============================================================================
proc0_print("Initializing SurfaceClassifier")
sc = SurfaceClassifier(s2, h=0.05, p=2)
stopper = LevelsetStoppingCriterion(sc.dist)

# ==============================================================================
# 4. Build interpolation domain from inflated surface
# ==============================================================================
gamma = s2.gamma()
x, y, z = gamma[...,0], gamma[...,1], gamma[...,2]
R = np.sqrt(x*x + y*y)

Rmin, Rmax = R.min(), R.max()
Zmin, Zmax = z.min(), z.max()

Rbuf, Zbuf = 0.2, 0.2

rrange   = (Rmin - Rbuf, Rmax + Rbuf, nR)
#zrange   = (Zmin - Zbuf, Zmax + Zbuf, nZ)
zrange   = (0.0, Zmax + Zbuf, nZ)
phirange = (0, 2*np.pi/nfp, nPhi)

proc0_print("Interpolation domain:")
proc0_print("  rrange  =", rrange)
proc0_print("  zrange  =", zrange)
proc0_print("  phirange=", phirange)

# ==============================================================================
# 5. Load Biot–Savart and build interpolated field
# ==============================================================================
proc0_print("Loading Biot–Savart field")
bs = load(BIOT_FILE)

proc0_print("Initializing InterpolatedField")
bsh = InterpolatedField(
    bs,
    degree,
    rrange,
    phirange,
    zrange,
    extrapolate=False,
    nfp=nfp,
    stellsym=True,
)

# ==============================================================================
# 6. Initial conditions (radial fan)
# ==============================================================================

# ------------------------------------------------------------
# Fieldline initialization from a surface cross-section
# ------------------------------------------------------------
""" phi0 = 0.0  # toroidal launch plane

# Surface cross-section at phi0
theta = np.linspace(0, 1, 300, endpoint=True)
cross = s.cross_section(phi0, thetas=theta)

Rsurf = np.sqrt(cross[:, 0]**2 + cross[:, 1]**2)
Zsurf = cross[:, 2]

# Simple magnetic-axis proxy (sufficient for Poincaré)
R_axis = np.mean(Rsurf)
Z_axis = np.mean(Zsurf)

# Launch nested fieldlines from axis outward
nfieldlines = 16
R0 = R_axis + np.linspace(
    0.0,
    0.65 * (Rsurf.max() - R_axis),
    nfieldlines
)
Z0 = np.full_like(R0, Z_axis)

if rank == 0:
    print("Fieldline initialization:")
    print(f"  R_axis ≈ {R_axis:.3f}, Z_axis ≈ {Z_axis:.3f}")
    print(f"  R0 range: {R0[0]:.3f} → {R0[-1]:.3f}") """

# ------------------------------------------------------------
# Fieldline initialization along Z = 0 ray (axis -> boundary)
# ------------------------------------------------------------
phi0 = phis[0] if isinstance(phis, (list, np.ndarray)) else 0.0

# Surface cross-section at phi0
theta = np.linspace(0, 1, 800, endpoint=True)
cross = s.cross_section(phi0, thetas=theta)

Rsurf = np.sqrt(cross[:, 0]**2 + cross[:, 1]**2)
Zsurf = cross[:, 2]

# --- Find boundary intersection with Z = 0 plane ---
# Identify points near Z = 0
tolZ = 1e-3 * (Zsurf.max() - Zsurf.min())
mask = np.abs(Zsurf) < tolZ

if not np.any(mask):
    raise RuntimeError("No surface points found near Z=0; increase tolZ.")

# Outboard midplane point = max R at Z ≈ 0
R_boundary = np.max(Rsurf[mask])

# Axis proxy (centroid of cross-section)
R_axis = np.mean(Rsurf)
Z_axis = 0.0

# Number of fieldlines
nfieldlines = 16

# Uniform sampling from axis to boundary
R0 = np.linspace(R_axis, R_boundary, nfieldlines, endpoint=False)
Z0 = np.zeros_like(R0)

if rank == 0:
    print("\nFieldline initialization (Z=0 ray):")
    print(f"  R_axis     ≈ {R_axis:.4f}")
    print(f"  R_boundary ≈ {R_boundary:.4f}")
    print(f"  R0 range   = [{R0[0]:.4f}, {R0[-1]:.4f}]")



# Distribute fieldlines across MPI ranks
my_indices = list(range(rank, nfieldlines, size))

# Storage (gather later)
my_phi_hits = {}

# ==============================================================================
# 7. Trace fieldlines (WITH PROGRESS)
# ==============================================================================
proc0_print(f"Tracing {nfieldlines} fieldlines using {size} MPI ranks")

t_start = time.time()

for count, i in enumerate(my_indices):
    proc0_print(f"[rank {rank}] tracing fieldline {i+1}/{nfieldlines}")

    tys, hits = compute_fieldlines(
        bsh,
        [R0[i]],
        [Z0[i]],
        tmax=tmax,
        tol=tol,
        comm=None,                 # IMPORTANT: already parallelized by us
        phis=phis,
        stopping_criteria=[stopper],
    )

    my_phi_hits[i] = hits[0]

t_end = time.time()
proc0_print(f"Tracing done in {t_end - t_start:.1f} s")

# ==============================================================================
# 8. Gather results to rank 0
# ==============================================================================
all_hits = comm.gather(my_phi_hits, root=0)

if rank == 0:
    fieldlines_phi_hits = [None] * nfieldlines
    for d in all_hits:
        for i, hits in d.items():
            fieldlines_phi_hits[i] = hits
    # ------------------------------------------------------------
    # Phi-hit diagnostics: how many intersections per fieldline
    # ------------------------------------------------------------
    print("\nPhi-hit diagnostics:")
    for i, hits in enumerate(fieldlines_phi_hits):
        nhits = 0 if hits is None else len(hits)
        print(f"  Fieldline {i:02d}: {nhits:6d} hits")

    # ==============================================================================
    # 9. Plot Poincaré
    # ==============================================================================
    proc0_print("Plotting Poincaré")
    plot_poincare_data(
        fieldlines_phi_hits,
        phis,
        OUT_DIR / "poincare.pdf",
        dpi=300,
        surf=s,
        marker=".",
    )
    proc0_print("Saved:", OUT_DIR / "poincare.pdf")

