import numpy as np

import simsoptpp as sopp
from .curve import Curve

__all__ = ['CurvePlanarFourier','jax_planar_xyz_fourier_pure', 'JaxCurvePlanarXYZFourier']


class CurvePlanarFourier(sopp.CurvePlanarFourier, Curve):
    r"""
    ``CurvePlanarFourier`` is a curve that is restricted to lie in a plane. The
    shape of the curve within the plane is represented by a Fourier series in
    polar coordinates. The resulting planar curve is then rotated in three
    dimensions using a quaternion, and finally a translation is applied. The
    Fourier series in polar coordinates is

    .. math::

       r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi).

    The rotation quaternion is

    .. math::

       \bf{q} &= [q_0,q_i,q_j,q_k]

       &= [\cos(\theta / 2), \hat{x}\sin(\theta / 2), \hat{y}\sin(\theta / 2), \hat{z}\sin(\theta / 2)]

    where :math:`\theta` is the counterclockwise rotation angle about a unit axis
    :math:`(\hat{x},\hat{y},\hat{z})`. Details of the quaternion rotation can be
    found for example in pages 575-576 of
    https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf.


    A quaternion is used for rotation rather than other methods for rotation to
    prevent gimbal locking during optimization. The quaternion is normalized
    before being applied to prevent scaling of the curve. The dofs themselves are not normalized. This
    results in a redundancy in the optimization, where several different sets of 
    dofs may correspond to the same normalized quaternion. Normalizing the dofs 
    directly would create a dependence between the quaternion dofs, which may cause 
    issues during optimization.

    The dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, q_0, q_i, q_j, q_k, x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]


    """

    def __init__(self, quadpoints, order, nfp, stellsym, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1., quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurvePlanarFourier.__init__(self, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           x0=self.get_dofs())
        else:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           dofs=dofs)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurvePlanarFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurvePlanarFourier.set_dofs(self, dofs)

import numpy as np
import jax
import jax.numpy as jnp
from math import pi

from .jit import jit
from .._core.optimizable import Optimizable  # only if you need it elsewhere
from ..geo.curve import JaxCurve  # adjust import path to your JaxCurve base




def _quat_normalize(q):
    # Safe normalization to avoid NaNs if q ~ 0
    n = jnp.linalg.norm(q)
    n = jnp.where(n > 0.0, n, 1.0)
    return q / n


def _quat_to_rotmat(q):
    """
    q = [q0, qi, qj, qk] with q0 = scalar part.
    Returns 3x3 rotation matrix.
    """
    q = _quat_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Standard quaternion to rotation matrix
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = jnp.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ])
    return R


def _fourier_series(coeffs, t, order):
    """
    coeffs has length (2*order + 1) with layout:
      [c0, s1, c1, s2, c2, ..., s_order, c_order]

    t is in [0,1) (quadpoints convention in simsopt)
    """
    val = coeffs[0] * jnp.ones_like(t)
    for m in range(1, order + 1):
        sm = coeffs[2*m - 1]
        cm = coeffs[2*m]
        ang = 2 * pi * m * t
        val = val + sm * jnp.sin(ang) + cm * jnp.cos(ang)
    return val


def jax_planar_xyz_fourier_pure(dofs, quadpoints, order):
    """
    Planar curve parameterized in 2D (u(t), v(t)) Fourier series,
    then embedded in 3D via a rotated plane basis and translated.

    DOFs layout:
      [ u_coeffs(2*order+1),
        v_coeffs(2*order+1),
        q0, qi, qj, qk,
        xc, yc, zc ]

    Returns:
      gamma: (npts, 3)
    """
    n2 = 2 * order + 1

    u_coeffs = dofs[0:n2]
    v_coeffs = dofs[n2:2*n2]
    q = dofs[2*n2:2*n2 + 4]
    c = dofs[2*n2 + 4:2*n2 + 7]

    t = quadpoints

    u = _fourier_series(u_coeffs, t, order)
    v = _fourier_series(v_coeffs, t, order)

    R = _quat_to_rotmat(q)

    # Plane basis vectors: start from x-hat and y-hat, rotate them
    e1 = R @ jnp.array([1.0, 0.0, 0.0])
    e2 = R @ jnp.array([0.0, 1.0, 0.0])

    # Embed in 3D
    gamma = c[None, :] + u[:, None] * e1[None, :] + v[:, None] * e2[None, :]
    return gamma


class JaxCurvePlanarXYZFourier(JaxCurve):
    """
    Planar Cartesian Fourier curve:
      u(t), v(t) are Fourier series in a plane,
      the plane is oriented by a quaternion, and translated by a center.

    This avoids polar-coordinate issues of CurvePlanarFourier,
    while keeping exact planarity by construction.
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0.0, 1.0, quadpoints, endpoint=False)
        elif isinstance(quadpoints, np.ndarray):
            pass
        else:
            quadpoints = np.asarray(list(quadpoints))

        self.order = int(order)
        n2 = 2 * self.order + 1

        # Default: a unit circle in the x-y plane, centered at origin,
        # represented in (u,v) with u = cos(2πt), v = sin(2πt)
        u = np.zeros((n2,))
        v = np.zeros((n2,))
        if self.order >= 1:
            u[2] = 1.0   # c1 term for cos(2πt)
            v[1] = 1.0   # s1 term for sin(2πt)

        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        c = np.array([0.0, 0.0, 0.0])       # center

        x0 = np.concatenate([u, v, q, c])

        pure = lambda dofs_, points_: jax_planar_xyz_fourier_pure(dofs_, points_, self.order)

        if dofs is None:
            super().__init__(quadpoints, pure, x0=x0,
                             external_dof_setter=JaxCurvePlanarXYZFourier.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=JaxCurvePlanarXYZFourier.set_dofs_impl)

    def num_dofs(self):
        n2 = 2 * self.order + 1
        return 2*n2 + 4 + 3

    def get_dofs(self):
        return np.asarray(self.local_x)

    def set_dofs_impl(self, dofs):
        # JaxCurve stores dofs in local_x; we just mirror that behavior.
        self.local_x = np.asarray(dofs)

    @staticmethod
    def ellipse_dofs(order, a, b, center=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        """
        Build DOFs for a *perfect ellipse* in the local plane with semi-axes (a,b):
          u = a cos(2πt)
          v = b sin(2πt)
        All higher modes are exactly zero (so no wiggles even if order=20).

        Returns a dofs vector compatible with JaxCurvePlanarXYZFourier(order=...).
        """
        order = int(order)
        n2 = 2 * order + 1
        u = np.zeros((n2,))
        v = np.zeros((n2,))
        if order < 1:
            raise ValueError("Need order>=1 to represent an ellipse (mode 1).")
        u[2] = float(a)  # c1
        v[1] = float(b)  # s1
        q = np.asarray(quat, dtype=float)
        c = np.asarray(center, dtype=float)
        return np.concatenate([u, v, q, c])
