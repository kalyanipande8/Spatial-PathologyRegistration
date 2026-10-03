"""
Target Registration Error (TRE) computation.

Given the known ground-truth affine transform (HE -> IHC space) used to
synthesize a pair, and the affine transform estimated by the registration
pipeline (IHC -> HE space, i.e. mapping the moving image onto the fixed
image), TRE is computed as the mean Euclidean pixel distance, over a grid
of control points, between:
  * where the true (inverse ground-truth) transform sends a point, and
  * where the estimated transform sends that same point.
"""
import numpy as np


def affine_to_3x3(M):
    out = np.eye(3)
    out[:2, :] = M
    return out


def invert_affine(M):
    M3 = affine_to_3x3(M)
    inv = np.linalg.inv(M3)
    return inv[:2, :]


def apply_affine(M, pts):
    """pts: Nx2 array of (x, y). Returns Nx2 transformed points."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (M @ pts_h.T).T
    return out[:, :2]


def control_point_grid(size, n_per_side=10, margin=0.1):
    lo = size * margin
    hi = size * (1 - margin)
    coords = np.linspace(lo, hi, n_per_side)
    xx, yy = np.meshgrid(coords, coords)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return pts


def compute_tre(M_gt_he_to_ihc, M_est_ihc_to_he, patch_size):
    """Returns (mean_tre, per_point_errors) in pixels.

    M_gt_he_to_ihc: ground truth transform mapping HE-space points to IHC-space.
    M_est_ihc_to_he: estimated transform mapping IHC-space points to HE-space
                      (this is exactly what register.estimate_affine returns
                      when called with fixed=HE, moving=IHC).
    """
    pts_he = control_point_grid(patch_size)

    # True correspondence: take HE control points, map to IHC space (ground
    # truth), then that's where those structures actually are in IHC image.
    pts_ihc_true = apply_affine(M_gt_he_to_ihc, pts_he)

    if M_est_ihc_to_he is None:
        # No registration achieved -- error is undefined/large; penalize.
        return float("nan"), np.array([])

    # Apply the estimated IHC->HE transform to the true IHC-space points;
    # a perfect registration would recover the original HE control points.
    pts_he_recovered = apply_affine(M_est_ihc_to_he, pts_ihc_true)

    errors = np.linalg.norm(pts_he_recovered - pts_he, axis=1)
    return float(np.mean(errors)), errors
