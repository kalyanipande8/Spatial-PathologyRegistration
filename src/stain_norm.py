"""
Real implementations of two classical stain-normalization algorithms:

  * Macenko et al. (2009), "A method for normalizing histology slides for
    quantitative analysis" -- SVD/plane-fitting based stain-vector
    estimation in optical density space, followed by concentration-based
    recoloring onto a target stain vector pair.

  * Reinhard et al. (2001), "Color Transfer between Images" -- Lab-space
    mean/std matching between source and a target reference image.

Both take an RGB uint8 image and a reference/target and return a
normalized RGB uint8 image.
"""
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Macenko normalization
# ---------------------------------------------------------------------------

def _rgb_to_od(img):
    """Convert RGB [0,255] image to optical density."""
    img = img.astype(np.float64)
    img[img == 0] = 1.0
    return -np.log(img / 255.0)


def _od_to_rgb(od):
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def macenko_stain_vectors(img, od_threshold=0.15, angular_percentile=1):
    """Estimate the 2 dominant stain vectors (columns) via SVD-plane fit."""
    od = _rgb_to_od(img).reshape(-1, 3)
    od_thresh = od[np.all(od > od_threshold, axis=1)]
    if od_thresh.shape[0] < 10:
        od_thresh = od

    # Eigenvectors of the covariance of OD values (project onto plane of
    # greatest variance, i.e. the 2 principal stain directions).
    cov = np.cov(od_thresh.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # top 2 eigenvectors
    top2 = eigvecs[:, [2, 1]]

    proj = od_thresh @ top2
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    min_angle = np.percentile(angles, angular_percentile)
    max_angle = np.percentile(angles, 100 - angular_percentile)

    v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])
    # OD vectors must point into the positive octant (Macenko convention)
    if v_min[0] < 0:
        v_min = -v_min
    if v_max[0] < 0:
        v_max = -v_max

    candidates = [v_min, v_max]
    candidates = [c / (np.linalg.norm(c) + 1e-8) for c in candidates]

    # Order columns to match the reference (target) stain matrix by cosine
    # similarity, so "column 1" is consistently the hematoxylin-like
    # direction and "column 2" the eosin/DAB-like direction across every
    # image -- without this, independent per-image SVD sign/order flips
    # cause wildly inconsistent recoloring.
    ref1, ref2 = REFERENCE_STAIN_MATRIX[:, 0], REFERENCE_STAIN_MATRIX[:, 1]
    sim_identity = (candidates[0] @ ref1) + (candidates[1] @ ref2)
    sim_swapped = (candidates[0] @ ref2) + (candidates[1] @ ref1)
    if sim_swapped > sim_identity:
        candidates = [candidates[1], candidates[0]]

    stain_matrix = np.stack(candidates, axis=1)  # 3x2
    return stain_matrix


def macenko_concentrations(img, stain_matrix, ridge=1e-2):
    od = _rgb_to_od(img).reshape(-1, 3).T  # 3 x N
    # Ridge-regularized least squares: od = stain_matrix (3x2) @ concentrations
    # (2xN). The small ridge term keeps the solve well-posed when the two
    # estimated stain vectors are nearly collinear (can happen on low-contrast
    # patches), avoiding the extreme/blown-up concentrations that would
    # otherwise saturate the reconstructed image.
    A = stain_matrix
    AtA = A.T @ A + ridge * np.eye(A.shape[1])
    AtB = A.T @ od
    concentrations = np.linalg.solve(AtA, AtB)
    concentrations = np.clip(concentrations, 0, None)
    return concentrations  # 2 x N


# Reference (target) stain matrix -- standard Macenko reference H&E vectors.
REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
])
REFERENCE_MAX_C = np.array([1.9705, 1.0308])


def macenko_normalize(img, target_stain_matrix=REFERENCE_STAIN_MATRIX,
                       target_max_c=REFERENCE_MAX_C):
    """Normalize `img` so its stain concentrations are rescaled onto the
    target stain vectors / target max concentration (classic Macenko
    color transfer)."""
    h, w, _ = img.shape
    source_stain_matrix = macenko_stain_vectors(img)
    concentrations = macenko_concentrations(img, source_stain_matrix)  # 2xN

    max_c = np.percentile(concentrations, 99, axis=1)
    max_c[max_c == 0] = 1e-8
    concentrations_norm = concentrations / max_c[:, None] * target_max_c[:, None]

    od_recon = target_stain_matrix @ concentrations_norm  # 3xN
    od_recon = od_recon.T.reshape(h, w, 3)
    return _od_to_rgb(od_recon)


# ---------------------------------------------------------------------------
# Reinhard normalization
# ---------------------------------------------------------------------------

def reinhard_normalize(img, target_img):
    """Reinhard et al. 2001 Lab-space mean/std color transfer."""
    src_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float64)

    src_mean, src_std = src_lab.reshape(-1, 3).mean(0), src_lab.reshape(-1, 3).std(0) + 1e-8
    tgt_mean, tgt_std = tgt_lab.reshape(-1, 3).mean(0), tgt_lab.reshape(-1, 3).std(0) + 1e-8

    result = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Color-consistency metric
# ---------------------------------------------------------------------------

def mean_std_color_signature(img):
    """Per-channel (mean, std) of the Lab a*/b* (chrominance/hue) channels
    -- a compact color signature used to measure cross-image / cross-subtype
    color consistency. L* (lightness) is deliberately excluded: it tracks
    overall tissue density/content (how much tissue vs. background a patch
    contains), which is a texture property stain normalization is not meant
    to change -- Macenko/Reinhard correct stain *color* (hue), not how much
    tissue is in the field of view."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    flat = lab.reshape(-1, 3)[:, 1:]  # a*, b* only
    return np.concatenate([flat.mean(0), flat.std(0)])


def color_consistency_distance(signatures):
    """Given a list of per-image color signatures, return the mean pairwise
    Euclidean distance -- lower = more consistent colors across images."""
    sigs = np.stack(signatures, axis=0)
    n = sigs.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(sigs[i] - sigs[j]))
    return float(np.mean(dists))
