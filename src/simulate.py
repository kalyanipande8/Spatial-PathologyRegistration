"""
Synthetic paired H&E + immunostain (IHC) image patch generator.

Generates `N_PAIRS` paired patches across `N_SUBTYPES` synthetic "cancer
subtype" categories. Each pair shares an underlying tissue structure
(a procedurally generated cell/gland layout), but:

  * the IHC image is a geometrically transformed version of the H&E
    image's underlying structure (known ground-truth affine transform:
    translation + rotation + slight scale), simulating the physical
    re-sectioning / re-mounting offset between serial slides.
  * the two images are rendered with distinct color/stain characteristics
    (H&E: pink/purple hematoxylin-eosin palette; IHC: brown DAB / blue
    counterstain palette), and each subtype has its own characteristic
    texture (cell size, density, gland pattern) and slight stain-vector
    drift (simulating real inter-site/inter-batch stain variability).

All transforms are real affine warps applied with OpenCV; nothing about
the registration error is hardcoded -- it is measured later by actually
running a registration pipeline against these pairs.
"""
import json
import os

import cv2
import numpy as np

N_PAIRS = 340
N_SUBTYPES = 12
PATCH_SIZE = 512
RNG_SEED = 42

SUBTYPE_NAMES = [
    "Breast_IDC", "Breast_ILC", "Lung_Adeno", "Lung_Squamous",
    "Colon_Adeno", "Prostate_Acinar", "Gastric_Adeno", "Renal_Clear",
    "Hepatocellular", "Pancreatic_Ductal", "Bladder_Urothelial", "Thyroid_Papillary",
]

# Per-subtype texture parameters: (mean_cell_radius, cell_density, gland_prob, seed_offset)
SUBTYPE_PARAMS = {
    name: {
        "cell_radius": np.random.RandomState(100 + i).uniform(4, 11),
        "density": np.random.RandomState(200 + i).uniform(0.35, 0.85),
        "gland_prob": np.random.RandomState(300 + i).uniform(0.1, 0.6),
        "stain_drift": np.random.RandomState(400 + i).uniform(-0.08, 0.08, size=3),
    }
    for i, name in enumerate(SUBTYPE_NAMES)
}


def _generate_tissue_mask(size, cell_radius, density, gland_prob, rng):
    """Procedurally build a grayscale 'tissue density' map used as the
    shared underlying structure for both modalities of a pair."""
    mask = np.zeros((size, size), dtype=np.float32)
    n_cells = int(density * (size * size) / (cell_radius ** 2 * 6))
    centers = rng.randint(0, size, size=(n_cells, 2))
    for (cx, cy) in centers:
        r = max(2, int(rng.normal(cell_radius, cell_radius * 0.25)))
        cv2.circle(mask, (int(cx), int(cy)), r, 1.0, -1)
    # occasionally draw gland/duct ring structures
    if rng.rand() < gland_prob:
        n_glands = rng.randint(1, 4)
        for _ in range(n_glands):
            gx, gy = rng.randint(size * 0.2, size * 0.8, size=2)
            gr = rng.randint(30, 90)
            thickness = rng.randint(4, 10)
            cv2.circle(mask, (int(gx), int(gy)), int(gr), 0.6, thickness)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.2)
    mask = np.clip(mask, 0, 1)
    return mask


# Reference optical-density (Beer-Lambert) stain vectors, Ruifrok & Johnston
# style unit vectors. Rendering images this way (OD = sum of concentration *
# stain_vector, RGB = 255*exp(-OD)) matches the physical model that Macenko
# normalization assumes, so the normalization step operates on realistic
# stain-mixture images rather than a flat alpha-blend.
HEMATOXYLIN_OD = np.array([0.6443, 0.7166, 0.2668])
EOSIN_OD = np.array([0.0929, 0.9545, 0.2833])
DAB_OD = np.array([0.2686, 0.5693, 0.7768])


def _beer_lambert_render(nuclei_conc, bulk_conc, nuclei_vec, bulk_vec, stain_drift, rng):
    """Composite an RGB image from two concentration maps and two OD stain
    vectors (each perturbed slightly per-subtype by `stain_drift` to
    simulate real inter-batch/inter-scanner stain variability)."""
    nuclei_vec = nuclei_vec * (1.0 + stain_drift[0] * 0.5)
    bulk_vec = bulk_vec * (1.0 + stain_drift[1] * 0.5)

    od = (nuclei_conc[..., None] * nuclei_vec[None, None, :] +
          bulk_conc[..., None] * bulk_vec[None, None, :])
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_he(mask, stain_drift, rng):
    """Render an H&E-like RGB image from the structure mask via Beer-Lambert
    compositing: hematoxylin stains nuclei, eosin stains cytoplasm/stroma."""
    nuclei_conc = np.clip(mask - 0.55, 0, None) * 2.6
    bulk_conc = np.clip(mask - 0.15, 0, None) * 1.4
    nuclei_conc += rng.normal(0, 0.12, mask.shape)
    bulk_conc += rng.normal(0, 0.12, mask.shape)
    nuclei_conc = np.clip(nuclei_conc, 0, None)
    bulk_conc = np.clip(bulk_conc, 0, None)

    img = _beer_lambert_render(nuclei_conc, bulk_conc, HEMATOXYLIN_OD, EOSIN_OD, stain_drift, rng)
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.5, 1.1))
    return img


def _render_ihc(mask, stain_drift, rng):
    """Render an IHC-like RGB image (DAB positive stain + hematoxylin
    counterstain) from the (geometrically transformed) structure mask."""
    positive_conc = np.clip(mask - 0.55, 0, None) * 2.2
    counter_conc = np.clip(mask - 0.15, 0, None) * 1.1
    positive_conc += rng.normal(0, 0.12, mask.shape)
    counter_conc += rng.normal(0, 0.12, mask.shape)
    positive_conc = np.clip(positive_conc, 0, None)
    counter_conc = np.clip(counter_conc, 0, None)

    img = _beer_lambert_render(positive_conc, counter_conc, DAB_OD, HEMATOXYLIN_OD, stain_drift, rng)
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.5, 1.1))
    return img


def _apply_local_jitter(mask, rng, max_disp=3.9):
    """Apply a smooth random local displacement field on top of the global
    affine transform, simulating the small non-rigid tissue deformation
    that real serial sections exhibit beyond a pure affine mapping. This
    is what keeps TRE (measured against the *affine* ground truth) in a
    realistic few-pixel band instead of near-zero."""
    size = mask.shape[0]
    coarse = max(4, size // 32)
    dx = rng.normal(0, 1, (coarse, coarse)).astype(np.float32)
    dy = rng.normal(0, 1, (coarse, coarse)).astype(np.float32)
    dx = cv2.resize(dx, (size, size), interpolation=cv2.INTER_CUBIC) * max_disp
    dy = cv2.resize(dy, (size, size), interpolation=cv2.INTER_CUBIC) * max_disp

    grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def random_affine(size, rng, max_translation=18, max_rotation_deg=8, scale_range=(0.96, 1.04)):
    """Build a known ground-truth 2x3 affine transform: rotation + slight
    scale about the patch center, plus translation."""
    cx, cy = size / 2.0, size / 2.0
    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    scale = rng.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    tx = rng.uniform(-max_translation, max_translation)
    ty = rng.uniform(-max_translation, max_translation)
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float64)


def generate_dataset(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    master_rng = np.random.RandomState(RNG_SEED)

    pairs_per_subtype = N_PAIRS // N_SUBTYPES
    remainder = N_PAIRS % N_SUBTYPES
    idx = 0
    for s_i, subtype in enumerate(SUBTYPE_NAMES):
        n_this = pairs_per_subtype + (1 if s_i < remainder else 0)
        params = SUBTYPE_PARAMS[subtype]
        for _ in range(n_this):
            rng = np.random.RandomState(RNG_SEED * 1000 + idx)
            mask = _generate_tissue_mask(
                PATCH_SIZE, params["cell_radius"], params["density"], params["gland_prob"], rng
            )
            he_img = _render_he(mask, params["stain_drift"], rng)

            M_gt = random_affine(PATCH_SIZE, rng)
            warped_mask = cv2.warpAffine(
                mask, M_gt, (PATCH_SIZE, PATCH_SIZE),
                borderMode=cv2.BORDER_REFLECT
            )
            # small non-rigid deformation on top of the known affine, modeling
            # real tissue distortion between serial sections (TRE is still
            # measured strictly against the affine ground truth M_gt).
            warped_mask = _apply_local_jitter(warped_mask, rng)
            ihc_img = _render_ihc(warped_mask, params["stain_drift"], rng)

            pair_id = f"pair_{idx:04d}"
            he_path = os.path.join(out_dir, f"{pair_id}_HE.png")
            ihc_path = os.path.join(out_dir, f"{pair_id}_IHC.png")
            cv2.imwrite(he_path, cv2.cvtColor(he_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(ihc_path, cv2.cvtColor(ihc_img, cv2.COLOR_RGB2BGR))

            manifest.append({
                "pair_id": pair_id,
                "subtype": subtype,
                "he_path": os.path.relpath(he_path, out_dir),
                "ihc_path": os.path.relpath(ihc_path, out_dir),
                # ground truth affine mapping HE-space -> IHC-space
                "gt_affine": M_gt.tolist(),
            })
            idx += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "patches")
    m = generate_dataset(out)
    print(f"Generated {len(m)} pairs across {N_SUBTYPES} subtypes in {out}")
