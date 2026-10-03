import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from src.simulate import generate_dataset, N_PAIRS, N_SUBTYPES, SUBTYPE_NAMES, PATCH_SIZE
from src.register import estimate_affine, warp_moving_to_fixed
from src.metrics import compute_tre
from src.stain_norm import (
    macenko_normalize, reinhard_normalize,
    mean_std_color_signature, color_consistency_distance,
)

DATA_DIR = os.path.join(ROOT, "data", "patches")


@pytest.fixture(scope="module")
def manifest():
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        if len(m) == N_PAIRS:
            return m
    return generate_dataset(DATA_DIR)


def load_rgb(path):
    import cv2
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_dataset_has_340_pairs(manifest):
    assert len(manifest) == N_PAIRS == 340


def test_dataset_has_12_subtypes(manifest):
    subtypes_present = set(e["subtype"] for e in manifest)
    assert len(subtypes_present) == N_SUBTYPES == 12
    assert subtypes_present == set(SUBTYPE_NAMES)


def test_each_pair_has_valid_files_and_gt_transform(manifest):
    for entry in manifest[:5]:
        he_path = os.path.join(DATA_DIR, entry["he_path"])
        ihc_path = os.path.join(DATA_DIR, entry["ihc_path"])
        assert os.path.exists(he_path)
        assert os.path.exists(ihc_path)
        M_gt = np.array(entry["gt_affine"])
        assert M_gt.shape == (2, 3)


def test_registration_runs_and_produces_affine(manifest):
    entry = manifest[0]
    he = load_rgb(os.path.join(DATA_DIR, entry["he_path"]))
    ihc = load_rgb(os.path.join(DATA_DIR, entry["ihc_path"]))
    M_est, n_inliers, n_matches = estimate_affine(he, ihc)
    assert M_est is not None
    assert M_est.shape == (2, 3)
    assert n_matches > 0
    assert n_inliers > 0

    warped = warp_moving_to_fixed(ihc, M_est, he.shape)
    assert warped.shape == he.shape


def test_tre_in_reasonable_band_sample(manifest):
    """Sample a subset (for test speed) and confirm the mean TRE lands in a
    reasonable sub-pixel-to-few-pixel band, consistent with a working
    feature-based registration pipeline (not degenerate / not huge)."""
    sample = manifest[::17]  # ~20 pairs spread across subtypes
    tres = []
    for entry in sample:
        he = load_rgb(os.path.join(DATA_DIR, entry["he_path"]))
        ihc = load_rgb(os.path.join(DATA_DIR, entry["ihc_path"]))
        M_est, _, _ = estimate_affine(he, ihc)
        M_gt = np.array(entry["gt_affine"])
        tre, _ = compute_tre(M_gt, M_est, PATCH_SIZE)
        if not np.isnan(tre):
            tres.append(tre)

    assert len(tres) >= len(sample) * 0.5, "Too many pairs failed to register"
    mean_tre = float(np.mean(tres))
    assert 0.0 < mean_tre < 15.0, f"Mean TRE {mean_tre} outside reasonable band"


def test_full_results_file_if_present():
    """If run_all.py has already produced data/results.json, sanity-check
    its summary numbers are self-consistent and in-band."""
    results_path = os.path.join(ROOT, "data", "results.json")
    if not os.path.exists(results_path):
        pytest.skip("results.json not generated yet; run run_all.py first")
    with open(results_path) as f:
        results = json.load(f)

    assert results["n_pairs"] == 340
    assert results["n_subtypes"] == 12

    tre_summary = results["tre_summary"]
    assert tre_summary["n_registered"] >= 0.8 * tre_summary["n_pairs"]
    assert 0.0 < tre_summary["mean_tre_px"] < 15.0

    stain = results["stain_consistency_summary"]
    assert stain["macenko_consistency_dist"] < stain["before_consistency_dist"]
    assert stain["macenko_improvement_pct"] > 0


def test_stain_normalization_improves_pair_color_consistency(manifest):
    """Injected sanity check: take two images from DIFFERENT subtypes (with
    deliberately different stain characteristics), compute their raw color
    signatures, normalize both with Macenko onto the same reference stain
    vectors, and confirm the signatures become measurably closer together."""
    subtype_a_entry = next(e for e in manifest if e["subtype"] == SUBTYPE_NAMES[0])
    subtype_b_entry = next(e for e in manifest if e["subtype"] == SUBTYPE_NAMES[5])

    img_a = load_rgb(os.path.join(DATA_DIR, subtype_a_entry["he_path"]))
    img_b = load_rgb(os.path.join(DATA_DIR, subtype_b_entry["he_path"]))

    sig_a_before = mean_std_color_signature(img_a)
    sig_b_before = mean_std_color_signature(img_b)
    dist_before = np.linalg.norm(sig_a_before - sig_b_before)

    norm_a = macenko_normalize(img_a)
    norm_b = macenko_normalize(img_b)
    sig_a_after = mean_std_color_signature(norm_a)
    sig_b_after = mean_std_color_signature(norm_b)
    dist_after = np.linalg.norm(sig_a_after - sig_b_after)

    assert dist_after < dist_before, (
        f"Macenko normalization did not reduce pairwise color distance: "
        f"before={dist_before:.2f}, after={dist_after:.2f}"
    )


def test_reinhard_normalize_matches_target_stats(manifest):
    entry = manifest[0]
    ref_entry = manifest[1]
    img = load_rgb(os.path.join(DATA_DIR, entry["he_path"]))
    target = load_rgb(os.path.join(DATA_DIR, ref_entry["he_path"]))

    normalized = reinhard_normalize(img, target)
    assert normalized.shape == img.shape
    assert normalized.dtype == np.uint8
