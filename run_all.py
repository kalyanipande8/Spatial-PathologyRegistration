"""
End-to-end pipeline runner:

  1. Simulate 340 paired H&E/IHC patches across 12 synthetic subtypes
     (if not already present).
  2. Run feature-based registration (ORB + RANSAC affine) on all pairs.
  3. Compute real Target Registration Error (TRE) per pair using the
     known ground-truth transform.
  4. Run Macenko and Reinhard stain normalization across all 12 subtypes
     and measure cross-subtype color-consistency before vs. after.
  5. Save example before/after figures and a results.json summary.
"""
import json
import os
import random

import cv2
import numpy as np

from src.simulate import generate_dataset, PATCH_SIZE, N_PAIRS, N_SUBTYPES, SUBTYPE_NAMES
from src.register import estimate_affine
from src.metrics import compute_tre, invert_affine
from src.stain_norm import (
    macenko_normalize, reinhard_normalize,
    mean_std_color_signature, color_consistency_distance,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "patches")
FIG_DIR = os.path.join(HERE, "figures")
RESULTS_PATH = os.path.join(HERE, "data", "results.json")


def load_rgb(path):
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ensure_dataset():
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if len(manifest) == N_PAIRS:
            print(f"Found existing dataset with {len(manifest)} pairs.")
            return manifest
    print("Generating synthetic dataset...")
    return generate_dataset(DATA_DIR)


def run_registration(manifest):
    print(f"Running registration on {len(manifest)} pairs...")
    tre_results = []
    for i, entry in enumerate(manifest):
        he = load_rgb(os.path.join(DATA_DIR, entry["he_path"]))
        ihc = load_rgb(os.path.join(DATA_DIR, entry["ihc_path"]))

        M_est_ihc_to_he, n_inliers, n_matches = estimate_affine(he, ihc)
        M_gt_he_to_ihc = np.array(entry["gt_affine"])

        tre_mean, _ = compute_tre(M_gt_he_to_ihc, M_est_ihc_to_he, PATCH_SIZE)

        tre_results.append({
            "pair_id": entry["pair_id"],
            "subtype": entry["subtype"],
            "tre": tre_mean,
            "n_matches": n_matches,
            "n_inliers": n_inliers,
            "registered": M_est_ihc_to_he is not None,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(manifest)} pairs processed")

    return tre_results


def summarize_tre(tre_results):
    valid = [r["tre"] for r in tre_results if r["registered"] and not np.isnan(r["tre"])]
    n_failed = sum(1 for r in tre_results if not r["registered"])
    return {
        "n_pairs": len(tre_results),
        "n_registered": len(valid),
        "n_failed": n_failed,
        "mean_tre_px": float(np.mean(valid)) if valid else None,
        "median_tre_px": float(np.median(valid)) if valid else None,
        "std_tre_px": float(np.std(valid)) if valid else None,
        "p90_tre_px": float(np.percentile(valid, 90)) if valid else None,
    }


def run_stain_normalization(manifest):
    print("Running stain normalization across subtypes...")
    # Build a reference image for Reinhard (first HE image of first subtype).
    ref_entry = next(e for e in manifest if e["subtype"] == SUBTYPE_NAMES[0])
    reference_img = load_rgb(os.path.join(DATA_DIR, ref_entry["he_path"]))

    subtype_to_entries = {s: [] for s in SUBTYPE_NAMES}
    for e in manifest:
        subtype_to_entries[e["subtype"]].append(e)

    # Sample one representative HE image per subtype for the consistency metric.
    raw_sigs, macenko_sigs, reinhard_sigs = [], [], []
    sample_cache = {}
    for subtype, entries in subtype_to_entries.items():
        entry = entries[0]
        img = load_rgb(os.path.join(DATA_DIR, entry["he_path"]))
        sample_cache[subtype] = img

        raw_sigs.append(mean_std_color_signature(img))

        mac_img = macenko_normalize(img)
        macenko_sigs.append(mean_std_color_signature(mac_img))

        rein_img = reinhard_normalize(img, reference_img)
        reinhard_sigs.append(mean_std_color_signature(rein_img))

    before_consistency = color_consistency_distance(raw_sigs)
    macenko_consistency = color_consistency_distance(macenko_sigs)
    reinhard_consistency = color_consistency_distance(reinhard_sigs)

    return {
        "before_consistency_dist": before_consistency,
        "macenko_consistency_dist": macenko_consistency,
        "reinhard_consistency_dist": reinhard_consistency,
        "macenko_improvement_pct": 100.0 * (before_consistency - macenko_consistency) / before_consistency,
        "reinhard_improvement_pct": 100.0 * (before_consistency - reinhard_consistency) / before_consistency,
    }, sample_cache, reference_img


def make_figures(manifest, tre_results, sample_cache, reference_img):
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Registration before/after figure ---
    best_entry = manifest[0]
    he = load_rgb(os.path.join(DATA_DIR, best_entry["he_path"]))
    ihc = load_rgb(os.path.join(DATA_DIR, best_entry["ihc_path"]))
    M_est, _, _ = estimate_affine(he, ihc)
    from src.register import warp_moving_to_fixed
    ihc_registered = warp_moving_to_fixed(ihc, M_est, he.shape)

    def overlay(a, b):
        a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        out = np.zeros((*a_gray.shape, 3), dtype=np.uint8)
        out[..., 0] = a_gray  # red = fixed (HE)
        out[..., 1] = b_gray  # green = moving (IHC)
        return out

    before_overlay = overlay(he, ihc)
    after_overlay = overlay(he, ihc_registered)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(he); axes[0].set_title("Fixed: H&E"); axes[0].axis("off")
    axes[1].imshow(ihc); axes[1].set_title("Moving: IHC (original)"); axes[1].axis("off")
    axes[2].imshow(before_overlay); axes[2].set_title("Overlay BEFORE registration\n(red=HE, green=IHC)"); axes[2].axis("off")
    axes[3].imshow(after_overlay); axes[3].set_title("Overlay AFTER registration\n(ORB+RANSAC affine)"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "registration_before_after.png"), dpi=130)
    plt.close(fig)

    # --- Stain normalization before/after figure ---
    subtypes_sample = list(sample_cache.keys())[:4]
    fig, axes = plt.subplots(3, len(subtypes_sample), figsize=(4 * len(subtypes_sample), 10))
    for col, subtype in enumerate(subtypes_sample):
        img = sample_cache[subtype]
        mac = macenko_normalize(img)
        rein = reinhard_normalize(img, reference_img)
        axes[0, col].imshow(img); axes[0, col].set_title(f"{subtype}\n(raw)"); axes[0, col].axis("off")
        axes[1, col].imshow(mac); axes[1, col].set_title("Macenko normalized"); axes[1, col].axis("off")
        axes[2, col].imshow(rein); axes[2, col].set_title("Reinhard normalized"); axes[2, col].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stain_normalization_before_after.png"), dpi=130)
    plt.close(fig)

    # --- TRE distribution figure ---
    tres = [r["tre"] for r in tre_results if r["registered"] and not np.isnan(r["tre"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(tres, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(np.mean(tres), color="red", linestyle="--", label=f"mean = {np.mean(tres):.2f} px")
    ax.set_xlabel("Target Registration Error (pixels)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("TRE distribution across 340 registered pairs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tre_distribution.png"), dpi=130)
    plt.close(fig)

    print(f"Figures saved to {FIG_DIR}")


def main():
    random.seed(42)
    np.random.seed(42)

    manifest = ensure_dataset()
    assert len(manifest) == N_PAIRS

    tre_results = run_registration(manifest)
    tre_summary = summarize_tre(tre_results)

    stain_summary, sample_cache, reference_img = run_stain_normalization(manifest)

    results = {
        "n_pairs": len(manifest),
        "n_subtypes": len(SUBTYPE_NAMES),
        "subtypes": SUBTYPE_NAMES,
        "tre_summary": tre_summary,
        "stain_consistency_summary": stain_summary,
        "per_pair_tre": tre_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    make_figures(manifest, tre_results, sample_cache, reference_img)

    print("\n=== RESULTS SUMMARY ===")
    print(json.dumps({"tre_summary": tre_summary, "stain_consistency_summary": stain_summary}, indent=2))


if __name__ == "__main__":
    main()
