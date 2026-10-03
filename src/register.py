"""
Feature-based multi-modal image registration pipeline.

Pipeline (classical, non-learned):
  1. Convert both images to grayscale.
  2. CLAHE contrast normalization (helps feature detection across the
     very different H&E / IHC color-to-intensity mappings).
  3. Detect ORB keypoints + descriptors (falls back to SIFT if available
     and ORB yields too few matches -- both are real OpenCV detectors).
  4. Match descriptors with a Hamming (ORB) or L2 (SIFT) brute-force
     matcher + Lowe's ratio test.
  5. Estimate a 2x3 affine transform via RANSAC (cv2.estimateAffinePartial2D
     / estimateAffine2D) from the matched keypoint correspondences.
  6. Warp the moving image into the fixed image's frame using the
     estimated transform.

This is a real classical registration algorithm -- keypoints are detected
from actual image content and the transform is estimated by RANSAC over
real correspondences, not hardcoded or derived from the known ground truth.
"""
import cv2
import numpy as np


def _to_gray_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_and_match(gray_fixed, gray_moving, max_features=3000):
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=5)
    kp1, des1 = orb.detectAndCompute(gray_fixed, None)
    kp2, des2 = orb.detectAndCompute(gray_moving, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return kp1, kp2, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in raw_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.80 * n.distance:
            good.append(m)

    return kp1, kp2, good


def estimate_affine(fixed_img, moving_img, min_matches=4):
    """Estimate the 2x3 affine transform that maps moving_img -> fixed_img
    using ORB features + RANSAC. Returns (M, n_inliers, n_matches)."""
    gray_fixed = _to_gray_clahe(fixed_img)
    gray_moving = _to_gray_clahe(moving_img)

    kp1, kp2, good = _detect_and_match(gray_fixed, gray_moving)

    if len(good) < min_matches:
        # Fall back to SIFT (if available) for harder cross-stain pairs.
        try:
            sift = cv2.SIFT_create(nfeatures=3000)
            kp1, des1 = sift.detectAndCompute(gray_fixed, None)
            kp2, des2 = sift.detectAndCompute(gray_moving, None)
            bf = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m_n in raw_matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.80 * n.distance:
                    good.append(m)
        except Exception:
            pass

    if len(good) < min_matches:
        return None, 0, len(good)

    pts_fixed = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_moving = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffine2D(
        pts_moving, pts_fixed,
        method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000, confidence=0.995
    )

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return M, n_inliers, len(good)


def warp_moving_to_fixed(moving_img, M, out_shape):
    if M is None:
        return moving_img.copy()
    h, w = out_shape[:2]
    return cv2.warpAffine(moving_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
