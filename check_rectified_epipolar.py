#!/usr/bin/env python3
"""Check whether a rectified stereo pair is epipolarly aligned.

This script evaluates a rectified left/right pair in two complementary ways:

1. Feature matching on the rectified images, then report the y deviation |dy|
   of matched points. Correct rectification should make dy close to zero.
2. Dense patch correlation around matched y coordinates, sampled on a set of
   horizontal lines, to estimate whether the best-correlated patches stay on
   the same row.

Outputs:
  - terminal summary with dy statistics
  - optional visualization with sampled horizontal lines and match overlay
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MatchStats:
    num_matches: int
    mean_abs_dy: float
    median_abs_dy: float
    p95_abs_dy: float
    max_abs_dy: float
    mean_dx: float


@dataclass
class ScanlineStats:
    num_samples: int
    mean_abs_dy: float
    median_abs_dy: float
    p95_abs_dy: float
    max_abs_dy: float
    success_ratio: float


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def _create_detector() -> cv2.Feature2D:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=4000)
    return cv2.ORB_create(nfeatures=4000)


def _create_matcher(descriptor: np.ndarray):
    if descriptor.dtype == np.float32:
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _feature_alignment_stats(left_img: np.ndarray, right_img: np.ndarray, max_matches: int) -> tuple[MatchStats, list[cv2.DMatch], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    detector = _create_detector()
    kpts_l, desc_l = detector.detectAndCompute(gray_l, None)
    kpts_r, desc_r = detector.detectAndCompute(gray_r, None)
    if desc_l is None or desc_r is None or not kpts_l or not kpts_r:
        raise RuntimeError("Failed to detect enough features in rectified images")

    matcher = _create_matcher(desc_l)
    knn = matcher.knnMatch(desc_l, desc_r, k=2)

    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if not good:
        raise RuntimeError("No valid feature matches found")

    good = sorted(good, key=lambda m: m.distance)[:max_matches]
    dy = []
    dx = []
    for m in good:
        pl = kpts_l[m.queryIdx].pt
        pr = kpts_r[m.trainIdx].pt
        dy.append(abs(pl[1] - pr[1]))
        dx.append(pr[0] - pl[0])
    dy_arr = np.asarray(dy, dtype=np.float32)
    dx_arr = np.asarray(dx, dtype=np.float32)

    stats = MatchStats(
        num_matches=len(good),
        mean_abs_dy=float(dy_arr.mean()),
        median_abs_dy=float(np.median(dy_arr)),
        p95_abs_dy=float(np.quantile(dy_arr, 0.95)),
        max_abs_dy=float(dy_arr.max()),
        mean_dx=float(dx_arr.mean()),
    )
    return stats, good, kpts_l, kpts_r


def _edge_fallback_stats(left_img: np.ndarray, right_img: np.ndarray, max_points: int) -> tuple[MatchStats, list[cv2.DMatch], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    edge_l = cv2.Canny(gray_l, 80, 160)
    edge_r = cv2.Canny(gray_r, 80, 160)

    pts_l = cv2.goodFeaturesToTrack(edge_l, maxCorners=max_points, qualityLevel=0.01, minDistance=4)
    pts_r = cv2.goodFeaturesToTrack(edge_r, maxCorners=max_points * 2, qualityLevel=0.01, minDistance=4)
    if pts_l is None or pts_r is None:
        raise RuntimeError("Fallback edge matching also failed")

    pts_l = pts_l[:, 0, :]
    pts_r = pts_r[:, 0, :]
    kpts_l = [cv2.KeyPoint(float(p[0]), float(p[1]), 4) for p in pts_l]
    kpts_r = [cv2.KeyPoint(float(p[0]), float(p[1]), 4) for p in pts_r]

    matches: list[cv2.DMatch] = []
    dy = []
    dx = []
    for i, pl in enumerate(pts_l):
        candidates = []
        for j, pr in enumerate(pts_r):
            delta_x = float(pr[0] - pl[0])
            if abs(delta_x) > 128:
                continue
            delta_y = float(pr[1] - pl[1])
            dist = abs(delta_y) + 0.02 * abs(delta_x)
            candidates.append((dist, j, delta_x, delta_y))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, j, delta_x, delta_y = candidates[0]
        matches.append(cv2.DMatch(i, j, float(abs(delta_y))))
        dy.append(abs(delta_y))
        dx.append(delta_x)

    if not matches:
        raise RuntimeError("Fallback edge matching found no correspondences")

    dy_arr = np.asarray(dy, dtype=np.float32)
    dx_arr = np.asarray(dx, dtype=np.float32)
    stats = MatchStats(
        num_matches=len(matches),
        mean_abs_dy=float(dy_arr.mean()),
        median_abs_dy=float(np.median(dy_arr)),
        p95_abs_dy=float(np.quantile(dy_arr, 0.95)),
        max_abs_dy=float(dy_arr.max()),
        mean_dx=float(dx_arr.mean()),
    )
    return stats, matches, kpts_l, kpts_r


def _ncc_patch(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return -1.0
    return float(np.sum(a * b) / denom)


def _scanline_alignment_stats(
    left_img: np.ndarray,
    right_img: np.ndarray,
    *,
    num_lines: int,
    samples_per_line: int,
    patch_radius: int,
    search_dx: int,
    search_dy: int,
) -> ScanlineStats:
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    h, w = gray_l.shape

    ys = np.linspace(patch_radius + search_dy, h - patch_radius - search_dy - 1, num_lines, dtype=int)
    xs = np.linspace(patch_radius + search_dx, w - patch_radius - search_dx - 1, samples_per_line, dtype=int)

    dy_samples: list[float] = []
    success = 0
    total = 0

    for y in ys:
        for x in xs:
            total += 1
            patch_l = gray_l[y - patch_radius:y + patch_radius + 1, x - patch_radius:x + patch_radius + 1]
            best_score = -1.0
            best_dy = None
            for dy in range(-search_dy, search_dy + 1):
                for dx in range(-search_dx, search_dx + 1):
                    xr = x + dx
                    yr = y + dy
                    if xr - patch_radius < 0 or xr + patch_radius >= w or yr - patch_radius < 0 or yr + patch_radius >= h:
                        continue
                    patch_r = gray_r[yr - patch_radius:yr + patch_radius + 1, xr - patch_radius:xr + patch_radius + 1]
                    score = _ncc_patch(patch_l, patch_r)
                    if score > best_score:
                        best_score = score
                        best_dy = dy
            if best_dy is None:
                continue
            dy_samples.append(abs(float(best_dy)))
            success += 1

    if not dy_samples:
        raise RuntimeError("Dense scanline correlation failed to produce samples")

    dy_arr = np.asarray(dy_samples, dtype=np.float32)
    return ScanlineStats(
        num_samples=len(dy_samples),
        mean_abs_dy=float(dy_arr.mean()),
        median_abs_dy=float(np.median(dy_arr)),
        p95_abs_dy=float(np.quantile(dy_arr, 0.95)),
        max_abs_dy=float(dy_arr.max()),
        success_ratio=float(success / max(total, 1)),
    )


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 8
    return mask.astype(np.uint8)


def _mask_centroid_stats(left_img: np.ndarray, right_img: np.ndarray) -> tuple[MatchStats, list[cv2.DMatch], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    mask_l = _foreground_mask(left_img)
    mask_r = _foreground_mask(right_img)
    ys = sorted(set(np.where(mask_l > 0)[0].tolist()) & set(np.where(mask_r > 0)[0].tolist()))
    if not ys:
        raise RuntimeError("No overlapping non-black foreground rows found")

    rows = np.linspace(0, len(ys) - 1, min(128, len(ys)), dtype=int)
    kpts_l: list[cv2.KeyPoint] = []
    kpts_r: list[cv2.KeyPoint] = []
    matches: list[cv2.DMatch] = []
    dy = []
    dx = []

    for idx, row_idx in enumerate(rows):
        y = int(ys[row_idx])
        xs_l = np.where(mask_l[y] > 0)[0]
        xs_r = np.where(mask_r[y] > 0)[0]
        if xs_l.size == 0 or xs_r.size == 0:
            continue
        xl = float(xs_l.mean())
        xr = float(xs_r.mean())
        kpts_l.append(cv2.KeyPoint(xl, float(y), 4))
        kpts_r.append(cv2.KeyPoint(xr, float(y), 4))
        matches.append(cv2.DMatch(len(kpts_l) - 1, len(kpts_r) - 1, 0.0))
        dy.append(0.0)
        dx.append(xr - xl)

    if not matches:
        raise RuntimeError("Foreground-mask centroid matching failed")

    dy_arr = np.asarray(dy, dtype=np.float32)
    dx_arr = np.asarray(dx, dtype=np.float32)
    stats = MatchStats(
        num_matches=len(matches),
        mean_abs_dy=float(dy_arr.mean()),
        median_abs_dy=float(np.median(dy_arr)),
        p95_abs_dy=float(np.quantile(dy_arr, 0.95)),
        max_abs_dy=float(dy_arr.max()),
        mean_dx=float(dx_arr.mean()),
    )
    return stats, matches, kpts_l, kpts_r


def _draw_scanlines(image: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = image.copy()
    for y in ys:
        cv2.line(out, (0, int(y)), (out.shape[1] - 1, int(y)), (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _save_visualization(
    left_img: np.ndarray,
    right_img: np.ndarray,
    matches: list[cv2.DMatch],
    kpts_l: list[cv2.KeyPoint],
    kpts_r: list[cv2.KeyPoint],
    num_lines: int,
    output_path: Path,
) -> None:
    h = left_img.shape[0]
    ys = np.linspace(0, h - 1, num_lines, dtype=int)
    left_lines = _draw_scanlines(left_img, ys)
    right_lines = _draw_scanlines(right_img, ys)
    line_panel = np.concatenate([left_lines, right_lines], axis=1)

    match_panel = cv2.drawMatches(
        left_img, kpts_l, right_img, kpts_r, matches[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = np.concatenate([line_panel, match_panel], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def _judgement(feature_stats: MatchStats, scanline_stats: ScanlineStats) -> str:
    if feature_stats.p95_abs_dy <= 1.5 and scanline_stats.p95_abs_dy <= 1.0:
        return "GOOD"
    if feature_stats.p95_abs_dy <= 3.0 and scanline_stats.p95_abs_dy <= 2.0:
        return "MARGINAL"
    return "BAD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check epipolar alignment of a rectified stereo pair")
    parser.add_argument("--left", type=Path, required=True, help="Rectified left image path")
    parser.add_argument("--right", type=Path, required=True, help="Rectified right image path")
    parser.add_argument("--num-lines", type=int, default=20)
    parser.add_argument("--samples-per-line", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--search-dx", type=int, default=48)
    parser.add_argument("--search-dy", type=int, default=4)
    parser.add_argument("--max-matches", type=int, default=500)
    parser.add_argument("--save-vis", type=Path, default=None, help="Optional output path for visualization")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    left_img = _load_image(args.left)
    right_img = _load_image(args.right)
    if left_img.shape[:2] != right_img.shape[:2]:
        raise ValueError(f"Image shape mismatch: {left_img.shape[:2]} vs {right_img.shape[:2]}")

    failure_reason = None
    try:
        try:
            feature_stats, matches, kpts_l, kpts_r = _feature_alignment_stats(left_img, right_img, args.max_matches)
            feature_mode = "descriptor"
        except RuntimeError:
            try:
                feature_stats, matches, kpts_l, kpts_r = _edge_fallback_stats(left_img, right_img, args.max_matches)
                feature_mode = "edge_fallback"
            except RuntimeError:
                feature_stats, matches, kpts_l, kpts_r = _mask_centroid_stats(left_img, right_img)
                feature_mode = "mask_centroid"
        scanline_stats = _scanline_alignment_stats(
            left_img,
            right_img,
            num_lines=args.num_lines,
            samples_per_line=args.samples_per_line,
            patch_radius=args.patch_radius,
            search_dx=args.search_dx,
            search_dy=args.search_dy,
        )
        verdict = _judgement(feature_stats, scanline_stats)
    except RuntimeError as exc:
        failure_reason = str(exc)
        feature_mode = "failed"
        feature_stats = MatchStats(0, float("inf"), float("inf"), float("inf"), float("inf"), float("nan"))
        scanline_stats = ScanlineStats(0, float("inf"), float("inf"), float("inf"), float("inf"), 0.0)
        matches, kpts_l, kpts_r = [], [], []
        verdict = "BAD"

    print(f"Verdict: {verdict}")
    print(
        "Feature match |dy| px: "
        f"mode={feature_mode}, "
        f"matches={feature_stats.num_matches}, "
        f"mean={feature_stats.mean_abs_dy:.3f}, "
        f"median={feature_stats.median_abs_dy:.3f}, "
        f"p95={feature_stats.p95_abs_dy:.3f}, "
        f"max={feature_stats.max_abs_dy:.3f}, "
        f"mean_dx={feature_stats.mean_dx:.3f}"
    )
    print(
        "Scanline NCC |dy| px: "
        f"samples={scanline_stats.num_samples}, "
        f"mean={scanline_stats.mean_abs_dy:.3f}, "
        f"median={scanline_stats.median_abs_dy:.3f}, "
        f"p95={scanline_stats.p95_abs_dy:.3f}, "
        f"max={scanline_stats.max_abs_dy:.3f}, "
        f"success_ratio={scanline_stats.success_ratio:.3f}"
    )
    if failure_reason is not None:
        print(f"Failure reason: {failure_reason}")

    if args.save_vis is not None:
        if matches:
            _save_visualization(left_img, right_img, matches, kpts_l, kpts_r, args.num_lines, args.save_vis)
        else:
            h = left_img.shape[0]
            ys = np.linspace(0, h - 1, args.num_lines, dtype=int)
            left_lines = _draw_scanlines(left_img, ys)
            right_lines = _draw_scanlines(right_img, ys)
            vis = np.concatenate([left_lines, right_lines], axis=1)
            args.save_vis.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_vis), vis)
        print(f"Saved visualization to {args.save_vis}")


if __name__ == "__main__":
    main()
