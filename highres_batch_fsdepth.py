#!/usr/bin/env python3
"""Batch FoundationStereo depth generation for DNA-Rendering highres views.

This script targets the 48-58 high-resolution view group and tries to find a
usable stereo pairing/configuration for each left view by searching:
  - right view within {48, 50, 52, 54, 56, 58}
  - RT convention: c2w / w2c
  - distortion usage: assume-undistorted / use D
  - stereoRectify alpha

Unlike the earlier pipeline, it does not trust OpenCV's ROI rectangles alone.
It computes overlap from the actual rectification maps and only runs inference
when the rectified pair has enough valid pixel overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from foundationstereo_model import (  # noqa: E402
    disparity_to_depth,
    load_camera_npz,
    load_model,
    predict_disparity,
)


HIGHRES_VIEWS = [f"{i:02d}" for i in range(48, 60, 2)]


@dataclass
class RectifyPlan:
    left_view: str
    right_view: str
    frame_id: str
    rt_type: str
    assume_undistorted: bool
    alpha: float
    match_size: str
    rotate_vertical: bool
    crop_to_overlap: bool
    overlap_ratio: float
    valid_left_ratio: float
    valid_right_ratio: float
    baseline_m: float
    rectification_mode: str
    rotation: str
    expected_disp: float
    score: float


def _ensure_3ch(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _to_4x4(rt: np.ndarray) -> np.ndarray:
    if rt.shape == (4, 4):
        return rt.astype(np.float64)
    if rt.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = rt.astype(np.float64)
        return out
    raise ValueError(f"Unsupported RT shape: {rt.shape}")


def _w2c_from_rt(rt: np.ndarray, rt_type: str) -> np.ndarray:
    rt4 = _to_4x4(rt)
    if rt_type == "w2c":
        return rt4
    if rt_type == "c2w":
        return np.linalg.inv(rt4)
    raise ValueError(f"Unknown rt_type: {rt_type}")


def _candidate_right_views(left_view: str) -> list[str]:
    idx = HIGHRES_VIEWS.index(left_view)
    candidates: list[str] = []
    for step in range(1, len(HIGHRES_VIEWS)):
        hi = idx + step
        lo = idx - step
        if hi < len(HIGHRES_VIEWS):
            candidates.append(HIGHRES_VIEWS[hi])
        if lo >= 0:
            candidates.append(HIGHRES_VIEWS[lo])
    return candidates


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _rotate_k_rect(k_rect: np.ndarray, width: int, height: int, rotation: str) -> np.ndarray:
    fx = float(k_rect[0, 0])
    fy = float(k_rect[1, 1])
    cx = float(k_rect[0, 2])
    cy = float(k_rect[1, 2])
    if rotation == "ccw":
        return np.array(
            [[fy, 0.0, cy],
             [0.0, fx, width - 1.0 - cx],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if rotation == "cw":
        return np.array(
            [[fy, 0.0, height - 1.0 - cy],
             [0.0, fx, cx],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown rotation: {rotation}")


def _colorize_scalar_map(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not valid_mask.any():
        return vis
    valid = values[valid_mask]
    lo = float(np.percentile(valid, 5))
    hi = float(np.percentile(valid, 95))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    vis[~valid_mask] = 0
    return vis


def _rectify_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_cam: dict[str, np.ndarray],
    right_cam: dict[str, np.ndarray],
    *,
    rt_type: str,
    assume_undistorted: bool,
    alpha: float,
    match_size: str,
    rotate_vertical: bool,
    crop_to_overlap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    left = _ensure_3ch(left_img)
    right = _ensure_3ch(right_img)

    kl = left_cam["K"].astype(np.float64).copy()
    kr = right_cam["K"].astype(np.float64).copy()
    dl = left_cam["D"].astype(np.float64).reshape(-1).copy()
    dr = right_cam["D"].astype(np.float64).reshape(-1).copy()

    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    if (h_l, w_l) != (h_r, w_r):
        if match_size == "left":
            sx = w_l / float(w_r)
            sy = h_l / float(h_r)
            kr[0, :] *= sx
            kr[1, :] *= sy
            right = cv2.resize(right, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
        elif match_size == "right":
            sx = w_r / float(w_l)
            sy = h_r / float(h_l)
            kl[0, :] *= sx
            kl[1, :] *= sy
            left = cv2.resize(left, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unknown match_size: {match_size}")

    if assume_undistorted:
        dl = np.zeros_like(dl)
        dr = np.zeros_like(dr)

    w2c_l = _w2c_from_rt(left_cam["RT"], rt_type)
    w2c_r = _w2c_from_rt(right_cam["RT"], rt_type)
    rl, tl = w2c_l[:3, :3], w2c_l[:3, 3]
    rr, tr = w2c_r[:3, :3], w2c_r[:3, 3]
    r = rr @ rl.T
    t = tr - r @ tl

    image_size = (left.shape[1], left.shape[0])
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        kl, dl, kr, dr, image_size, r, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    map1x, map1y = cv2.initUndistortRectifyMap(kl, dl, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(kr, dr, r2, p2, image_size, cv2.CV_32FC1)

    left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    valid_left = (map1x >= 0.0) & (map1x <= image_size[0] - 1) & (map1y >= 0.0) & (map1y <= image_size[1] - 1)
    valid_right = (map2x >= 0.0) & (map2x <= image_size[0] - 1) & (map2y >= 0.0) & (map2y <= image_size[1] - 1)
    overlap_mask = valid_left & valid_right

    overlap_ratio = float(overlap_mask.mean())
    valid_left_ratio = float(valid_left.mean())
    valid_right_ratio = float(valid_right.mean())

    horizontal = abs(float(p2[0, 3])) >= abs(float(p2[1, 3]))
    baseline_x = abs(float(p2[0, 3]) / float(p2[0, 0])) if abs(float(p2[0, 0])) > 1e-9 else 0.0
    baseline_y = abs(float(p2[1, 3]) / float(p2[1, 1])) if abs(float(p2[1, 1])) > 1e-9 else 0.0
    baseline_m = baseline_x if horizontal else baseline_y
    k_rect = p1[:3, :3].copy()
    rectification_mode = "horizontal" if horizontal else "vertical"
    rotation = "none"

    if not horizontal:
        if not rotate_vertical:
            raise ValueError("Vertical rectification encountered with rotate_vertical disabled.")
        rect_h, rect_w = left_rect.shape[:2]
        if float(p2[1, 3]) < 0:
            left_rect = cv2.rotate(left_rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
            right_rect = cv2.rotate(right_rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
            valid_left = cv2.rotate(valid_left.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
            valid_right = cv2.rotate(valid_right.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
            overlap_mask = cv2.rotate(overlap_mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
            rotation = "ccw"
        else:
            left_rect = cv2.rotate(left_rect, cv2.ROTATE_90_CLOCKWISE)
            right_rect = cv2.rotate(right_rect, cv2.ROTATE_90_CLOCKWISE)
            valid_left = cv2.rotate(valid_left.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
            valid_right = cv2.rotate(valid_right.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
            overlap_mask = cv2.rotate(overlap_mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
            rotation = "cw"
        k_rect = _rotate_k_rect(k_rect, rect_w, rect_h, rotation)
        rectification_mode = "vertical_rotated"
        baseline_m = baseline_y

    if crop_to_overlap:
        bbox = _bbox_from_mask(overlap_mask)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            left_rect = left_rect[y0:y1, x0:x1]
            right_rect = right_rect[y0:y1, x0:x1]
            valid_left = valid_left[y0:y1, x0:x1]
            valid_right = valid_right[y0:y1, x0:x1]
            overlap_mask = overlap_mask[y0:y1, x0:x1]
            k_rect = k_rect.copy()
            k_rect[0, 2] -= float(x0)
            k_rect[1, 2] -= float(y0)

    meta = {
        "R": r,
        "T": t,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "roi1": np.array(roi1, dtype=np.int32),
        "roi2": np.array(roi2, dtype=np.int32),
        "baseline_x": float(baseline_x),
        "baseline_y": float(baseline_y),
        "baseline_m": float(baseline_m),
        "valid_left_ratio": valid_left_ratio,
        "valid_right_ratio": valid_right_ratio,
        "overlap_ratio": overlap_ratio,
        "horizontal": bool(horizontal),
        "rectification_mode": rectification_mode,
        "rotation": rotation,
        "K_rect": k_rect,
        "overlap_bbox": _bbox_from_mask(overlap_mask),
    }
    return left_rect, right_rect, k_rect, float(baseline_m), meta


def _score_candidate(
    *,
    overlap_ratio: float,
    valid_left_ratio: float,
    valid_right_ratio: float,
    expected_disp: float,
    right_view: str,
    left_view: str,
    min_overlap: float,
    min_valid_ratio: float,
    disp_range: tuple[float, float],
    target_disp: float,
) -> float | None:
    if overlap_ratio < min_overlap:
        return None
    if valid_left_ratio < min_valid_ratio or valid_right_ratio < min_valid_ratio:
        return None
    if not np.isfinite(expected_disp) or expected_disp <= 0.0:
        return None

    disp_min, disp_max = disp_range
    range_penalty = 0.0
    if expected_disp < disp_min:
        range_penalty = (disp_min - expected_disp) / max(target_disp, 1e-6)
    elif expected_disp > disp_max:
        range_penalty = (expected_disp - disp_max) / max(target_disp, 1e-6)

    disp_penalty = abs(math.log(max(expected_disp, 1e-6) / max(target_disp, 1e-6)))
    gap_penalty = abs(int(right_view) - int(left_view)) / 10.0

    return range_penalty * 10.0 + disp_penalty * 0.6 + gap_penalty * 0.05 - overlap_ratio * 1.2


def _search_best_plan_for_frame(
    seq_dir: Path,
    left_view: str,
    frame_id: str,
    args: argparse.Namespace,
) -> RectifyPlan | None:
    left_img_path = seq_dir / "images" / left_view / f"{frame_id}.png"
    left_cam_path = seq_dir / "cameras" / left_view / f"{frame_id}.npz"
    if not left_img_path.exists() or not left_cam_path.exists():
        return None

    left_img = cv2.imread(str(left_img_path), cv2.IMREAD_COLOR)
    left_cam = load_camera_npz(left_cam_path)
    if left_img is None:
        return None

    best: RectifyPlan | None = None

    for right_view in _candidate_right_views(left_view):
        right_img_path = seq_dir / "images" / right_view / f"{frame_id}.png"
        right_cam_path = seq_dir / "cameras" / right_view / f"{frame_id}.npz"
        if not right_img_path.exists() or not right_cam_path.exists():
            continue
        right_img = cv2.imread(str(right_img_path), cv2.IMREAD_COLOR)
        if right_img is None:
            continue
        right_cam = load_camera_npz(right_cam_path)

        for rt_type in args.rt_type_candidates:
            for assume_undistorted in args.assume_undistorted_candidates:
                for alpha in args.alpha_candidates:
                    try:
                        _, _, k_rect, baseline_m, meta = _rectify_pair(
                            left_img,
                            right_img,
                            left_cam,
                            right_cam,
                            rt_type=rt_type,
                            assume_undistorted=assume_undistorted,
                            alpha=alpha,
                            match_size=args.match_size,
                            rotate_vertical=args.rotate_vertical,
                            crop_to_overlap=args.crop_to_overlap,
                        )
                    except Exception:
                        continue

                    fx_scaled = float(k_rect[0, 0]) * float(args.inference_scale)
                    expected_disp = fx_scaled * float(baseline_m) / float(args.target_depth_m)
                    score = _score_candidate(
                        overlap_ratio=float(meta["overlap_ratio"]),
                        valid_left_ratio=float(meta["valid_left_ratio"]),
                        valid_right_ratio=float(meta["valid_right_ratio"]),
                        expected_disp=expected_disp,
                        right_view=right_view,
                        left_view=left_view,
                        min_overlap=args.min_overlap_ratio,
                        min_valid_ratio=args.min_valid_ratio,
                        disp_range=(args.min_expected_disp, args.max_expected_disp),
                        target_disp=args.target_disp,
                    )
                    if score is None:
                        continue

                    plan = RectifyPlan(
                        left_view=left_view,
                        right_view=right_view,
                        frame_id=frame_id,
                        rt_type=rt_type,
                        assume_undistorted=assume_undistorted,
                        alpha=float(alpha),
                        match_size=args.match_size,
                        rotate_vertical=bool(args.rotate_vertical),
                        crop_to_overlap=bool(args.crop_to_overlap),
                        overlap_ratio=float(meta["overlap_ratio"]),
                        valid_left_ratio=float(meta["valid_left_ratio"]),
                        valid_right_ratio=float(meta["valid_right_ratio"]),
                        baseline_m=float(baseline_m),
                        rectification_mode=str(meta["rectification_mode"]),
                        rotation=str(meta["rotation"]),
                        expected_disp=float(expected_disp),
                        score=float(score),
                    )
                    if best is None or plan.score < best.score:
                        best = plan

    return best


def _probe_plan_quality(
    model: Any,
    seq_dir: Path,
    plan: RectifyPlan,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    left_img = cv2.imread(str(seq_dir / "images" / plan.left_view / f"{plan.frame_id}.png"), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(seq_dir / "images" / plan.right_view / f"{plan.frame_id}.png"), cv2.IMREAD_COLOR)
    if left_img is None or right_img is None:
        return None
    left_cam = load_camera_npz(seq_dir / "cameras" / plan.left_view / f"{plan.frame_id}.npz")
    right_cam = load_camera_npz(seq_dir / "cameras" / plan.right_view / f"{plan.frame_id}.npz")
    try:
        left_rect, right_rect, k_rect, baseline_m, _meta = _rectify_pair(
            left_img,
            right_img,
            left_cam,
            right_cam,
            rt_type=plan.rt_type,
            assume_undistorted=plan.assume_undistorted,
            alpha=plan.alpha,
            match_size=plan.match_size,
            rotate_vertical=plan.rotate_vertical,
            crop_to_overlap=plan.crop_to_overlap,
        )
    except Exception:
        return None

    if args.inference_scale < 1.0:
        left_rect = cv2.resize(left_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
        right_rect = cv2.resize(right_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
        k_rect = k_rect.copy()
        k_rect[0, :] *= float(args.inference_scale)
        k_rect[1, :] *= float(args.inference_scale)

    disp = predict_disparity(
        left_rect,
        right_rect,
        model,
        device=args.device,
        valid_iters=args.valid_iters,
        input_color=args.input_color,
    )
    depth = disparity_to_depth(disp, k_rect, baseline_m)
    disp_valid = np.isfinite(disp) & (disp > 1e-4)
    depth_valid = np.isfinite(depth) & (depth > 0)
    if not disp_valid.any() or not depth_valid.any():
        return None

    disp_med = float(np.median(disp[disp_valid]))
    depth_med = float(np.median(depth[depth_valid]))
    depth_q95 = float(np.quantile(depth[depth_valid], 0.95))
    return {
        "disp_median": disp_med,
        "depth_median": depth_med,
        "depth_q95": depth_q95,
    }


def _first_shared_frame(seq_dir: Path, left_view: str, right_view: str) -> str | None:
    left_frames = {p.stem for p in (seq_dir / "images" / left_view).glob("*.png")}
    right_frames = {p.stem for p in (seq_dir / "images" / right_view).glob("*.png")}
    shared = sorted(left_frames & right_frames)
    return shared[0] if shared else None


def _save_outputs(
    seq_dir: Path,
    left_view: str,
    frame_id: str,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    disp: np.ndarray,
    depth: np.ndarray,
    k_rect: np.ndarray,
    plan: RectifyPlan,
    meta: dict[str, Any],
) -> None:
    rect_dir = seq_dir / "fs_rectified" / "highres" / left_view
    depth_dir = seq_dir / "fs_depth" / "highres" / left_view
    rect_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(rect_dir / f"{frame_id}_left.png"), left_rect)
    cv2.imwrite(str(rect_dir / f"{frame_id}_right.png"), right_rect)

    np.save(depth_dir / f"{frame_id}_depth.npy", depth.astype(np.float32))
    np.save(depth_dir / f"{frame_id}_disp.npy", disp.astype(np.float32))

    depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(depth_dir / f"{frame_id}_depth_mm.png"), depth_mm)

    depth_valid = np.isfinite(depth) & (depth > 0)
    disp_valid = np.isfinite(disp) & (disp > 1e-4)
    depth_vis = _colorize_scalar_map(depth, depth_valid)
    disp_vis = _colorize_scalar_map(disp, disp_valid)
    overview = np.concatenate([left_rect, right_rect, disp_vis, depth_vis], axis=1)
    cv2.imwrite(str(depth_dir / f"{frame_id}_depth_vis.png"), depth_vis)
    cv2.imwrite(str(depth_dir / f"{frame_id}_disp_vis.png"), disp_vis)
    cv2.imwrite(str(depth_dir / f"{frame_id}_vis.png"), overview)

    np.savez(
        depth_dir / f"{frame_id}_meta.npz",
        K_rect=k_rect.astype(np.float32),
        baseline_m=np.array([plan.baseline_m], dtype=np.float32),
        left_view=np.array(left_view),
        right_view=np.array(plan.right_view),
        frame_id=np.array(frame_id),
        rt_type=np.array(plan.rt_type),
        assume_undistorted=np.array([int(plan.assume_undistorted)], dtype=np.int32),
        alpha=np.array([plan.alpha], dtype=np.float32),
        overlap_ratio=np.array([plan.overlap_ratio], dtype=np.float32),
        valid_left_ratio=np.array([plan.valid_left_ratio], dtype=np.float32),
        valid_right_ratio=np.array([plan.valid_right_ratio], dtype=np.float32),
        expected_disp=np.array([plan.expected_disp], dtype=np.float32),
        rectification_mode=np.array(plan.rectification_mode),
        rotation=np.array(plan.rotation),
        horizontal=np.array([int(meta["horizontal"])], dtype=np.int32),
    )


def _process_view(
    model: Any,
    seq_dir: Path,
    left_view: str,
    initial_plan: RectifyPlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    right_view = initial_plan.right_view
    frame_ids = sorted(
        {p.stem for p in (seq_dir / "images" / left_view).glob("*.png")} &
        {p.stem for p in (seq_dir / "images" / right_view).glob("*.png")}
    )
    if args.frame_limit is not None:
        frame_ids = frame_ids[:args.frame_limit]

    processed = 0
    skipped = 0
    fallback_count = 0
    depth_medians: list[float] = []

    for frame_id in frame_ids:
        depth_path = seq_dir / "fs_depth" / "highres" / left_view / f"{frame_id}_depth.npy"
        if depth_path.exists() and not args.overwrite:
            continue

        plan = initial_plan
        left_img = cv2.imread(str(seq_dir / "images" / left_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(seq_dir / "images" / plan.right_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            skipped += 1
            continue
        left_cam = load_camera_npz(seq_dir / "cameras" / left_view / f"{frame_id}.npz")
        right_cam = load_camera_npz(seq_dir / "cameras" / plan.right_view / f"{frame_id}.npz")

        try:
            left_rect, right_rect, k_rect, baseline_m, meta = _rectify_pair(
                left_img,
                right_img,
                left_cam,
                right_cam,
                rt_type=plan.rt_type,
                assume_undistorted=plan.assume_undistorted,
                alpha=plan.alpha,
                match_size=plan.match_size,
                rotate_vertical=plan.rotate_vertical,
                crop_to_overlap=plan.crop_to_overlap,
            )
            fx_scaled = float(k_rect[0, 0]) * float(args.inference_scale)
            expected_disp = fx_scaled * float(baseline_m) / float(args.target_depth_m)
            score = _score_candidate(
                overlap_ratio=float(meta["overlap_ratio"]),
                valid_left_ratio=float(meta["valid_left_ratio"]),
                valid_right_ratio=float(meta["valid_right_ratio"]),
                expected_disp=expected_disp,
                right_view=plan.right_view,
                left_view=left_view,
                min_overlap=args.min_overlap_ratio,
                min_valid_ratio=args.min_valid_ratio,
                disp_range=(args.min_expected_disp, args.max_expected_disp),
                target_disp=args.target_disp,
            )
            if score is None:
                raise ValueError("initial plan no longer passes thresholds")
            plan = RectifyPlan(
                left_view=left_view,
                right_view=plan.right_view,
                frame_id=frame_id,
                rt_type=plan.rt_type,
                assume_undistorted=plan.assume_undistorted,
                alpha=plan.alpha,
                match_size=plan.match_size,
                rotate_vertical=plan.rotate_vertical,
                crop_to_overlap=plan.crop_to_overlap,
                overlap_ratio=float(meta["overlap_ratio"]),
                valid_left_ratio=float(meta["valid_left_ratio"]),
                valid_right_ratio=float(meta["valid_right_ratio"]),
                baseline_m=float(baseline_m),
                rectification_mode=str(meta["rectification_mode"]),
                rotation=str(meta["rotation"]),
                expected_disp=float(expected_disp),
                score=float(score),
            )
        except Exception:
            fallback = _search_best_plan_for_frame(seq_dir, left_view, frame_id, args)
            if fallback is None:
                skipped += 1
                continue
            fallback_count += 1
            plan = fallback
            right_img = cv2.imread(str(seq_dir / "images" / plan.right_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            right_cam = load_camera_npz(seq_dir / "cameras" / plan.right_view / f"{frame_id}.npz")
            if right_img is None:
                skipped += 1
                continue
            try:
                left_rect, right_rect, k_rect, baseline_m, meta = _rectify_pair(
                    left_img,
                    right_img,
                    left_cam,
                    right_cam,
                    rt_type=plan.rt_type,
                    assume_undistorted=plan.assume_undistorted,
                    alpha=plan.alpha,
                    match_size=plan.match_size,
                    rotate_vertical=plan.rotate_vertical,
                    crop_to_overlap=plan.crop_to_overlap,
                )
            except Exception:
                skipped += 1
                continue

        if args.inference_scale < 1.0:
            left_rect = cv2.resize(left_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
            right_rect = cv2.resize(right_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
            k_rect = k_rect.copy()
            k_rect[0, :] *= float(args.inference_scale)
            k_rect[1, :] *= float(args.inference_scale)

        disp = predict_disparity(
            left_rect,
            right_rect,
            model,
            device=args.device,
            valid_iters=args.valid_iters,
            input_color=args.input_color,
        )
        depth = disparity_to_depth(disp, k_rect, baseline_m)
        _save_outputs(seq_dir, left_view, frame_id, left_rect, right_rect, disp, depth, k_rect, plan, meta)

        valid_depth = np.isfinite(depth) & (depth > 0)
        if valid_depth.any():
            depth_medians.append(float(np.median(depth[valid_depth])))
        processed += 1
        if processed == 1:
            print(
                f"[OK] {seq_dir.name} {left_view}->{plan.right_view} frame={frame_id} "
                f"overlap={plan.overlap_ratio:.4f} valid=({plan.valid_left_ratio:.4f},{plan.valid_right_ratio:.4f}) "
                f"disp_target={plan.expected_disp:.2f} rect={plan.rectification_mode} rt={plan.rt_type}"
            )

    return {
        "left_view": left_view,
        "right_view": initial_plan.right_view,
        "frames_total": len(frame_ids),
        "processed": processed,
        "skipped": skipped,
        "fallback_count": fallback_count,
        "depth_median_mean": float(np.mean(depth_medians)) if depth_medians else None,
        "depth_median_min": float(np.min(depth_medians)) if depth_medians else None,
        "depth_median_max": float(np.max(depth_medians)) if depth_medians else None,
        "plan": asdict(initial_plan),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNA-Rendering highres batch depth with robust ROI search")
    parser.add_argument("--fs-root", type=Path, default=CODE_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequence", type=str, default="0007_04")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--input-color", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument("--match-size", choices=["left", "right"], default="left")
    parser.add_argument("--rotate-vertical", action="store_true", default=True)
    parser.add_argument("--no-rotate-vertical", dest="rotate_vertical", action="store_false")
    parser.add_argument("--crop-to-overlap", action="store_true", default=True)
    parser.add_argument("--no-crop-to-overlap", dest="crop_to_overlap", action="store_false")
    parser.add_argument("--inference-scale", type=float, default=0.25)
    parser.add_argument("--target-depth-m", type=float, default=3.0)
    parser.add_argument("--target-disp", type=float, default=192.0)
    parser.add_argument("--min-expected-disp", type=float, default=32.0)
    parser.add_argument("--max-expected-disp", type=float, default=512.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.30)
    parser.add_argument("--min-valid-ratio", type=float, default=0.60)
    parser.add_argument("--min-probe-disp", type=float, default=16.0)
    parser.add_argument("--max-probe-disp", type=float, default=512.0)
    parser.add_argument("--max-probe-depth-m", type=float, default=25.0)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not (0.0 < args.inference_scale <= 1.0):
        raise ValueError(f"inference_scale must be in (0, 1], got {args.inference_scale}")

    args.rt_type_candidates = ["c2w", "w2c"]
    args.assume_undistorted_candidates = [True, False]
    args.alpha_candidates = [0.0, 0.25, 1.0]

    seq_dir = args.data_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    model = load_model(args.fs_root, device=args.device)

    chosen_plans: dict[str, RectifyPlan] = {}
    for left_view in HIGHRES_VIEWS:
        plan_candidates: list[tuple[float, RectifyPlan, dict[str, float]]] = []
        seen_frames: set[str] = set()
        for right_view in _candidate_right_views(left_view):
            sample_frame = _first_shared_frame(seq_dir, left_view, right_view)
            if sample_frame is None or sample_frame in seen_frames:
                continue
            seen_frames.add(sample_frame)
            candidate = _search_best_plan_for_frame(seq_dir, left_view, sample_frame, args)
            if candidate is None:
                continue
            probe = _probe_plan_quality(model, seq_dir, candidate, args)
            if probe is None:
                continue
            disp_median = float(probe["disp_median"])
            depth_median = float(probe["depth_median"])
            depth_q95 = float(probe["depth_q95"])
            if disp_median < args.min_probe_disp or disp_median > args.max_probe_disp:
                continue
            if depth_median > args.max_probe_depth_m or depth_q95 > args.max_probe_depth_m:
                continue
            rank = abs(math.log(disp_median / max(args.target_disp, 1e-6))) + candidate.score
            plan_candidates.append((rank, candidate, probe))

        if not plan_candidates:
            print(f"[WARN] no valid plan for left view {left_view}")
            continue
        plan_candidates.sort(key=lambda x: x[0])
        _, probe_plan, probe = plan_candidates[0]
        chosen_plans[left_view] = probe_plan
        print(
            f"[PLAN] left={left_view} right={probe_plan.right_view} rt={probe_plan.rt_type} "
            f"alpha={probe_plan.alpha} undist={probe_plan.assume_undistorted} "
            f"overlap={probe_plan.overlap_ratio:.4f} valid=({probe_plan.valid_left_ratio:.4f},{probe_plan.valid_right_ratio:.4f}) "
            f"disp={probe_plan.expected_disp:.2f} probe_disp={probe['disp_median']:.2f} "
            f"probe_depth={probe['depth_median']:.2f} rect={probe_plan.rectification_mode}"
        )

    summaries: list[dict[str, Any]] = []
    for left_view in HIGHRES_VIEWS:
        plan = chosen_plans.get(left_view)
        if plan is None:
            continue
        summaries.append(_process_view(model, seq_dir, left_view, plan, args))

    summary = {
        "sequence": args.sequence,
        "inference_scale": args.inference_scale,
        "target_depth_m": args.target_depth_m,
        "target_disp": args.target_disp,
        "min_overlap_ratio": args.min_overlap_ratio,
        "min_valid_ratio": args.min_valid_ratio,
        "views": summaries,
    }

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = seq_dir / "fs_depth" / "highres_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
