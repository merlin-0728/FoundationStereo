#!/usr/bin/env python3
"""DNA-Rendering depth generation with FoundationStereo.

This script can run a single test pair or batch-process DNA-Rendering
sequences by view groups:
  - lowres: 00-46
  - highres: 48-58
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from foundationstereo_model import (
    disparity_to_depth,
    load_camera_npz,
    load_model,
    predict_disparity,
    rectify_stereo_pair,
)


LOWRES_VIEWS = [f"{i:02d}" for i in range(0, 48, 2)]
HIGHRES_VIEWS = [f"{i:02d}" for i in range(48, 60, 2)]


def _discover_sequences(data_root: Path) -> list[str]:
    seqs = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and (path / "cameras").exists() and (path / "images").exists():
            seqs.append(path.name)
    return seqs


def _candidate_right_views(left_view: str, group_views: list[str]) -> list[str]:
    idx = group_views.index(left_view)
    candidates = []
    for step in range(1, len(group_views)):
        hi = idx + step
        lo = idx - step
        if hi < len(group_views):
            candidates.append(group_views[hi])
        if lo >= 0:
            candidates.append(group_views[lo])
    return candidates


def _find_right_view(seq_dir: Path, group_views: list[str], left_view: str,
                     rt_type: str, assume_undistorted: bool, alpha: float,
                     crop_to_intersection: bool, require_horizontal: bool,
                     rotate_vertical: bool, min_intersection_ratio: float,
                     inference_scale: float, match_size: str) -> str | None:
    left_img_dir = seq_dir / "images" / left_view
    left_cam_dir = seq_dir / "cameras" / left_view
    left_frames = sorted(p.stem for p in left_img_dir.glob("*.png"))
    if not left_frames:
        return None
    sample_frame = left_frames[0]
    left_img = cv2.imread(str(left_img_dir / f"{sample_frame}.png"), cv2.IMREAD_COLOR)
    left_cam = load_camera_npz(left_cam_dir / f"{sample_frame}.npz")

    best = None
    best_score = None
    target_disp = 128.0

    for right_view in _candidate_right_views(left_view, group_views):
        right_img_dir = seq_dir / "images" / right_view
        right_cam_dir = seq_dir / "cameras" / right_view
        if not right_img_dir.exists() or not right_cam_dir.exists():
            continue
        right_img_path = right_img_dir / f"{sample_frame}.png"
        right_cam_path = right_cam_dir / f"{sample_frame}.npz"
        if not right_img_path.exists() or not right_cam_path.exists():
            continue
        right_img = cv2.imread(str(right_img_path), cv2.IMREAD_COLOR)
        right_cam = load_camera_npz(right_cam_path)
        try:
            _, _, K_rect, baseline_m, meta = rectify_stereo_pair(
                left_img, right_img, left_cam, right_cam,
                rt_type=rt_type,
                assume_undistorted=assume_undistorted,
                alpha=alpha,
                crop_to_intersection=crop_to_intersection,
                min_intersection_ratio=min_intersection_ratio,
                require_horizontal=require_horizontal,
                rotate_vertical=rotate_vertical,
                match_size=match_size,
            )
            K_score = K_rect.copy()
            if inference_scale < 1:
                K_score[0, :] *= inference_scale
                K_score[1, :] *= inference_scale
            expected_disp = float(K_score[0, 0] * baseline_m / 3.0)
            score = abs(expected_disp - target_disp)
            if best_score is None or score < best_score:
                best = right_view
                best_score = score
        except Exception:
            continue
    return best


def _colorize_scalar_map(values: np.ndarray, valid_mask: np.ndarray, min_val: float | None = None,
                         max_val: float | None = None) -> np.ndarray:
    vis = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not valid_mask.any():
        return vis
    valid = values[valid_mask]
    if min_val is None:
        min_val = float(np.percentile(valid, 5))
    if max_val is None:
        max_val = float(np.percentile(valid, 95))
    if max_val <= min_val:
        max_val = min_val + 1e-6
    norm = np.clip((values - min_val) / (max_val - min_val), 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    vis[~valid_mask] = 0
    return vis


def _save_frame_outputs(out_root: Path,
                        group_name: str,
                        left_view: str,
                        frame_id: str,
                        left_rect: np.ndarray,
                        right_rect: np.ndarray,
                        disp: np.ndarray,
                        depth: np.ndarray,
                        K_rect: np.ndarray,
                        baseline_m: float,
                        meta: dict,
                        save_png_depth: bool = True) -> None:
    rect_dir = out_root / "fs_rectified" / group_name / left_view
    depth_dir = out_root / "fs_depth" / group_name / left_view
    rect_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(rect_dir / f"{frame_id}_left.png"), left_rect)
    cv2.imwrite(str(rect_dir / f"{frame_id}_right.png"), right_rect)

    np.save(depth_dir / f"{frame_id}_depth.npy", depth.astype(np.float32))
    np.save(depth_dir / f"{frame_id}_disp.npy", disp.astype(np.float32))
    np.savez(depth_dir / f"{frame_id}_meta.npz",
             K_rect=K_rect.astype(np.float32),
             baseline_m=np.array([baseline_m], dtype=np.float32),
             left_view=np.array(left_view),
             right_view=np.array(meta["right_view"]),
             frame_id=np.array(frame_id),
             horizontal=np.array([int(meta["horizontal"])], dtype=np.int32),
             rectification_mode=np.array(meta["rectification_mode"]),
             rotation=np.array(meta["rotation"]),
             intersection_ratio=np.array([float(meta["intersection_ratio"])], dtype=np.float32))

    if save_png_depth:
        valid = np.isfinite(depth) & (depth > 0)
        depth_vis = _colorize_scalar_map(depth, valid)
        disp_valid = np.isfinite(disp) & (disp > 1e-4)
        disp_vis = _colorize_scalar_map(disp, disp_valid)
        overview = np.concatenate([left_rect, right_rect, disp_vis, depth_vis], axis=1)
        cv2.imwrite(str(depth_dir / f"{frame_id}_vis.png"), overview)
        cv2.imwrite(str(depth_dir / f"{frame_id}_depth_vis.png"), depth_vis)
        cv2.imwrite(str(depth_dir / f"{frame_id}_disp_vis.png"), disp_vis)

        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{frame_id}_depth_mm.png"), depth_mm)


def _process_pair(model, fs_root: Path, seq_dir: Path, group_name: str, left_view: str,
                  right_view: str, device: str, rt_type: str, valid_iters: int,
                  assume_undistorted: bool, alpha: float, crop_to_intersection: bool,
                  require_horizontal: bool, rotate_vertical: bool, min_intersection_ratio: float,
                  inference_scale: float, match_size: str, input_color: str, skip_existing: bool,
                  frame_id_filter: str | None = None) -> None:
    left_img_dir = seq_dir / "images" / left_view
    right_img_dir = seq_dir / "images" / right_view
    left_cam_dir = seq_dir / "cameras" / left_view
    right_cam_dir = seq_dir / "cameras" / right_view

    frame_ids = sorted({p.stem for p in left_img_dir.glob("*.png")} & {p.stem for p in right_img_dir.glob("*.png")})
    if frame_id_filter is not None:
        frame_ids = [f for f in frame_ids if f == frame_id_filter]
    if not frame_ids:
        print(f"[WARN] No shared frames for {seq_dir.name} {left_view}/{right_view}")
        return

    print(f"[PAIR] {seq_dir.name} {group_name} left={left_view} right={right_view} frames={len(frame_ids)}")

    for frame_id in frame_ids:
        depth_dir = seq_dir / "fs_depth" / group_name / left_view
        if skip_existing and (depth_dir / f"{frame_id}_depth.npy").exists():
            continue

        left_img = cv2.imread(str(left_img_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(right_img_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            print(f"[WARN] Missing image for {seq_dir.name} frame {frame_id} pair {left_view}/{right_view}")
            continue

        left_cam = load_camera_npz(left_cam_dir / f"{frame_id}.npz")
        right_cam = load_camera_npz(right_cam_dir / f"{frame_id}.npz")

        try:
            left_rect, right_rect, K_rect, baseline_m, meta = rectify_stereo_pair(
                left_img, right_img, left_cam, right_cam,
                rt_type=rt_type,
                assume_undistorted=assume_undistorted,
                alpha=alpha,
                crop_to_intersection=crop_to_intersection,
                min_intersection_ratio=min_intersection_ratio,
                require_horizontal=require_horizontal,
                rotate_vertical=rotate_vertical,
                match_size=match_size,
            )
        except Exception as exc:
            print(f"[WARN] Rectify failed for {seq_dir.name} {left_view}/{right_view} frame {frame_id}: {exc}")
            continue

        if inference_scale <= 0 or inference_scale > 1:
            raise ValueError(f"inference_scale must be in (0, 1], got {inference_scale}")
        if inference_scale < 1:
            left_rect = cv2.resize(left_rect, dsize=None, fx=inference_scale, fy=inference_scale,
                                   interpolation=cv2.INTER_AREA)
            right_rect = cv2.resize(right_rect, dsize=None, fx=inference_scale, fy=inference_scale,
                                    interpolation=cv2.INTER_AREA)
            K_rect = K_rect.copy()
            K_rect[0, :] *= inference_scale
            K_rect[1, :] *= inference_scale

        disp = predict_disparity(
            left_rect, right_rect, model,
            device=device, valid_iters=valid_iters, input_color=input_color
        )
        depth = disparity_to_depth(disp, K_rect, baseline_m)

        meta = dict(meta)
        meta["right_view"] = right_view

        _save_frame_outputs(
            seq_dir, group_name, left_view, frame_id,
            left_rect, right_rect, disp, depth, K_rect, baseline_m, meta,
        )

        if frame_id == frame_ids[0]:
            print(f"[OK] {seq_dir.name} {group_name} {left_view}->{right_view} frame {frame_id}: "
                  f"disp[{np.nanmin(disp):.4f}, {np.nanmax(disp):.4f}] "
                  f"depth[{np.nanmin(depth):.4f}, {np.nanmax(depth):.4f}] "
                  f"baseline={baseline_m:.6f}m")


def _run_single(args) -> None:
    model = load_model(args.fs_root, device=args.device)
    seq_dir = args.data_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    group_views = LOWRES_VIEWS if args.group != "highres" else HIGHRES_VIEWS
    left_view = args.left_view or group_views[0]
    right_view = args.right_view
    if right_view is None:
        right_view = _find_right_view(
            seq_dir, group_views, left_view,
            rt_type=args.rt_type,
            assume_undistorted=args.assume_undistorted,
            alpha=args.alpha,
            crop_to_intersection=args.crop_to_intersection,
            require_horizontal=args.require_horizontal,
            rotate_vertical=args.rotate_vertical,
            min_intersection_ratio=args.min_intersection_ratio,
            inference_scale=args.inference_scale,
            match_size=args.match_size,
        )
        if right_view is None:
            raise RuntimeError(f"No valid right view found for {left_view} in {args.sequence}")

    _process_pair(
        model=model,
        fs_root=args.fs_root,
        seq_dir=seq_dir,
        group_name=args.group,
        left_view=left_view,
        right_view=right_view,
        device=args.device,
        rt_type=args.rt_type,
        valid_iters=args.valid_iters,
        assume_undistorted=args.assume_undistorted,
        alpha=args.alpha,
        crop_to_intersection=args.crop_to_intersection,
        require_horizontal=args.require_horizontal,
        rotate_vertical=args.rotate_vertical,
        min_intersection_ratio=args.min_intersection_ratio,
        inference_scale=args.inference_scale,
        match_size=args.match_size,
        input_color=args.input_color,
        skip_existing=False,
        frame_id_filter=args.frame_id,
    )


def _run_batch(args) -> None:
    model = load_model(args.fs_root, device=args.device)
    sequences = args.sequences or _discover_sequences(args.data_root)
    groups = []
    if args.group in ("both", "lowres"):
        groups.append(("lowres", LOWRES_VIEWS))
    if args.group in ("both", "highres"):
        groups.append(("highres", HIGHRES_VIEWS))

    for sequence in sequences:
        seq_dir = args.data_root / sequence
        if not seq_dir.exists():
            print(f"[WARN] Skip missing sequence {seq_dir}")
            continue
        print(f"[SEQ] {sequence}")
        for group_name, group_views in groups:
            for left_view in group_views:
                left_dir = seq_dir / "images" / left_view
                if not left_dir.exists():
                    continue
                chosen = _find_right_view(
                    seq_dir, group_views, left_view,
                    rt_type=args.rt_type,
                    assume_undistorted=args.assume_undistorted,
                    alpha=args.alpha,
                    crop_to_intersection=args.crop_to_intersection,
                    require_horizontal=args.require_horizontal,
                    rotate_vertical=args.rotate_vertical,
                    min_intersection_ratio=args.min_intersection_ratio,
                    inference_scale=args.inference_scale,
                    match_size=args.match_size,
                )
                if chosen is None:
                    print(f"[WARN] No right view found for {sequence} {group_name} {left_view}")
                    continue
                _process_pair(
                    model=model,
                    fs_root=args.fs_root,
                    seq_dir=seq_dir,
                    group_name=group_name,
                    left_view=left_view,
                    right_view=chosen,
                    device=args.device,
                    rt_type=args.rt_type,
                    valid_iters=args.valid_iters,
                    assume_undistorted=args.assume_undistorted,
                    alpha=args.alpha,
                    crop_to_intersection=args.crop_to_intersection,
                    require_horizontal=args.require_horizontal,
                    rotate_vertical=args.rotate_vertical,
                    min_intersection_ratio=args.min_intersection_ratio,
                    inference_scale=args.inference_scale,
                    match_size=args.match_size,
                    input_color=args.input_color,
                    skip_existing=args.skip_existing,
                    frame_id_filter=args.frame_id,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoundationStereo DNA-Rendering depth generation")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--fs-root", type=Path, default=CODE_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequence", type=str, default="0007_04")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--group", choices=["lowres", "highres", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--left-view", type=str, default=None)
    parser.add_argument("--right-view", type=str, default=None)
    parser.add_argument("--frame-id", type=str, default=None)
    parser.add_argument("--rt-type", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--min-intersection-ratio", type=float, default=0.0)
    parser.add_argument("--inference-scale", type=float, default=1.0)
    parser.add_argument("--assume-undistorted", action="store_true", default=True)
    parser.add_argument("--no-assume-undistorted", dest="assume_undistorted", action="store_false")
    parser.add_argument("--crop-to-intersection", action="store_true", default=False)
    parser.add_argument("--no-crop-to-intersection", dest="crop_to_intersection", action="store_false")
    parser.add_argument("--require-horizontal", action="store_true", default=True)
    parser.add_argument("--no-require-horizontal", dest="require_horizontal", action="store_false")
    parser.add_argument("--rotate-vertical", action="store_true", default=True)
    parser.add_argument("--no-rotate-vertical", dest="rotate_vertical", action="store_false")
    parser.add_argument("--match-size", choices=["left", "right"], default="left")
    parser.add_argument("--input-color", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-summary", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "single":
        _run_single(args)
    else:
        _run_batch(args)

    if args.save_summary:
        summary = {
            "mode": args.mode,
            "data_root": str(args.data_root),
            "fs_root": str(args.fs_root),
            "group": args.group,
            "sequences": args.sequences if args.sequences is not None else "auto",
        }
        summary_path = args.data_root / "fs_depth_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
