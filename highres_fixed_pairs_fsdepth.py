#!/usr/bin/env python3
"""Generate FoundationStereo depth for fixed highres view pairs.

Fixed pairs requested by user:
  - 54 -> 48
  - 50 -> 52
  - 56 -> 58

Frames are processed one-to-one by frame id. For each left view, outputs are
saved under:
  fs_rectified/highres/<left_view>/
  fs_depth/highres/<left_view>/
"""

from __future__ import annotations

import argparse
import json
import sys
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
    rectify_stereo_pair,
)


FIXED_PAIRS = [("54", "48"), ("50", "52"), ("56", "58")]


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
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid_mask] = 0
    return vis


def _save_outputs(
    seq_dir: Path,
    left_view: str,
    frame_id: str,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    disp: np.ndarray,
    depth: np.ndarray,
    k_rect: np.ndarray,
    baseline_m: float,
    meta: dict[str, Any],
    right_view: str,
) -> None:
    rect_dir = seq_dir / "fs_rectified" / "highres" / left_view
    depth_dir = seq_dir / "fs_depth" / "highres" / left_view
    rect_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(rect_dir / f"{frame_id}_left.png"), left_rect)
    cv2.imwrite(str(rect_dir / f"{frame_id}_right.png"), right_rect)

    np.save(depth_dir / f"{frame_id}_depth.npy", depth.astype(np.float32))
    np.save(depth_dir / f"{frame_id}_disp.npy", disp.astype(np.float32))
    np.savez(
        depth_dir / f"{frame_id}_meta.npz",
        K_rect=k_rect.astype(np.float32),
        baseline_m=np.array([baseline_m], dtype=np.float32),
        left_view=np.array(left_view),
        right_view=np.array(right_view),
        frame_id=np.array(frame_id),
        horizontal=np.array([int(meta["horizontal"])], dtype=np.int32),
        rectification_mode=np.array(meta["rectification_mode"]),
        rotation=np.array(meta["rotation"]),
        intersection_ratio=np.array([float(meta["intersection_ratio"])], dtype=np.float32),
    )

    depth_valid = np.isfinite(depth) & (depth > 0)
    disp_valid = np.isfinite(disp) & (disp > 1e-4)
    depth_vis = _colorize_scalar_map(depth, depth_valid)
    disp_vis = _colorize_scalar_map(disp, disp_valid)
    overview = np.concatenate([left_rect, right_rect, disp_vis, depth_vis], axis=1)
    cv2.imwrite(str(depth_dir / f"{frame_id}_depth_vis.png"), depth_vis)
    cv2.imwrite(str(depth_dir / f"{frame_id}_disp_vis.png"), disp_vis)
    cv2.imwrite(str(depth_dir / f"{frame_id}_vis.png"), overview)
    cv2.imwrite(str(depth_dir / f"{frame_id}_depth_mm.png"), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))


def _shared_frames(seq_dir: Path, left_view: str, right_view: str) -> list[str]:
    left_frames = {p.stem for p in (seq_dir / "images" / left_view).glob("*.png")}
    right_frames = {p.stem for p in (seq_dir / "images" / right_view).glob("*.png")}
    return sorted(left_frames & right_frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoundationStereo depth with fixed highres pairs")
    parser.add_argument("--fs-root", type=Path, default=CODE_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rt-type", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--min-intersection-ratio", type=float, default=0.0)
    parser.add_argument("--inference-scale", type=float, default=0.5)
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
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sequences = args.sequences or sorted([p.name for p in args.data_root.iterdir() if p.is_dir()])
    model = load_model(args.fs_root, device=args.device)
    summary: dict[str, Any] = {"sequences": {}}

    for sequence in sequences:
        seq_dir = args.data_root / sequence
        if not seq_dir.exists():
            print(f"[WARN] sequence missing: {sequence}")
            continue
        print(f"[SEQ] {sequence}")
        seq_summary: dict[str, Any] = {}
        for left_view, right_view in FIXED_PAIRS:
            frame_ids = _shared_frames(seq_dir, left_view, right_view)
            if args.frame_limit is not None:
                frame_ids = frame_ids[: args.frame_limit]
            if not frame_ids:
                print(f"[WARN] no shared frames for {sequence} {left_view}->{right_view}")
                continue
            processed = 0
            failed = 0
            first_ok = None
            for frame_id in frame_ids:
                depth_path = seq_dir / "fs_depth" / "highres" / left_view / f"{frame_id}_depth.npy"
                if depth_path.exists() and not args.overwrite:
                    continue

                left_img = cv2.imread(str(seq_dir / "images" / left_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
                right_img = cv2.imread(str(seq_dir / "images" / right_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
                if left_img is None or right_img is None:
                    failed += 1
                    continue
                left_cam = load_camera_npz(seq_dir / "cameras" / left_view / f"{frame_id}.npz")
                right_cam = load_camera_npz(seq_dir / "cameras" / right_view / f"{frame_id}.npz")

                try:
                    left_rect, right_rect, k_rect, baseline_m, meta = rectify_stereo_pair(
                        left_img,
                        right_img,
                        left_cam,
                        right_cam,
                        rt_type=args.rt_type,
                        assume_undistorted=args.assume_undistorted,
                        alpha=args.alpha,
                        crop_to_intersection=args.crop_to_intersection,
                        min_intersection_ratio=args.min_intersection_ratio,
                        require_horizontal=args.require_horizontal,
                        rotate_vertical=args.rotate_vertical,
                        match_size=args.match_size,
                    )
                except Exception as exc:
                    print(f"[WARN] rectify failed {sequence} {left_view}->{right_view} frame {frame_id}: {exc}")
                    failed += 1
                    continue

                if args.inference_scale < 1.0:
                    left_rect = cv2.resize(left_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
                    right_rect = cv2.resize(right_rect, dsize=None, fx=args.inference_scale, fy=args.inference_scale, interpolation=cv2.INTER_AREA)
                    k_rect = k_rect.copy()
                    k_rect[0, :] *= args.inference_scale
                    k_rect[1, :] *= args.inference_scale

                disp = predict_disparity(
                    left_rect,
                    right_rect,
                    model,
                    device=args.device,
                    valid_iters=args.valid_iters,
                    input_color=args.input_color,
                )
                depth = disparity_to_depth(disp, k_rect, baseline_m)
                _save_outputs(seq_dir, left_view, frame_id, left_rect, right_rect, disp, depth, k_rect, baseline_m, meta, right_view)
                processed += 1
                if first_ok is None:
                    first_ok = {
                        "frame_id": frame_id,
                        "depth_min": float(np.nanmin(depth)),
                        "depth_max": float(np.nanmax(depth)),
                        "intersection_ratio": float(meta["intersection_ratio"]),
                        "rectification_mode": meta["rectification_mode"],
                    }
                    print(
                        f"[OK] {sequence} {left_view}->{right_view} frame={frame_id} "
                        f"depth[{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}] inter={meta['intersection_ratio']:.4f}"
                    )

            seq_summary[f"{left_view}->{right_view}"] = {
                "frames_total": len(frame_ids),
                "processed": processed,
                "failed": failed,
                "first_ok": first_ok,
            }
        summary["sequences"][sequence] = seq_summary

    summary_path = args.summary_path or (args.data_root / "fixed_pairs_highres_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
