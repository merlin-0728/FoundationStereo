#!/usr/bin/env python3
"""Diagnose stereo rectification geometry for DNA-Rendering highres cameras.

The goal is to identify why rectification fails by comparing, for each
candidate pair and pose convention:
  - baseline direction
  - stereoRectify valid overlap
  - foreground overlap after remap
  - foreground row overlap

This script does not run FoundationStereo. It only diagnoses geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from foundationstereo_model import load_camera_npz


HIGHRES_VIEWS = [f"{i:02d}" for i in range(48, 60, 2)]


@dataclass
class FrameDiagnosis:
    sequence: str
    frame_id: str
    left_view: str
    right_view: str
    rt_type: str
    assume_undistorted: bool
    alpha: float
    match_size: str
    baseline_norm_m: float
    baseline_dir_left: list[float]
    dominant_axis: str
    dominant_axis_ratio: float
    rectification_mode: str
    roi_intersection_ratio: float
    valid_left_ratio: float
    valid_right_ratio: float
    valid_overlap_ratio: float
    fg_left_ratio: float
    fg_right_ratio: float
    fg_overlap_ratio: float
    fg_iou: float
    fg_row_overlap_ratio: float
    fg_row_overlap_rows: int
    left_foreground_rows: int
    right_foreground_rows: int
    image_size: list[int]
    notes: list[str]


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


def _resize_match(
    left_img: np.ndarray,
    right_img: np.ndarray,
    kl: np.ndarray,
    kr: np.ndarray,
    match_size: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = left_img
    right = right_img
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    if (h_l, w_l) == (h_r, w_r):
        return left, right, kl, kr
    if match_size == "left":
        sx = w_l / float(w_r)
        sy = h_l / float(h_r)
        kr = kr.copy()
        kr[0, :] *= sx
        kr[1, :] *= sy
        right = cv2.resize(right, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
        return left, right, kl, kr
    if match_size == "right":
        sx = w_r / float(w_l)
        sy = h_r / float(h_l)
        kl = kl.copy()
        kl[0, :] *= sx
        kl[1, :] *= sy
        left = cv2.resize(left, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
        return left, right, kl, kr
    raise ValueError(f"Unknown match_size: {match_size}")


def _roi_intersection_ratio(roi1: tuple[int, int, int, int], roi2: tuple[int, int, int, int], image_size: tuple[int, int]) -> float:
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    xi = max(x1, x2)
    yi = max(y1, y2)
    xa = min(x1 + w1, x2 + w2)
    ya = min(y1 + h1, y2 + h2)
    iw = max(0, xa - xi)
    ih = max(0, ya - yi)
    return float((iw * ih) / max(image_size[0] * image_size[1], 1))


def _foreground_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image.max(axis=2) > threshold)


def _dominant_axis(vec: np.ndarray) -> tuple[str, float]:
    abs_vec = np.abs(vec)
    idx = int(abs_vec.argmax())
    axis = "xyz"[idx]
    denom = float(np.linalg.norm(vec)) + 1e-9
    return axis, float(abs_vec[idx] / denom)


def diagnose_pair_frame(
    *,
    seq_dir: Path,
    frame_id: str,
    left_view: str,
    right_view: str,
    rt_type: str,
    assume_undistorted: bool,
    alpha: float,
    match_size: str,
    foreground_threshold: int,
) -> FrameDiagnosis:
    left_img = cv2.imread(str(seq_dir / "images" / left_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(seq_dir / "images" / right_view / f"{frame_id}.png"), cv2.IMREAD_COLOR)
    if left_img is None or right_img is None:
        raise FileNotFoundError(f"Missing images for {left_view}/{right_view} frame {frame_id}")

    left_cam = load_camera_npz(seq_dir / "cameras" / left_view / f"{frame_id}.npz")
    right_cam = load_camera_npz(seq_dir / "cameras" / right_view / f"{frame_id}.npz")

    kl = left_cam["K"].astype(np.float64).copy()
    kr = right_cam["K"].astype(np.float64).copy()
    dl = left_cam["D"].astype(np.float64).reshape(-1).copy()
    dr = right_cam["D"].astype(np.float64).reshape(-1).copy()
    if assume_undistorted:
        dl = np.zeros_like(dl)
        dr = np.zeros_like(dr)

    left_img, right_img, kl, kr = _resize_match(left_img, right_img, kl, kr, match_size)

    w2c_l = _w2c_from_rt(left_cam["RT"], rt_type)
    w2c_r = _w2c_from_rt(right_cam["RT"], rt_type)
    rl, tl = w2c_l[:3, :3], w2c_l[:3, 3]
    rr, tr = w2c_r[:3, :3], w2c_r[:3, 3]
    r = rr @ rl.T
    t = tr - r @ tl

    image_size = (left_img.shape[1], left_img.shape[0])
    r1, r2, p1, p2, _q, roi1, roi2 = cv2.stereoRectify(
        kl, dl, kr, dr, image_size, r, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map1x, map1y = cv2.initUndistortRectifyMap(kl, dl, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(kr, dr, r2, p2, image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    valid_left = (map1x >= 0.0) & (map1x <= image_size[0] - 1) & (map1y >= 0.0) & (map1y <= image_size[1] - 1)
    valid_right = (map2x >= 0.0) & (map2x <= image_size[0] - 1) & (map2y >= 0.0) & (map2y <= image_size[1] - 1)
    valid_overlap = valid_left & valid_right

    fg_left = _foreground_mask(left_rect, foreground_threshold)
    fg_right = _foreground_mask(right_rect, foreground_threshold)
    fg_overlap = fg_left & fg_right
    fg_union = fg_left | fg_right

    fg_rows_left = fg_left.any(axis=1)
    fg_rows_right = fg_right.any(axis=1)
    fg_rows_overlap = fg_rows_left & fg_rows_right
    fg_rows_union = fg_rows_left | fg_rows_right

    baseline_norm = float(np.linalg.norm(t))
    baseline_dir = (t / baseline_norm).tolist() if baseline_norm > 1e-9 else [0.0, 0.0, 0.0]
    dominant_axis, dominant_ratio = _dominant_axis(t)
    horizontal = abs(float(p2[0, 3])) >= abs(float(p2[1, 3]))
    rectification_mode = "horizontal" if horizontal else "vertical"

    notes: list[str] = []
    roi_inter_ratio = _roi_intersection_ratio(roi1, roi2, image_size)
    valid_overlap_ratio = float(valid_overlap.mean())
    fg_iou = float(fg_overlap.sum() / max(fg_union.sum(), 1))
    fg_row_overlap_ratio = float(fg_rows_overlap.sum() / max(fg_rows_union.sum(), 1))

    if roi_inter_ratio == 0.0:
        notes.append("roi_intersection_zero")
    if valid_overlap_ratio < 0.1:
        notes.append("valid_overlap_low")
    if fg_iou < 0.05:
        notes.append("foreground_overlap_low")
    if fg_row_overlap_ratio == 0.0:
        notes.append("foreground_rows_disjoint")
    if dominant_ratio < 0.6:
        notes.append("baseline_not_axis_aligned")

    return FrameDiagnosis(
        sequence=seq_dir.name,
        frame_id=frame_id,
        left_view=left_view,
        right_view=right_view,
        rt_type=rt_type,
        assume_undistorted=assume_undistorted,
        alpha=float(alpha),
        match_size=match_size,
        baseline_norm_m=baseline_norm,
        baseline_dir_left=[float(v) for v in baseline_dir],
        dominant_axis=dominant_axis,
        dominant_axis_ratio=dominant_ratio,
        rectification_mode=rectification_mode,
        roi_intersection_ratio=roi_inter_ratio,
        valid_left_ratio=float(valid_left.mean()),
        valid_right_ratio=float(valid_right.mean()),
        valid_overlap_ratio=valid_overlap_ratio,
        fg_left_ratio=float(fg_left.mean()),
        fg_right_ratio=float(fg_right.mean()),
        fg_overlap_ratio=float(fg_overlap.mean()),
        fg_iou=fg_iou,
        fg_row_overlap_ratio=fg_row_overlap_ratio,
        fg_row_overlap_rows=int(fg_rows_overlap.sum()),
        left_foreground_rows=int(fg_rows_left.sum()),
        right_foreground_rows=int(fg_rows_right.sum()),
        image_size=[int(image_size[0]), int(image_size[1])],
        notes=notes,
    )


def _shared_frames(seq_dir: Path, left_view: str, right_view: str) -> list[str]:
    left_frames = {p.stem for p in (seq_dir / "images" / left_view).glob("*.png")}
    right_frames = {p.stem for p in (seq_dir / "images" / right_view).glob("*.png")}
    return sorted(left_frames & right_frames)


def _aggregate(items: list[FrameDiagnosis]) -> dict[str, Any]:
    numeric_keys = [
        "baseline_norm_m",
        "dominant_axis_ratio",
        "roi_intersection_ratio",
        "valid_left_ratio",
        "valid_right_ratio",
        "valid_overlap_ratio",
        "fg_left_ratio",
        "fg_right_ratio",
        "fg_overlap_ratio",
        "fg_iou",
        "fg_row_overlap_ratio",
        "fg_row_overlap_rows",
        "left_foreground_rows",
        "right_foreground_rows",
    ]
    out: dict[str, Any] = {}
    first = items[0]
    out.update(
        sequence=first.sequence,
        left_view=first.left_view,
        right_view=first.right_view,
        rt_type=first.rt_type,
        assume_undistorted=first.assume_undistorted,
        alpha=first.alpha,
        match_size=first.match_size,
        rectification_mode=first.rectification_mode,
        dominant_axis=first.dominant_axis,
        baseline_dir_left=first.baseline_dir_left,
        frame_ids=[item.frame_id for item in items],
        num_frames=len(items),
    )
    for key in numeric_keys:
        vals = [float(getattr(item, key)) for item in items]
        out[key] = float(sum(vals) / len(vals))
    note_set = sorted({note for item in items for note in item.notes})
    out["notes"] = note_set
    return out


def _config_score(record: dict[str, Any]) -> float:
    return (
        record["fg_row_overlap_ratio"] * 4.0
        + record["fg_iou"] * 3.0
        + record["valid_overlap_ratio"] * 1.5
        + record["roi_intersection_ratio"] * 0.5
        - max(0.0, 0.15 - record["fg_row_overlap_ratio"]) * 5.0
    )


def _infer_root_cause(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["no_records"]

    c2w = [r for r in records if r["rt_type"] == "c2w"]
    w2c = [r for r in records if r["rt_type"] == "w2c"]
    mean_fg_c2w = statistics.fmean(r["fg_row_overlap_ratio"] for r in c2w) if c2w else 0.0
    mean_fg_w2c = statistics.fmean(r["fg_row_overlap_ratio"] for r in w2c) if w2c else 0.0
    mean_valid_c2w = statistics.fmean(r["valid_overlap_ratio"] for r in c2w) if c2w else 0.0
    mean_valid_w2c = statistics.fmean(r["valid_overlap_ratio"] for r in w2c) if w2c else 0.0

    causes: list[str] = []
    if mean_fg_c2w > 3 * max(mean_fg_w2c, 1e-6) and mean_fg_c2w > 0.05:
        causes.append("rt_type_likely_c2w")
    elif mean_fg_w2c > 3 * max(mean_fg_c2w, 1e-6) and mean_fg_w2c > 0.05:
        causes.append("rt_type_likely_w2c")

    if max(mean_valid_c2w, mean_valid_w2c) < 0.1:
        causes.append("rectification_valid_overlap_globally_low")
    elif max(mean_fg_c2w, mean_fg_w2c) < 0.05 and max(mean_valid_c2w, mean_valid_w2c) > 0.3:
        causes.append("foreground_shift_after_rectification")

    best = max(records, key=_config_score)
    if best["fg_row_overlap_ratio"] < 0.05:
        causes.append("pair_selection_or_pose_consistency_problem")
    elif best["dominant_axis_ratio"] < 0.6:
        causes.append("baseline_direction_not_stable")

    return causes or ["no_clear_single_cause"]


def _format_summary(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("Highres Rectification Geometry Diagnosis")
    lines.append("")

    by_left: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_left.setdefault(rec["left_view"], []).append(rec)

    for left_view in sorted(by_left):
        subset = sorted(by_left[left_view], key=_config_score, reverse=True)
        lines.append(f"[LEFT {left_view}]")
        for rec in subset[:4]:
            lines.append(
                "  "
                f"R={rec['right_view']} rt={rec['rt_type']} undist={rec['assume_undistorted']} alpha={rec['alpha']:.2f} "
                f"mode={rec['rectification_mode']} base_axis={rec['dominant_axis']}({rec['dominant_axis_ratio']:.2f}) "
                f"valid={rec['valid_overlap_ratio']:.3f} fg_iou={rec['fg_iou']:.3f} fg_rows={rec['fg_row_overlap_ratio']:.3f} "
                f"notes={','.join(rec['notes']) if rec['notes'] else '-'}"
            )
        lines.append("")

    causes = _infer_root_cause(records)
    lines.append("Likely Causes:")
    for cause in causes:
        lines.append(f"  - {cause}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose highres rectification geometry")
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequence", type=str, default="0007_04")
    parser.add_argument("--views", nargs="*", default=HIGHRES_VIEWS)
    parser.add_argument("--frame-id", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 1.0])
    parser.add_argument("--rt-types", nargs="*", default=["c2w", "w2c"])
    parser.add_argument("--assume-undistorted", nargs="*", default=["true", "false"])
    parser.add_argument("--match-size", choices=["left", "right"], default="left")
    parser.add_argument("--foreground-threshold", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    return parser


def _parse_bool_tokens(tokens: list[str]) -> list[bool]:
    out: list[bool] = []
    for token in tokens:
        val = token.strip().lower()
        if val in {"1", "true", "yes", "y"}:
            out.append(True)
        elif val in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Unknown boolean token: {token}")
    return out


def main() -> None:
    args = build_parser().parse_args()
    seq_dir = args.data_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    undistorted_candidates = _parse_bool_tokens(args.assume_undistorted)

    aggregated: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for left_view in args.views:
        for right_view in args.views:
            if left_view == right_view:
                continue
            shared = _shared_frames(seq_dir, left_view, right_view)
            if args.frame_id is not None:
                shared = [f for f in shared if f == args.frame_id]
            else:
                shared = shared[: args.max_frames]
            if not shared:
                continue

            for rt_type in args.rt_types:
                for assume_undistorted in undistorted_candidates:
                    for alpha in args.alphas:
                        items: list[FrameDiagnosis] = []
                        for frame_id in shared:
                            diag = diagnose_pair_frame(
                                seq_dir=seq_dir,
                                frame_id=frame_id,
                                left_view=left_view,
                                right_view=right_view,
                                rt_type=rt_type,
                                assume_undistorted=assume_undistorted,
                                alpha=float(alpha),
                                match_size=args.match_size,
                                foreground_threshold=args.foreground_threshold,
                            )
                            items.append(diag)
                            raw.append(asdict(diag))
                        aggregated.append(_aggregate(items))

    aggregated.sort(key=_config_score, reverse=True)
    summary_txt = _format_summary(aggregated)
    print(summary_txt)

    out_json = args.output_json
    if out_json is None:
        out_json = seq_dir / "fs_depth_debug" / "highres_geometry_diagnosis.json"
    out_txt = args.output_txt
    if out_txt is None:
        out_txt = seq_dir / "fs_depth_debug" / "highres_geometry_diagnosis.txt"

    payload = {
        "sequence": args.sequence,
        "views": args.views,
        "frame_id": args.frame_id,
        "max_frames": args.max_frames,
        "aggregated": aggregated,
        "raw": raw,
        "likely_causes": _infer_root_cause(aggregated),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_txt.write_text(summary_txt, encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_txt}")


if __name__ == "__main__":
    main()
