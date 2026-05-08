#!/usr/bin/env python3
"""Search reliable highres stereo pairs for DNA-Rendering.

This script ranks 48-58 candidate pairs using both:
  1. raw foreground overlap on the original images
  2. foreground overlap after stereo rectification

It is stricter than a naive original-image overlap filter. A pair is only
useful if rectification keeps the foreground in overlapping rows/regions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from diagnose_rectification_geometry import (
    HIGHRES_VIEWS,
    _parse_bool_tokens,
    _shared_frames,
    diagnose_pair_frame,
)


@dataclass
class ReliablePairRecord:
    left_view: str
    right_view: str
    frame_id: str
    rt_type: str
    assume_undistorted: bool
    alpha: float
    raw_fg_iou: float
    raw_fg_rows: float
    raw_fg_overlap_ratio: float
    rect_valid_overlap_ratio: float
    rect_fg_iou: float
    rect_fg_rows: float
    rect_fg_overlap_ratio: float
    baseline_norm_m: float
    dominant_axis: str
    dominant_axis_ratio: float
    rectification_mode: str
    score: float
    reliable: bool
    notes: list[str]


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask > 0


def _raw_foreground_metrics(seq_dir: Path, left_view: str, right_view: str, frame_id: str) -> tuple[float, float, float]:
    left_mask = _load_mask(seq_dir / "bkgd_masks" / left_view / f"{frame_id}.png")
    right_mask = _load_mask(seq_dir / "bkgd_masks" / right_view / f"{frame_id}.png")
    inter = left_mask & right_mask
    union = left_mask | right_mask
    fg_iou = float(inter.sum() / max(union.sum(), 1))
    fg_overlap_ratio = float(inter.mean())

    rows_left = left_mask.any(axis=1)
    rows_right = right_mask.any(axis=1)
    row_union = rows_left | rows_right
    row_inter = rows_left & rows_right
    fg_rows = float(row_inter.sum() / max(row_union.sum(), 1))
    return fg_iou, fg_rows, fg_overlap_ratio


def _score_pair(
    *,
    raw_fg_iou: float,
    raw_fg_rows: float,
    rect_valid_overlap_ratio: float,
    rect_fg_iou: float,
    rect_fg_rows: float,
    dominant_axis_ratio: float,
) -> float:
    return (
        raw_fg_iou * 2.0
        + raw_fg_rows * 1.5
        + rect_valid_overlap_ratio * 1.0
        + rect_fg_iou * 4.0
        + rect_fg_rows * 4.0
        + dominant_axis_ratio * 0.2
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search reliable highres pairs")
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
    parser.add_argument("--min-raw-fg-iou", type=float, default=0.05)
    parser.add_argument("--min-raw-fg-rows", type=float, default=0.20)
    parser.add_argument("--min-rect-valid-overlap", type=float, default=0.10)
    parser.add_argument("--min-rect-fg-iou", type=float, default=0.01)
    parser.add_argument("--min-rect-fg-rows", type=float, default=0.10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seq_dir = args.data_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    undistorted_candidates = _parse_bool_tokens(args.assume_undistorted)

    records: list[ReliablePairRecord] = []
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

            frame_id = shared[0]
            raw_fg_iou, raw_fg_rows, raw_fg_overlap_ratio = _raw_foreground_metrics(seq_dir, left_view, right_view, frame_id)

            for rt_type in args.rt_types:
                for assume_undistorted in undistorted_candidates:
                    for alpha in args.alphas:
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
                        score = _score_pair(
                            raw_fg_iou=raw_fg_iou,
                            raw_fg_rows=raw_fg_rows,
                            rect_valid_overlap_ratio=diag.valid_overlap_ratio,
                            rect_fg_iou=diag.fg_iou,
                            rect_fg_rows=diag.fg_row_overlap_ratio,
                            dominant_axis_ratio=diag.dominant_axis_ratio,
                        )
                        reliable = (
                            raw_fg_iou >= args.min_raw_fg_iou
                            and raw_fg_rows >= args.min_raw_fg_rows
                            and diag.valid_overlap_ratio >= args.min_rect_valid_overlap
                            and diag.fg_iou >= args.min_rect_fg_iou
                            and diag.fg_row_overlap_ratio >= args.min_rect_fg_rows
                        )
                        notes = list(diag.notes)
                        if not reliable:
                            notes = sorted(set(notes + ["rejected"]))

                        records.append(
                            ReliablePairRecord(
                                left_view=left_view,
                                right_view=right_view,
                                frame_id=frame_id,
                                rt_type=rt_type,
                                assume_undistorted=assume_undistorted,
                                alpha=float(alpha),
                                raw_fg_iou=raw_fg_iou,
                                raw_fg_rows=raw_fg_rows,
                                raw_fg_overlap_ratio=raw_fg_overlap_ratio,
                                rect_valid_overlap_ratio=diag.valid_overlap_ratio,
                                rect_fg_iou=diag.fg_iou,
                                rect_fg_rows=diag.fg_row_overlap_ratio,
                                rect_fg_overlap_ratio=diag.fg_overlap_ratio,
                                baseline_norm_m=diag.baseline_norm_m,
                                dominant_axis=diag.dominant_axis,
                                dominant_axis_ratio=diag.dominant_axis_ratio,
                                rectification_mode=diag.rectification_mode,
                                score=score,
                                reliable=reliable,
                                notes=notes,
                            )
                        )

    records.sort(key=lambda r: r.score, reverse=True)
    reliable_records = [r for r in records if r.reliable]

    lines: list[str] = []
    lines.append("Highres Reliable Pair Search")
    lines.append("")
    lines.append(f"Total configs: {len(records)}")
    lines.append(f"Reliable configs: {len(reliable_records)}")
    lines.append("")
    lines.append("Top 20 configs:")
    for rec in records[:20]:
        lines.append(
            f"  {rec.left_view}->{rec.right_view} rt={rec.rt_type} undist={rec.assume_undistorted} alpha={rec.alpha:.2f} "
            f"raw_fg_iou={rec.raw_fg_iou:.3f} raw_fg_rows={rec.raw_fg_rows:.3f} "
            f"rect_valid={rec.rect_valid_overlap_ratio:.3f} rect_fg_iou={rec.rect_fg_iou:.3f} rect_fg_rows={rec.rect_fg_rows:.3f} "
            f"mode={rec.rectification_mode} score={rec.score:.3f} reliable={rec.reliable}"
        )
    lines.append("")
    if reliable_records:
        lines.append("Reliable configs:")
        for rec in reliable_records:
            lines.append(
                f"  {rec.left_view}->{rec.right_view} rt={rec.rt_type} undist={rec.assume_undistorted} alpha={rec.alpha:.2f} "
                f"raw_fg_iou={rec.raw_fg_iou:.3f} rect_fg_iou={rec.rect_fg_iou:.3f} rect_fg_rows={rec.rect_fg_rows:.3f}"
            )
    else:
        lines.append("Reliable configs: none")
    lines.append("")
    summary_txt = "\n".join(lines)
    print(summary_txt)

    payload: dict[str, Any] = {
        "sequence": args.sequence,
        "records": [asdict(r) for r in records],
        "reliable_records": [asdict(r) for r in reliable_records],
    }

    out_json = args.output_json or (seq_dir / "fs_depth_debug" / "highres_reliable_pairs.json")
    out_txt = args.output_txt or (seq_dir / "fs_depth_debug" / "highres_reliable_pairs.txt")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_txt.write_text(summary_txt, encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_txt}")


if __name__ == "__main__":
    main()
