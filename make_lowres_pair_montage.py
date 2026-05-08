#!/usr/bin/env python3
"""Create labeled montages for lowres stereo candidate pairs.

Default behavior:
  - sequence: 0007_04
  - views: 00-46 (even indices)
  - pair pattern: consecutive neighbors (00-02, 02-04, ..., 44-46)
  - frame: first shared frame, default 000000

Outputs:
  - one full-resolution side-by-side image per pair
  - one contact sheet grid for quick comparison
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


LOWRES_VIEWS = [f"{i:02d}" for i in range(0, 48, 2)]


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def _find_frame(image_dir: Path, frame_id: str | None) -> str:
    if frame_id is not None:
        candidate = image_dir / f"{frame_id}.png"
        if candidate.exists():
            return frame_id
        raise FileNotFoundError(f"Frame not found: {candidate}")
    frames = sorted(p.stem for p in image_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {image_dir}")
    return frames[0]


def _add_label(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    pad_h = 44
    canvas = np.zeros((out.shape[0] + pad_h, out.shape[1], 3), dtype=np.uint8)
    canvas[pad_h:, :] = out
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, pad_h - 1), (18, 18, 18), -1)
    cv2.putText(canvas, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _make_pair_panel(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        target_h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, dsize=None, fx=target_h / left.shape[0], fy=target_h / left.shape[0], interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, dsize=None, fx=target_h / right.shape[0], fy=target_h / right.shape[0], interpolation=cv2.INTER_AREA)
    pair = np.concatenate([left, right], axis=1)
    return _add_label(pair, label)


def _tile_contact_sheet(pairs: list[tuple[str, np.ndarray]], columns: int, thumb_width: int) -> np.ndarray:
    thumbs = []
    for label, img in pairs:
        scale = thumb_width / float(img.shape[1])
        thumb = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        thumb = _add_label(thumb, label)
        thumbs.append(thumb)

    if not thumbs:
        raise RuntimeError("No thumbnails to tile")

    max_h = max(im.shape[0] for im in thumbs)
    max_w = max(im.shape[1] for im in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = np.zeros((rows * max_h, columns * max_w, 3), dtype=np.uint8)
    sheet[:] = (0, 0, 0)

    for idx, thumb in enumerate(thumbs):
        r = idx // columns
        c = idx % columns
        y = r * max_h
        x = c * max_w
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create lowres pair montage for DNA-Rendering")
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequence", type=str, default="0007_04")
    parser.add_argument("--frame-id", type=str, default=None, help="Use a specific frame; default is the first frame")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--thumb-width", type=int, default=900)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seq_dir = args.data_root / args.sequence
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    output_dir = args.output_dir or (seq_dir / "fs_depth_debug" / "lowres_pair_montage")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_id = args.frame_id
    pair_panels: list[tuple[str, np.ndarray]] = []
    print(f"Using frame: {frame_id or 'first shared frame per pair'}")

    for idx in range(len(LOWRES_VIEWS) - 1):
        left_view = LOWRES_VIEWS[idx]
        right_view = LOWRES_VIEWS[idx + 1]
        left_img_dir = seq_dir / "images" / left_view
        right_img_dir = seq_dir / "images" / right_view

        chosen_frame = frame_id
        if chosen_frame is None:
            left_frame = _find_frame(left_img_dir, None)
            right_frame = _find_frame(right_img_dir, None)
            shared = sorted({p.stem for p in left_img_dir.glob("*.png")} & {p.stem for p in right_img_dir.glob("*.png")})
            if not shared:
                print(f"[WARN] no shared frames for {left_view}->{right_view}")
                continue
            chosen_frame = shared[0]
        else:
            if not (left_img_dir / f"{chosen_frame}.png").exists() or not (right_img_dir / f"{chosen_frame}.png").exists():
                print(f"[WARN] missing frame {chosen_frame} for {left_view}->{right_view}")
                continue

        left = _load_image(left_img_dir / f"{chosen_frame}.png")
        right = _load_image(right_img_dir / f"{chosen_frame}.png")
        label = f"{idx + 1:02d}  {left_view}-{right_view}  frame {chosen_frame}"
        panel = _make_pair_panel(left, right, label)
        pair_panels.append((label, panel))
        cv2.imwrite(str(output_dir / f"{idx + 1:02d}_{left_view}_{right_view}_{chosen_frame}.png"), panel)
        print(f"[OK] {label}")

    sheet = _tile_contact_sheet(pair_panels, columns=args.columns, thumb_width=args.thumb_width)
    sheet_path = output_dir / "lowres_pair_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"[OK] wrote {sheet_path}")


if __name__ == "__main__":
    main()
