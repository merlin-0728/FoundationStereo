#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def _load_cam_npz(seq_dir: Path, cam_id: int, frame_id: int):
    cam_file = seq_dir / "cameras" / f"{cam_id:02d}" / f"{frame_id:06d}.npz"
    if not cam_file.exists():
        raise FileNotFoundError(f"Camera file not found: {cam_file}")
    data = np.load(str(cam_file), allow_pickle=True)
    required = ["K", "D", "RT"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {cam_file}: {missing}")
    return cam_file, data


def _to_4x4(rt: np.ndarray) -> np.ndarray:
    if rt.shape == (4, 4):
        return rt.astype(np.float64)
    if rt.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = rt.astype(np.float64)
        return out
    raise ValueError(f"Unsupported RT shape: {rt.shape}, expect (3,4) or (4,4)")


def _get_w2c(rt: np.ndarray, rt_type: str) -> np.ndarray:
    rt4 = _to_4x4(rt)
    if rt_type == "w2c":
        return rt4
    if rt_type == "c2w":
        return np.linalg.inv(rt4)
    raise ValueError(f"Unknown rt_type: {rt_type}")


def _write_k_txt(path: Path, K: np.ndarray, baseline_m: float):
    flat = " ".join(f"{v:.15g}" for v in K.reshape(-1))
    with open(path, "w", encoding="utf-8") as f:
        f.write(flat + "\n")
        f.write(f"{baseline_m:.15g}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare rectified stereo pair + K.txt for FoundationStereo from DNA-Rendering sequence.")
    parser.add_argument("--seq_dir", required=True, type=str,
                        help="Path to DNA-Rendering sequence, e.g. /.../DNA-Rendering/0007_04")
    parser.add_argument("--left_cam", required=True, type=int, help="Left camera id, e.g. 46")
    parser.add_argument("--right_cam", required=True, type=int, help="Right camera id, e.g. 48")
    parser.add_argument("--frame", default=0, type=int, help="Frame id, e.g. 0 for 000000")
    parser.add_argument("--rt_type", default="c2w", choices=["c2w", "w2c"],
                        help="Interpretation of RT stored in npz")
    parser.add_argument("--alpha", default=0.0, type=float,
                        help="stereoRectify alpha, 0=crop to valid area, 1=keep all")
    parser.add_argument("--match_size", default="left", choices=["left", "right"],
                        help="If left/right image sizes differ, resize the other side to match this side")
    parser.add_argument("--assume_undistorted", default=1, type=int,
                        help="If 1, ignore D and treat input images as already undistorted (recommended for DNA-Rendering)")
    parser.add_argument("--crop_to_intersection", default=1, type=int,
                        help="If 1, crop rectified pair to valid ROI intersection to avoid large black regions")
    parser.add_argument("--min_intersection_ratio", default=0.02, type=float,
                        help="Minimal valid ROI intersection ratio; fail if below this threshold")
    parser.add_argument("--require_horizontal", default=1, type=int,
                        help="If 1, require horizontal rectification (P2[0,3] dominates P2[1,3]) for FoundationStereo")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="Output directory. Default: <seq_dir>/foundationstereo_pairs")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir).resolve()
    if not seq_dir.exists():
        raise FileNotFoundError(f"seq_dir not found: {seq_dir}")

    out_root = Path(args.out_dir).resolve() if args.out_dir else (seq_dir / "foundationstereo_pairs")
    pair_name = f"cam{args.left_cam:02d}_cam{args.right_cam:02d}_frame{args.frame:06d}"
    out_dir = out_root / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    left_img_path = seq_dir / "images" / f"{args.left_cam:02d}" / f"{args.frame:06d}.png"
    right_img_path = seq_dir / "images" / f"{args.right_cam:02d}" / f"{args.frame:06d}.png"
    if not left_img_path.exists() or not right_img_path.exists():
        raise FileNotFoundError(f"Input images not found:\n{left_img_path}\n{right_img_path}")

    _, left_cam = _load_cam_npz(seq_dir, args.left_cam, args.frame)
    _, right_cam = _load_cam_npz(seq_dir, args.right_cam, args.frame)

    Kl = left_cam["K"].astype(np.float64)
    Dl = left_cam["D"].astype(np.float64).reshape(-1)
    Kr = right_cam["K"].astype(np.float64)
    Dr = right_cam["D"].astype(np.float64).reshape(-1)

    w2c_l = _get_w2c(left_cam["RT"], args.rt_type)
    w2c_r = _get_w2c(right_cam["RT"], args.rt_type)
    Rl, tl = w2c_l[:3, :3], w2c_l[:3, 3]
    Rr, tr = w2c_r[:3, :3], w2c_r[:3, 3]

    # Relative transform from left camera to right camera in the OpenCV convention.
    R = Rr @ Rl.T
    T = tr - R @ tl

    left = imageio.imread(left_img_path)
    right = imageio.imread(right_img_path)
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    if (h_l, w_l) != (h_r, w_r):
        if args.match_size == "left":
            sx = w_l / float(w_r)
            sy = h_l / float(h_r)
            Kr = Kr.copy()
            Kr[0, :] *= sx
            Kr[1, :] *= sy
            right = cv2.resize(right, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
            h, w = h_l, w_l
            print(f"[INFO] resized right image from {(h_r, w_r)} to {(h_l, w_l)}")
        else:
            sx = w_r / float(w_l)
            sy = h_r / float(h_l)
            Kl = Kl.copy()
            Kl[0, :] *= sx
            Kl[1, :] *= sy
            left = cv2.resize(left, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
            h, w = h_r, w_r
            print(f"[INFO] resized left image from {(h_l, w_l)} to {(h_r, w_r)}")
    else:
        h, w = h_l, w_l

    image_size = (w, h)

    d_left = np.zeros_like(Dl) if args.assume_undistorted else Dl
    d_right = np.zeros_like(Dr) if args.assume_undistorted else Dr

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, d_left, Kr, d_right, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(args.alpha),
    )

    map1x, map1y = cv2.initUndistortRectifyMap(Kl, d_left, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(Kr, d_right, R2, P2, image_size, cv2.CV_32FC1)

    left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    # Check valid intersection ROI.
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    xi = max(x1, x2)
    yi = max(y1, y2)
    xa = min(x1 + w1, x2 + w2)
    ya = min(y1 + h1, y2 + h2)
    iw = max(0, xa - xi)
    ih = max(0, ya - yi)
    inter_ratio = (iw * ih) / float(w * h)
    if inter_ratio < float(args.min_intersection_ratio):
        raise ValueError(
            f"ROI intersection too small: {inter_ratio:.4f} ({iw}x{ih}) for image size {w}x{h}. "
            "Choose another camera pair.")

    # FoundationStereo expects horizontal disparity.
    baseline_x = abs(P2[0, 3] / P2[0, 0]) if abs(P2[0, 0]) > 1e-9 else 0.0
    baseline_y = abs(P2[1, 3] / P2[1, 1]) if abs(P2[1, 1]) > 1e-9 else 0.0
    horizontal = abs(P2[0, 3]) >= abs(P2[1, 3])
    if args.require_horizontal and not horizontal:
        raise ValueError(
            f"Rectification is vertical-dominant (baseline_y={baseline_y:.6f}, baseline_x={baseline_x:.6f}). "
            "FoundationStereo expects horizontal stereo; choose another pair.")
    baseline = baseline_x if horizontal else baseline_y

    K_rect = P1[:3, :3].copy()

    # Optional crop to valid intersection to reduce black margins.
    if args.crop_to_intersection:
        left_rect = left_rect[yi:yi + ih, xi:xi + iw]
        right_rect = right_rect[yi:yi + ih, xi:xi + iw]
        K_rect[0, 2] -= float(xi)
        K_rect[1, 2] -= float(yi)

    left_out = out_dir / "left_rect.png"
    right_out = out_dir / "right_rect.png"
    k_out = out_dir / "K.txt"
    npz_out = out_dir / "rectify_meta.npz"

    imageio.imwrite(left_out, left_rect)
    imageio.imwrite(right_out, right_rect)
    _write_k_txt(k_out, K_rect, float(baseline))
    np.savez(
        npz_out,
        K_rect=K_rect,
        baseline_m=baseline,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
    )

    print(f"[OK] saved: {left_out}")
    print(f"[OK] saved: {right_out}")
    print(f"[OK] saved: {k_out}")
    print(f"[OK] baseline(m): {baseline:.6f}")
    print(f"[OK] intersection ratio: {inter_ratio:.4f} ({iw}x{ih} / {w}x{h})")
    print(f"[OK] rectification mode: {'horizontal' if horizontal else 'vertical'}")
    print(f"[OK] K_rect:\n{K_rect}")


if __name__ == "__main__":
    main()
