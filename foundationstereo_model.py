"""FoundationStereo convenience helpers for DNA-Rendering.

This module wraps model loading, stereo rectification, and disparity/depth
inference so the SeqAvatar pipeline can call FoundationStereo consistently.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


def load_model(fs_root: str | Path, device: str = "cuda",
               ckpt_relpath: str = "pretrained_models/23-51-11/model_best_bp2.pth") -> torch.nn.Module:
    """Load a FoundationStereo checkpoint.

    Args:
        fs_root: FoundationStereo repository root.
        device: Torch device string.
        ckpt_relpath: Relative path to the checkpoint file under ``fs_root``.
    """
    fs_root = Path(fs_root)
    ckpt_file = fs_root / ckpt_relpath
    cfg_file = ckpt_file.with_name("cfg.yaml")

    cfg = OmegaConf.load(str(cfg_file))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    model = FoundationStereo(cfg)
    ckpt = torch.load(str(ckpt_file), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def _ensure_3ch(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        return image[..., :3]
    if image.shape[2] == 3:
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


def load_camera_npz(camera_path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(str(camera_path), allow_pickle=True)
    required = ("K", "D", "RT")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {camera_path}: {missing}")
    return {k: data[k] for k in data.files}


def rectify_stereo_pair(left_img: np.ndarray,
                        right_img: np.ndarray,
                        left_cam: Dict[str, np.ndarray],
                        right_cam: Dict[str, np.ndarray],
                        rt_type: str = "c2w",
                        assume_undistorted: bool = True,
                        alpha: float = 0.0,
                        crop_to_intersection: bool = True,
                        min_intersection_ratio: float = 0.02,
                        require_horizontal: bool = True,
                        rotate_vertical: bool = True,
                        match_size: str = "left") -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Rectify a stereo pair and return rectified images plus metric baseline."""
    left = _ensure_3ch(left_img)
    right = _ensure_3ch(right_img)

    Kl = left_cam["K"].astype(np.float64).copy()
    Dl = left_cam["D"].astype(np.float64).reshape(-1)
    Kr = right_cam["K"].astype(np.float64).copy()
    Dr = right_cam["D"].astype(np.float64).reshape(-1)

    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    if (h_l, w_l) != (h_r, w_r):
        if match_size == "left":
            sx = w_l / float(w_r)
            sy = h_l / float(h_r)
            Kr[0, :] *= sx
            Kr[1, :] *= sy
            right = cv2.resize(right, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
        elif match_size == "right":
            sx = w_r / float(w_l)
            sy = h_r / float(h_l)
            Kl[0, :] *= sx
            Kl[1, :] *= sy
            left = cv2.resize(left, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unknown match_size: {match_size}")

    if assume_undistorted:
        Dl = np.zeros_like(Dl)
        Dr = np.zeros_like(Dr)

    w2c_l = _w2c_from_rt(left_cam["RT"], rt_type)
    w2c_r = _w2c_from_rt(right_cam["RT"], rt_type)
    Rl, tl = w2c_l[:3, :3], w2c_l[:3, 3]
    Rr, tr = w2c_r[:3, :3], w2c_r[:3, 3]
    R = Rr @ Rl.T
    T = tr - R @ tl

    image_size = (left.shape[1], left.shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    map1x, map1y = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    xi = max(x1, x2)
    yi = max(y1, y2)
    xa = min(x1 + w1, x2 + w2)
    ya = min(y1 + h1, y2 + h2)
    iw = max(0, xa - xi)
    ih = max(0, ya - yi)
    inter_ratio = (iw * ih) / float(image_size[0] * image_size[1])
    if inter_ratio < float(min_intersection_ratio):
        raise ValueError(
            f"ROI intersection too small: {inter_ratio:.4f} ({iw}x{ih}) for image size {image_size[0]}x{image_size[1]}."
        )

    if crop_to_intersection and iw > 0 and ih > 0:
        left_rect = left_rect[yi:yi + ih, xi:xi + iw]
        right_rect = right_rect[yi:yi + ih, xi:xi + iw]
        P1 = P1.copy()
        P1[0, 2] -= float(xi)
        P1[1, 2] -= float(yi)

    horizontal = abs(P2[0, 3]) >= abs(P2[1, 3])
    baseline_x = abs(P2[0, 3] / P2[0, 0]) if abs(P2[0, 0]) > 1e-9 else 0.0
    baseline_y = abs(P2[1, 3] / P2[1, 1]) if abs(P2[1, 1]) > 1e-9 else 0.0
    baseline = baseline_x if horizontal else baseline_y
    K_rect = P1[:3, :3].copy()

    rotation = "none"
    rectification_mode = "horizontal" if horizontal else "vertical"
    if not horizontal and require_horizontal and not rotate_vertical:
        raise ValueError(
            f"Vertical rectification detected (P2[0,3]={P2[0,3]:.6f}, P2[1,3]={P2[1,3]:.6f})."
        )

    if not horizontal and rotate_vertical:
        rect_h, rect_w = left_rect.shape[:2]
        fx, fy = K_rect[0, 0], K_rect[1, 1]
        cx, cy = K_rect[0, 2], K_rect[1, 2]
        if P2[1, 3] < 0:
            left_rect = cv2.rotate(left_rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
            right_rect = cv2.rotate(right_rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
            K_rect = np.array(
                [[fy, 0.0, cy],
                 [0.0, fx, rect_w - 1.0 - cx],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rotation = "ccw"
        else:
            left_rect = cv2.rotate(left_rect, cv2.ROTATE_90_CLOCKWISE)
            right_rect = cv2.rotate(right_rect, cv2.ROTATE_90_CLOCKWISE)
            K_rect = np.array(
                [[fy, 0.0, rect_h - 1.0 - cy],
                 [0.0, fx, cx],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rotation = "cw"
        rectification_mode = "vertical_rotated"
        baseline = baseline_y

    meta = {
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": np.array(roi1),
        "roi2": np.array(roi2),
        "baseline_x": baseline_x,
        "baseline_y": baseline_y,
        "baseline_m": baseline,
        "intersection_ratio": inter_ratio,
        "horizontal": horizontal,
        "rectification_mode": rectification_mode,
        "rotation": rotation,
        "K_rect": K_rect,
    }
    return left_rect, right_rect, K_rect, baseline, meta


def _prepare_tensor(image: np.ndarray, device: str, input_color: str) -> torch.Tensor:
    image = _ensure_3ch(np.asarray(image))
    if input_color.lower() == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif input_color.lower() != "rgb":
        raise ValueError(f"Unknown input_color: {input_color}")
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)[None].permute(0, 3, 1, 2)
    return tensor.contiguous()


def predict_disparity(left_img: np.ndarray,
                      right_img: np.ndarray,
                      model: torch.nn.Module,
                      device: str = "cuda",
                      valid_iters: int = 32,
                      input_color: str = "bgr",
                      use_amp: bool = True) -> np.ndarray:
    """Predict disparity for an already-rectified stereo pair."""
    left_tensor = _prepare_tensor(left_img, device, input_color)
    right_tensor = _prepare_tensor(right_img, device, input_color)

    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    amp_enabled = use_amp and str(device).startswith("cuda") and torch.cuda.is_available()
    amp_ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
    with torch.no_grad():
        with amp_ctx:
            disp = model.forward(left_tensor, right_tensor, iters=valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    return disp.squeeze().detach().cpu().numpy().astype(np.float32)


def disparity_to_depth(disp: np.ndarray, K_rect: np.ndarray, baseline_m: float,
                       invalid_value: float = 1e-4) -> np.ndarray:
    """Convert disparity to metric depth."""
    depth = (float(K_rect[0, 0]) * float(baseline_m)) / np.maximum(disp.astype(np.float32), invalid_value)
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[disp <= invalid_value] = 0.0
    return depth
