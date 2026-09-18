"""One-frame-only nuisance corruptions (evaluation suite).

Hard rule: apply every corruption to exactly one frame of the pair.
Corrupting both identically tests generic image quality, not nuisance
suppression. Call ``apply_to_pair(..., which='t1'|'t2')`` and run both
directions.

P1's training-negative mining should import these same functions — one
implementation, two callers. See docs/nuisance_label.md for the taxonomy.

Native cv2/numpy implementations so CI does not require albumentations
or imagecorruptions. Severity 1→5 follows research/05 §4.2.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

WhichFrame = Literal["t1", "t2"]

# nuisance_label taxonomy — agree with P1 before changing ids.
NUISANCE_ID = {
    "clean": 0,
    "brightness": 1,
    "gamma": 1,
    "colour_grading": 2,
    "gaussian_blur": 3,
    "motion_blur": 4,
    "jpeg": 5,
    "h264": 6,
    "hevc": 7,
    "shadow": 8,
    "viewpoint": 9,
    "occlusion": 10,
    "unknown": -1,
}

SEVERITY = (1, 2, 3, 4, 5)


def _chw_to_hwc_u8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _hwc_u8_to_chw(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64) / 255.0
    if x.ndim == 2:
        x = x[..., None]
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def _as_chw(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return np.transpose(x, (2, 0, 1))
    return x


def brightness(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive HSV-V shift, ImageNet-C-ish 0.1→0.5."""
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    x = _as_chw(img)
    return np.clip(x + sign * delta, 0.0, 1.0).astype(np.float32)


def gamma(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    ranges = [(0.85, 1.18), (0.7, 1.4), (0.55, 1.8), (0.4, 2.2), (0.25, 3.0)]
    lo, hi = ranges[severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    g = float(rng.uniform(lo, hi))
    x = np.clip(_as_chw(img), 0.0, 1.0)
    return np.power(x, 1.0 / g).astype(np.float32)


def colour_grading(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """ASC-CDL: out = clip(slope*in + offset, 0, 1) ** power, then saturation."""
    jitter = [0.03, 0.06, 0.10, 0.15, 0.25][severity - 1]
    power_span = [0.1, 0.2, 0.3, 0.45, 0.6][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    slope = 1.0 + rng.uniform(-jitter, jitter, size=(3, 1, 1))
    offset = rng.uniform(-jitter, jitter, size=(3, 1, 1))
    power = 1.0 + rng.uniform(-power_span, power_span, size=(3, 1, 1))
    sat = 1.0 + rng.uniform(-jitter, jitter)
    x = np.clip(_as_chw(img), 0.0, 1.0)
    out = np.clip(slope * x + offset, 0.0, 1.0) ** np.clip(power, 0.3, 3.0)
    luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    out = luma + sat * (out - luma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gaussian_blur(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    sigma = [0.5, 1.0, 2.0, 3.0, 4.0][severity - 1]
    hwc = _chw_to_hwc_u8(img)
    k = max(3, int(2 * round(3 * sigma) + 1))
    out = cv2.GaussianBlur(hwc, (k, k), sigmaX=sigma)
    return _hwc_u8_to_chw(out)


def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    length = [3, 7, 15, 25, 35][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    theta = float(rng.uniform(0, np.pi))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2, length / 2), np.degrees(theta), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    hwc = _chw_to_hwc_u8(img)
    out = cv2.filter2D(hwc, -1, kernel)
    return _hwc_u8_to_chw(out)


def jpeg(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    quality = [90, 70, 50, 30, 10][severity - 1]
    hwc = _chw_to_hwc_u8(img)
    ok, buf = cv2.imencode(".jpg", hwc, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return _as_chw(img)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return _hwc_u8_to_chw(dec)


def _ffmpeg_roundtrip(img: np.ndarray, codec: str, crf: int, cache_dir: Path | None) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH — skip codec corruptions in this environment")
    hwc = _chw_to_hwc_u8(img)
    key = hashlib.sha1(hwc.tobytes() + f"{codec}:{crf}".encode()).hexdigest()[:16]
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{key}.npy"
        if cached.exists():
            return np.load(cached)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        src = td_p / "frame.png"
        clip = td_p / "clip.mp4"
        out = td_p / "frame_c.png"
        cv2.imwrite(str(src), hwc)
        enc = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-loop", "1", "-i", str(src), "-t", "1", "-r", "10",
               "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p", str(clip)]
        subprocess.run(enc, check=True)
        dec = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(clip), "-vf", "select=eq(n\\,5)", "-vframes", "1", str(out)]
        subprocess.run(dec, check=True)
        dec_img = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if dec_img is None:
            raise RuntimeError("ffmpeg produced no frame")
        result = _hwc_u8_to_chw(dec_img)
    if cache_dir is not None:
        np.save(cache_dir / f"{key}.npy", result)
    return result


_H264_CRF = [18, 23, 28, 35, 45]
_HEVC_CRF = [20, 25, 30, 37, 47]


def h264(img: np.ndarray, severity: int, rng: np.random.Generator | None = None,
         cache_dir: Path | None = None) -> np.ndarray:
    return _ffmpeg_roundtrip(img, "libx264", _H264_CRF[severity - 1], cache_dir)


def hevc(img: np.ndarray, severity: int, rng: np.random.Generator | None = None,
         cache_dir: Path | None = None) -> np.ndarray:
    return _ffmpeg_roundtrip(img, "libx265", _HEVC_CRF[severity - 1], cache_dir)


def shadow(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    intensity = [0.1, 0.2, 0.35, 0.5, 0.7][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    x = _as_chw(img).copy()
    _, h, w = x.shape
    n_poly = int(rng.integers(1, 3))
    overlay = np.ones((h, w), dtype=np.float32)
    for _ in range(n_poly):
        n_pts = int(rng.integers(3, 6))
        pts = np.stack(
            [rng.integers(0, w, size=n_pts), rng.integers(0, h, size=n_pts)], axis=1
        ).astype(np.int32)
        cv2.fillPoly(overlay, [pts], color=1.0 - intensity)
    x *= overlay
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def viewpoint(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    translate = [0.01, 0.02, 0.04, 0.08, 0.16][severity - 1]
    rotate = [0.5, 1.0, 2.0, 4.0, 8.0][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    hwc = _chw_to_hwc_u8(img)
    h, w = hwc.shape[:2]
    tx = translate * w * float(rng.uniform(-1, 1))
    ty = translate * h * float(rng.uniform(-1, 1))
    ang = rotate * float(rng.uniform(-1, 1))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(hwc, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return _hwc_u8_to_chw(out)


def occlusion(img: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    frac = [0.02, 0.05, 0.10, 0.20, 0.35][severity - 1]
    if rng is None:
        rng = np.random.default_rng(0)
    x = _as_chw(img).copy()
    _, h, w = x.shape
    area = frac * h * w
    rh = max(1, int(np.sqrt(area) * rng.uniform(0.5, 1.5)))
    rw = max(1, int(area / rh))
    rh, rw = min(rh, h), min(rw, w)
    y0 = int(rng.integers(0, h - rh + 1))
    x0 = int(rng.integers(0, w - rw + 1))
    fill = float(rng.uniform(0, 1))
    x[:, y0 : y0 + rh, x0 : x0 + rw] = fill
    return x


CORRUPTIONS: dict[str, Callable[..., np.ndarray]] = {
    "brightness": brightness,
    "gamma": gamma,
    "colour_grading": colour_grading,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg": jpeg,
    "h264": h264,
    "hevc": hevc,
    "shadow": shadow,
    "viewpoint": viewpoint,
    "occlusion": occlusion,
}

EVAL_CORRUPTIONS = tuple(k for k in CORRUPTIONS if k not in ("h264", "hevc"))
CODEC_CORRUPTIONS = ("h264", "hevc")


def apply_corruption(
    img: np.ndarray,
    name: str,
    severity: int,
    rng: np.random.Generator | None = None,
    cache_dir: Path | None = None,
) -> np.ndarray:
    if severity not in SEVERITY:
        raise ValueError(f"severity must be 1–5, got {severity}")
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    fn = CORRUPTIONS[name]
    if name in CODEC_CORRUPTIONS:
        return fn(img, severity, rng=rng, cache_dir=cache_dir)
    return fn(img, severity, rng=rng)


def apply_to_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    name: str,
    severity: int,
    which: WhichFrame = "t2",
    rng: np.random.Generator | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt exactly one frame. Never both."""
    if which == "t2":
        return img1, apply_corruption(img2, name, severity, rng=rng, cache_dir=cache_dir)
    if which == "t1":
        return apply_corruption(img1, name, severity, rng=rng, cache_dir=cache_dir), img2
    raise ValueError("which must be 't1' or 't2' — corrupting both frames is forbidden")
