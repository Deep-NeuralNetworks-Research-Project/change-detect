"""Nuisance corruption primitives — one implementation, two callers.

``research/05`` §4.2 specifies a corruption suite for P5's robustness benchmark, and
P1 needs the *same* corruptions to synthesise nuisance-only hard negatives for
training (P1 role file, weeks 3-5). Building it twice would guarantee the training
nuisances and the evaluation nuisances drift apart, so the primitives live here, in
the data tree, and ``cdlib.metrics.robustness`` (P5) imports them. ``metrics -> data``
is the natural dependency direction; ``data -> metrics`` would be an import cycle.

**The split of responsibility, which is recorded nowhere else in the project:**

* **P5 owns the discrete severity 1-5 grid.** The functions here take *continuous*
  parameters only. :data:`SEVERITY_REFERENCE` carries the ``research/05`` §4.2
  schedule as inert data so P5 can build its grid from the same numbers, but nothing
  in this module consumes it.
* **P1's training negatives sample continuous ranges from a separate RNG stream.**
  Training on P5's exact severity parameters would make the robustness-retention
  table (``research/05`` §4.5) train-on-test: the model would have seen
  ``sigma = 2.0`` blur at training time and the "severity 3" column would measure
  memorisation, not robustness. Sample from intervals that *straddle* the grid
  instead, and keep the training RNG distinct from the evaluation RNG.

**Two call patterns over one implementation:**

* **P5 (evaluation)** — :func:`apply_nuisance` applies one corruption to exactly one
  frame. ``research/05`` §4.1: corrupting both frames identically lets a Siamese
  difference cancel it as common-mode, so the test would under-report the failure
  mode. Asymmetry is the point.
* **P1 (training)** — :class:`cdlib.data.transforms.IndependentPhotometric` applies
  photometric augmentation to *both* frames, independently sampled. Both frames get
  a corruption; they never get the *same* corruption. That is the nuisance signal the
  model must learn to ignore (root ``CLAUDE.md`` rule 4).

Every function takes ``HWC uint8`` RGB and returns ``HWC uint8`` RGB. Randomness is
drawn from an explicit :class:`numpy.random.Generator`; nothing here touches global
RNG state.

No ``imagecorruptions`` dependency: it is not in ``pyproject.toml`` and adding it for
five functions is not worth the install. Where ``research/05`` §4.2 names an
ImageNet-C schedule, the equivalent numbers are recorded in
:data:`SEVERITY_REFERENCE` and the corruption is implemented with cv2/numpy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from cdlib.data.contract import IGNORE_INDEX, NuisanceLabel

# --------------------------------------------------------------------------------------
# Severity reference — data for P5, never consumed here
# --------------------------------------------------------------------------------------

#: The ``research/05`` §4.2 severity 1-5 schedule, verbatim, as data.
#:
#: **P1 must not call the functions below with these values at training time.** They
#: are published here so P5's ``metrics/robustness.py`` builds its discrete grid from
#: the same source of truth this module was written against, and so a reviewer can
#: diff the code against the brief without opening the brief.
#:
#: ``source`` distinguishes numbers ``research/05`` states explicitly from the two rows
#: where it defers to "the package default 5-level schedule" — those are ImageNet-C's
#: published constants, recorded here so the deferral does not become a dangling
#: reference to a package we do not install.
SEVERITY_REFERENCE: dict[str, dict[str, Any]] = {
    "gamma": {
        "parameter": "gamma range (symmetric under/over-exposure)",
        "values": ((0.85, 1.18), (0.7, 1.4), (0.55, 1.8), (0.4, 2.2), (0.25, 3.0)),
        "source": "research/05 §4.2, explicit",
    },
    "brightness": {
        "parameter": "additive brightness fraction (ImageNet-C HSV-V shift)",
        "values": (0.1, 0.2, 0.3, 0.4, 0.5),
        "source": "research/05 §4.2 ('≈0.1→0.5'), ImageNet-C default",
    },
    "contrast": {
        "parameter": "contrast multiplier about the mean",
        "values": (0.4, 0.3, 0.2, 0.1, 0.05),
        "source": "research/05 §4.2 defers to the ImageNet-C default schedule",
    },
    "colour_grade": {
        "parameter": "slope/offset jitter fraction, power range",
        "values": (
            {"jitter": 0.03, "power": (0.9, 1.1)},
            {"jitter": 0.06, "power": (0.85, 1.2)},
            {"jitter": 0.10, "power": (0.8, 1.3)},
            {"jitter": 0.15, "power": (0.7, 1.45)},
            {"jitter": 0.25, "power": (0.6, 1.6)},
        ),
        "source": "research/05 §4.2; the intermediate power steps interpolate the "
        "stated 0.9-1.1 -> 0.6-1.6 endpoints",
    },
    "gaussian_blur": {
        "parameter": "sigma, px",
        "values": (0.5, 1.0, 2.0, 3.0, 4.0),
        "source": "research/05 §4.2, explicit",
    },
    "motion_blur": {
        "parameter": "kernel length L, px (angle uniform in [0,180) per sample)",
        "values": (3, 7, 15, 25, 35),
        "source": "research/05 §4.2, explicit",
    },
    "jpeg": {
        "parameter": "JPEG quality",
        "values": (90, 70, 50, 30, 10),
        "source": "research/05 §4.2, explicit",
    },
    "h264": {
        "parameter": "libx264 CRF",
        "values": (18, 23, 28, 35, 45),
        "source": "research/05 §4.2, explicit",
    },
    "hevc": {
        "parameter": "libx265 CRF",
        "values": (20, 25, 30, 37, 47),
        "source": "research/05 §4.2, explicit",
    },
    "shadow": {
        "parameter": "shadow intensity fraction (area fraction grows with severity)",
        "values": (0.1, 0.2, 0.35, 0.5, 0.7),
        "source": "research/05 §4.2, explicit",
    },
    "viewpoint": {
        "parameter": "translate fraction of W/H, rotate deg, corner jitter fraction",
        "values": (
            {"translate": 0.01, "rotate": 0.5, "corner_jitter": 0.005},
            {"translate": 0.02, "rotate": 1.0, "corner_jitter": 0.01},
            {"translate": 0.04, "rotate": 2.0, "corner_jitter": 0.02},
            {"translate": 0.08, "rotate": 4.0, "corner_jitter": 0.04},
            {"translate": 0.16, "rotate": 8.0, "corner_jitter": 0.08},
        ),
        "source": "research/05 §4.2, explicit",
    },
    "occlusion": {
        "parameter": "occluded area fraction of the frame",
        "values": (0.02, 0.05, 0.10, 0.20, 0.35),
        "source": "research/05 §4.2, explicit",
    },
    "sensor_noise": {
        "parameter": "noise sigma in [0,1] units",
        "values": (0.08, 0.12, 0.18, 0.26, 0.38),
        "source": "research/05 §4.2 defers to the ImageNet-C default schedule " "(optional row)",
    },
}

#: Corruption name -> the :class:`~cdlib.data.contract.NuisanceLabel` a sample carries
#: after it. P1 stamps ``nuisance_label`` on mined hard negatives with this; P5's
#: stratified rows key off the same mapping, so a stratum and a corruption row cannot
#: drift apart.
CORRUPTION_NUISANCE: dict[str, NuisanceLabel] = {
    "gamma": NuisanceLabel.LIGHTING,
    "brightness_contrast": NuisanceLabel.LIGHTING,
    "colour_grade": NuisanceLabel.COLOUR_GRADING,
    "gaussian_blur": NuisanceLabel.BLUR,
    "motion_blur": NuisanceLabel.BLUR,
    "jpeg": NuisanceLabel.CODEC_ARTIFACT,
    "video_codec": NuisanceLabel.CODEC_ARTIFACT,
    "shadow": NuisanceLabel.SHADOW,
    "viewpoint": NuisanceLabel.CAMERA_DISPLACEMENT,
    "occlusion": NuisanceLabel.OCCLUSION,
    "sensor_noise": NuisanceLabel.SENSOR_NOISE,
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous ``HWC uint8`` array."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_float(img: np.ndarray) -> np.ndarray:
    return _as_uint8_rgb(img).astype(np.float32) / 255.0


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clamp-and-round back to ``uint8``. Rounding, not truncation: ``astype(np.uint8)``
    on ``x*255`` loses up to one level per channel and biases every corruption dark."""
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def _rng_of(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Never seeds, never reads global NumPy state when a generator is supplied."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _luma(img_f: np.ndarray) -> np.ndarray:
    """Rec.709 luma of a float RGB image, kept as ``HW1`` for broadcasting."""
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (img_f * w).sum(axis=2, keepdims=True)


# --------------------------------------------------------------------------------------
# Photometric — lighting and colour
# --------------------------------------------------------------------------------------


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct an image: ``out = 255 * (img/255) ** (1/gamma)``.

    ``research/05`` §4.2 prefers explicit gamma over ImageNet-C's HSV-V shift "for
    domain realism" — a camera exposure change is a transfer-curve change, not an
    additive value offset.

    Args:
        img: ``HWC uint8`` RGB.
        gamma: Exposure factor. ``> 1`` brightens, ``< 1`` darkens, ``1`` is identity.
            §4.2 severity ranges span ``0.25`` to ``3.0``.

    Returns:
        ``HWC uint8`` RGB.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    # A 256-entry LUT is exact for uint8 input and ~20x faster than powering the array.
    table = np.clip(np.rint(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0), 0, 255)
    return cv2.LUT(_as_uint8_rgb(img), table.astype(np.uint8))


def adjust_brightness_contrast(
    img: np.ndarray, brightness: float = 0.0, contrast: float = 0.0
) -> np.ndarray:
    """Additive brightness and multiplicative contrast about the per-image mean.

    Args:
        img: ``HWC uint8`` RGB.
        brightness: Additive shift in ``[0,1]`` units. §4.2's ImageNet-C brightness
            row is ``0.1 -> 0.5``.
        contrast: Contrast delta. ``out = mean + (in - mean) * (1 + contrast)``, so
            ``contrast = -0.6`` reproduces ImageNet-C's severity-1 multiplier of
            ``0.4`` recorded in :data:`SEVERITY_REFERENCE`.

    Returns:
        ``HWC uint8`` RGB.
    """
    f = _to_float(img)
    # Contrast about the *image* mean, not about 0.5: a dark frame contrast-reduced
    # toward mid-grey would brighten, which is a brightness change wearing a contrast
    # label and would contaminate P5's per-nuisance strata.
    mean = float(f.mean())
    out = mean + (f - mean) * (1.0 + contrast) + brightness
    return _to_uint8(out)


def colour_grade(
    img: np.ndarray,
    slope: float | tuple[float, float, float] = 1.0,
    offset: float | tuple[float, float, float] = 0.0,
    power: float | tuple[float, float, float] = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """ASC-CDL colour grade plus a saturation blend, per ``research/05`` §4.2.

    ``out = clip(slope*in + offset, 0, 1) ** power`` per channel, then
    ``out = luma + saturation * (out - luma)``. This is the actual operator a colourist
    applies, which is why §4.2 specifies it rather than a hue/saturation jitter: our
    "colour grading / white-balance drift" nuisance comes from a re-grade between two
    versions of the same footage, not from a sensor hue shift.

    Args:
        img: ``HWC uint8`` RGB.
        slope: Per-channel gain (scalar broadcasts). §4.2 jitters it by ±3% to ±25%.
        offset: Per-channel lift in ``[0,1]`` units (scalar broadcasts).
        power: Per-channel gamma (scalar broadcasts). §4.2 spans ``0.6`` to ``1.6``.
        saturation: ``1.0`` leaves saturation alone, ``0.0`` is greyscale, ``> 1``
            oversaturates.

    Returns:
        ``HWC uint8`` RGB.
    """
    f = _to_float(img)
    s = np.broadcast_to(np.asarray(slope, dtype=np.float32), (3,)).astype(np.float32)
    o = np.broadcast_to(np.asarray(offset, dtype=np.float32), (3,)).astype(np.float32)
    p = np.broadcast_to(np.asarray(power, dtype=np.float32), (3,)).astype(np.float32)

    # The clip before the power is part of the ASC-CDL definition, not defensive
    # programming: a negative intermediate raised to a fractional power is NaN.
    graded = np.clip(f * s + o, 0.0, 1.0) ** p
    graded = _luma(graded) + float(saturation) * (graded - _luma(graded))
    return _to_uint8(graded)


def random_colour_grade(
    img: np.ndarray,
    jitter: float = 0.10,
    power_range: tuple[float, float] = (0.8, 1.3),
    saturation_range: tuple[float, float] = (0.85, 1.15),
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample an ASC-CDL grade with per-channel jitter and apply it.

    Args:
        img: ``HWC uint8`` RGB.
        jitter: Half-width of the uniform slope and offset jitter, as a fraction.
            §4.2's schedule runs ``0.03`` to ``0.25``.
        power_range: Uniform range for the per-channel power term.
        saturation_range: Uniform range for the saturation blend.
        rng: Generator or seed. A fresh generator is created when ``None``.

    Returns:
        ``HWC uint8`` RGB.
    """
    g = _rng_of(rng)
    slope = 1.0 + g.uniform(-jitter, jitter, size=3)
    # Offset is a lift in [0,1] units, so the same fraction is a much stronger move
    # than it is on slope; §4.2 quotes one percentage for both, scaled down here.
    offset = g.uniform(-jitter, jitter, size=3) * 0.25
    power = g.uniform(power_range[0], power_range[1], size=3)
    sat = float(g.uniform(*saturation_range))
    return colour_grade(img, tuple(slope), tuple(offset), tuple(power), sat)


# --------------------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------------------


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian (defocus proxy) blur, ``research/05`` §4.2 sigma row.

    Args:
        img: ``HWC uint8`` RGB.
        sigma: Standard deviation in pixels. §4.2 spans ``0.5`` to ``4.0``. ``<= 0``
            returns a copy unchanged.

    Returns:
        ``HWC uint8`` RGB.
    """
    arr = _as_uint8_rgb(img)
    if sigma <= 0:
        return arr.copy()
    # ksize=(0,0) lets OpenCV derive the kernel from sigma, which is what §4.2's
    # `cv2.GaussianBlur(img,(0,0),sigmaX=sigma)` snippet does.
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def motion_blur(
    img: np.ndarray,
    length: float,
    angle: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Linear motion blur: a length-``L`` line kernel at angle ``theta``.

    Args:
        img: ``HWC uint8`` RGB.
        length: Kernel length in pixels. §4.2 spans ``3`` to ``35``. ``< 2`` is a
            no-op copy.
        angle: Degrees, measured from the +x axis. ``None`` draws uniformly from
            ``[0, 180)`` as §4.2 specifies — 180, not 360, because a line kernel at
            ``theta`` and ``theta + 180`` are the same kernel.
        rng: Generator or seed used only when ``angle`` is ``None``.

    Returns:
        ``HWC uint8`` RGB.
    """
    arr = _as_uint8_rgb(img)
    k = int(round(length))
    if k < 2:
        return arr.copy()
    if k % 2 == 0:  # an even kernel has no centre pixel and shifts the image by half a px
        k += 1
    theta = float(_rng_of(rng).uniform(0.0, 180.0)) if angle is None else float(angle)

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), theta, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k), flags=cv2.INTER_LINEAR)
    total = float(kernel.sum())
    if total <= 0:
        return arr.copy()
    kernel /= total
    return cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


# --------------------------------------------------------------------------------------
# Codec artifacts
# --------------------------------------------------------------------------------------


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode/decode round-trip, ``research/05`` §4.2 quality row.

    Args:
        img: ``HWC uint8`` RGB.
        quality: libjpeg quality, ``1``-``100``. §4.2 spans ``90`` down to ``10``.

    Returns:
        ``HWC uint8`` RGB.
    """
    arr = _as_uint8_rgb(img)
    q = int(np.clip(quality, 1, 100))
    # RGB->BGR->RGB is not cosmetic: JPEG's chroma subsampling is applied in YCbCr,
    # and cv2 derives Y assuming BGR input. Encoding RGB as if it were BGR swaps the
    # R and B luma weights, so the artifact pattern would not be the one a real
    # encoder produces.
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def has_ffmpeg() -> bool:
    """True when an ``ffmpeg`` binary is on ``PATH``. Used to skip the codec test."""
    import shutil

    return shutil.which("ffmpeg") is not None


def video_codec_roundtrip(
    img: np.ndarray,
    crf: int = 28,
    codec: str = "h264",
    fps: int = 10,
    n_frames: int = 10,
    frame_index: int = 5,
) -> np.ndarray:
    """Real H.264/HEVC artifacts via an ffmpeg round-trip (``research/05`` §4.3).

    **Precompute this offline. Never call it inside a training or evaluation loop.**
    ``research/05`` §4.3 (line 234) is explicit: "ffmpeg round-trips are slow —
    precompute the corrupted image set once, don't do it inside the training/eval
    loop." Each call spawns two subprocesses and writes three temporary files; at a
    few hundred milliseconds per frame it dominates a data-loading step by orders of
    magnitude and turns a GPU-bound run CPU-bound.

    A single still through JPEG does not reproduce block-motion-compensation or
    deblocking-filter artifacts, which is why the still is looped into a short clip
    and a *mid-sequence* frame is read back: a lone I-frame under-represents typical
    inter-frame codec behaviour.

    Args:
        img: ``HWC uint8`` RGB.
        crf: Constant rate factor. §4.2: ``{18,23,28,35,45}`` for H.264 and
            ``{20,25,30,37,47}`` for HEVC (roughly perceptually matched).
        codec: ``"h264"`` (libx264) or ``"hevc"`` (libx265).
        fps: Encoded frame rate.
        n_frames: How many copies of the still to encode.
        frame_index: Which decoded frame to read back, 0-based.

    Returns:
        ``HWC uint8`` RGB.

    Raises:
        RuntimeError: If ``ffmpeg`` is missing or either subprocess fails.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    if not has_ffmpeg():
        raise RuntimeError("ffmpeg is not on PATH; video_codec_roundtrip is unavailable")
    encoder = {"h264": "libx264", "hevc": "libx265", "h265": "libx265"}.get(codec.lower())
    if encoder is None:
        raise ValueError(f"codec must be 'h264' or 'hevc', got {codec!r}")

    arr = _as_uint8_rgb(img)
    with tempfile.TemporaryDirectory(prefix="cdlib_codec_") as tmp:
        d = Path(tmp)
        src, clip, out = d / "frame.png", d / "clip.mp4", d / "out.png"
        cv2.imwrite(str(src), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        duration = max(1, int(math.ceil(n_frames / max(fps, 1))))
        cmds = [
            # -pix_fmt yuv420p forces 4:2:0 chroma subsampling, which is where most of
            # the visible codec nuisance lives; leaving it out would let ffmpeg pick
            # yuv444p for libx264 and quietly produce a much cleaner image.
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(src),
                "-t",
                str(duration),
                "-r",
                str(fps),
                "-c:v",
                encoder,
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(clip),
            ],
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(clip),
                "-vf",
                f"select=eq(n\\,{int(frame_index)})",
                "-vframes",
                "1",
                "-y",
                str(out),
            ],
        ]
        for cmd in cmds:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
        decoded = cv2.imread(str(out), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("ffmpeg produced no readable frame")
    # yuv420p is subsampled on even dimensions, so ffmpeg may pad an odd-sized frame.
    decoded = decoded[: arr.shape[0], : arr.shape[1]]
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# Shadow and occlusion
# --------------------------------------------------------------------------------------


def add_shadow(
    img: np.ndarray,
    intensity: float = 0.35,
    area_fraction: float = 0.15,
    n_shadows: int = 1,
    n_vertices: int = 5,
    softness: float = 0.02,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Darken random convex polygons — synthetic cast shadows (``research/05`` §4.2).

    Implemented as a multiplicative darkening rather than a subtraction: a shadow is
    reduced illumination, so it scales the reflected signal. Subtracting a constant
    would crush the darks to a flat black patch, which looks like an occlusion and
    would leak into P5's occlusion stratum.

    Args:
        img: ``HWC uint8`` RGB.
        intensity: Darkening fraction inside the shadow, ``0``-``1``. §4.2 spans
            ``0.1`` to ``0.7``.
        area_fraction: Approximate fraction of the frame each shadow covers. §4.2
            grows the area with severity.
        n_shadows: How many polygons to draw. §4.2 uses ``1``-``3``.
        n_vertices: Vertices per polygon.
        softness: Penumbra width as a fraction of the frame's shorter side. ``0``
            gives a hard edge.
        rng: Generator or seed.

    Returns:
        ``HWC uint8`` RGB.
    """
    arr = _as_uint8_rgb(img)
    h, w = arr.shape[:2]
    g = _rng_of(rng)

    shadow = np.zeros((h, w), dtype=np.float32)
    # Radius of a regular n-gon whose area is `area_fraction` of the frame.
    target_area = max(1.0, float(area_fraction) * h * w)
    radius = math.sqrt(target_area / (0.5 * n_vertices * math.sin(2 * math.pi / n_vertices)))
    for _ in range(max(1, int(n_shadows))):
        cx, cy = g.uniform(0, w), g.uniform(0, h)
        phases = np.sort(g.uniform(0, 2 * math.pi, size=n_vertices))
        radii = radius * g.uniform(0.6, 1.4, size=n_vertices)
        pts = np.stack([cx + radii * np.cos(phases), cy + radii * np.sin(phases)], axis=1).astype(
            np.int32
        )
        cv2.fillPoly(shadow, [pts], 1.0)

    if softness > 0:
        k = max(1, int(round(softness * min(h, w))))
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=k, sigmaY=k)
        shadow = np.clip(shadow, 0.0, 1.0)

    f = _to_float(arr)
    return _to_uint8(f * (1.0 - float(intensity) * shadow[:, :, None]))


def occlude(
    img: np.ndarray,
    area_fraction: float = 0.10,
    n_holes: int = 1,
    fill: str | int | tuple[int, int, int] = "random",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Cutout-style occlusion (``research/05`` §4.2 occlusion row).

    Args:
        img: ``HWC uint8`` RGB.
        area_fraction: Total occluded fraction of the frame, split across holes.
            §4.2 spans ``0.02`` to ``0.35``.
        n_holes: Number of rectangular holes.
        fill: ``"random"`` fills each hole with uniform noise, ``"patch"`` pastes an
            unrelated crop of the same image (§4.2's "paste an unrelated crop"
            variant, the harder case because the occluder is scene-plausible), or a
            scalar / RGB tuple for a constant fill.
        rng: Generator or seed.

    Returns:
        ``HWC uint8`` RGB.
    """
    arr = _as_uint8_rgb(img).copy()
    h, w = arr.shape[:2]
    g = _rng_of(rng)
    n = max(1, int(n_holes))
    side = int(round(math.sqrt(max(0.0, float(area_fraction)) * h * w / n)))
    side_h, side_w = min(max(1, side), h), min(max(1, side), w)

    for _ in range(n):
        y = int(g.integers(0, max(1, h - side_h + 1)))
        x = int(g.integers(0, max(1, w - side_w + 1)))
        if fill == "random":
            arr[y : y + side_h, x : x + side_w] = g.integers(
                0, 256, size=(side_h, side_w, arr.shape[2]), dtype=np.int16
            ).astype(np.uint8)
        elif fill == "patch":
            sy = int(g.integers(0, max(1, h - side_h + 1)))
            sx = int(g.integers(0, max(1, w - side_w + 1)))
            arr[y : y + side_h, x : x + side_w] = arr[sy : sy + side_h, sx : sx + side_w]
        else:
            arr[y : y + side_h, x : x + side_w] = np.asarray(fill, dtype=np.uint8)
    return arr


# --------------------------------------------------------------------------------------
# Sensor noise
# --------------------------------------------------------------------------------------


def sensor_noise(
    img: np.ndarray,
    sigma: float = 0.12,
    kind: str = "gaussian",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Gaussian / shot / speckle sensor noise (``research/05`` §4.2, optional row).

    Args:
        img: ``HWC uint8`` RGB.
        sigma: Noise scale in ``[0,1]`` units. For ``"shot"`` it is converted to a
            photon count ``lambda = 1/sigma**2``, which makes the three kinds roughly
            comparable at equal ``sigma``.
        kind: ``"gaussian"`` (additive), ``"shot"`` (Poisson, signal-dependent) or
            ``"speckle"`` (multiplicative).
        rng: Generator or seed.

    Returns:
        ``HWC uint8`` RGB.
    """
    f = _to_float(img)
    g = _rng_of(rng)
    s = max(float(sigma), 1e-6)
    if kind == "gaussian":
        out = f + g.normal(0.0, s, size=f.shape)
    elif kind == "shot":
        lam = 1.0 / (s * s)
        out = g.poisson(f * lam) / lam
    elif kind == "speckle":
        out = f + f * g.normal(0.0, s, size=f.shape)
    else:
        raise ValueError(f"kind must be 'gaussian', 'shot' or 'speckle', got {kind!r}")
    return _to_uint8(np.clip(out, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Viewpoint jitter — the one corruption that moves the mask
# --------------------------------------------------------------------------------------


def _jitter_matrix(
    h: int,
    w: int,
    translate: float,
    rotate: float,
    scale: float,
    corner_jitter: float,
    g: np.random.Generator,
) -> np.ndarray:
    """Sample a 3x3 homography combining a small similarity and a corner perturbation."""
    cx, cy = w / 2.0, h / 2.0
    ang = float(g.uniform(-rotate, rotate))
    sc = 1.0 + float(g.uniform(-scale, scale))
    m = np.eye(3, dtype=np.float64)
    m[:2] = cv2.getRotationMatrix2D((cx, cy), ang, sc)
    m[0, 2] += float(g.uniform(-translate, translate)) * w
    m[1, 2] += float(g.uniform(-translate, translate)) * h

    if corner_jitter > 0:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(g.uniform(-corner_jitter, corner_jitter, size=(4, 2)) * [w, h])
        m = cv2.getPerspectiveTransform(src, dst).astype(np.float64) @ m
    return m


def viewpoint_jitter(
    img: np.ndarray,
    translate: float = 0.02,
    rotate: float = 1.0,
    scale: float = 0.0,
    corner_jitter: float = 0.01,
    rng: np.random.Generator | int | None = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
    return_valid: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Small random homography — the camera-displacement nuisance (§4.2 viewpoint row).

    Args:
        img: ``HWC uint8`` RGB.
        translate: Half-width of the uniform translation, as a fraction of W/H.
            §4.2 spans ``0.01`` to ``0.16``.
        rotate: Half-width of the uniform rotation in degrees. §4.2 spans ``0.5`` to
            ``8``.
        scale: Half-width of the uniform zoom factor. Not in §4.2's grid; defaults to
            off so the sampled distribution matches the brief unless asked otherwise.
        corner_jitter: Half-width of the four-corner perspective perturbation, as a
            fraction of W/H. §4.2 spans ``0.005`` to ``0.08``.
        rng: Generator or seed.
        border_mode: OpenCV border mode for the *image*. Reflection avoids a hard
            black edge the model could latch onto as a displacement cue.
        return_valid: Also return a ``HW bool`` map that is ``False`` wherever the
            output pixel was extrapolated from outside the source frame.

    Returns:
        ``HWC uint8`` RGB, or ``(image, valid)`` when ``return_valid``.
    """
    arr = _as_uint8_rgb(img)
    h, w = arr.shape[:2]
    m = _jitter_matrix(h, w, translate, rotate, scale, corner_jitter, _rng_of(rng))
    out = cv2.warpPerspective(arr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
    if not return_valid:
        return out
    ones = np.ones((h, w), dtype=np.uint8)
    valid = cv2.warpPerspective(
        ones, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return out, valid.astype(bool)


def viewpoint_jitter_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray,
    which: str = "img2",
    translate: float = 0.02,
    rotate: float = 1.0,
    scale: float = 0.0,
    corner_jitter: float = 0.01,
    warp_mask: bool = False,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displace one frame of a pair and repair the ground truth it invalidates.

    This is the only corruption that has to touch the mask, and it is the one most
    easily got wrong. Displacing ``img2`` alone scrolls unseen content in at the
    border: those output pixels have no counterpart in ``img1``, so *no annotation
    exists for them* and no model could be right about them. They must become
    :data:`~cdlib.data.contract.IGNORE_INDEX`. Leaving them at ``0`` would score the
    model on a region the ground truth never covered — the same class of error as
    filling geometric padding with ``0`` in
    :class:`cdlib.data.transforms.PairedGeometric`.

    Args:
        img1: Reference frame, ``HWC uint8`` RGB.
        img2: Edited frame, ``HWC uint8`` RGB.
        mask: ``HW`` mask with values in ``{0, 1, IGNORE_INDEX}``.
        which: ``"img1"`` or ``"img2"`` — the frame to displace. ``research/05`` §4.1
            asks for both directions to be run, since asymmetric sensitivity between
            them is itself a pair-order finding.
        translate: See :func:`viewpoint_jitter`.
        rotate: See :func:`viewpoint_jitter`.
        scale: See :func:`viewpoint_jitter`.
        corner_jitter: See :func:`viewpoint_jitter`.
        warp_mask: ``True`` moves the mask with the displaced frame, i.e. the
            prediction is expected in the *displaced* frame's coordinates. ``False``
            (default) keeps the mask in the undisplaced frame, which is the usual
            change-detection convention: ground truth stays in the reference view and
            the model is asked to undo the displacement.
        rng: Generator or seed.

    Returns:
        ``(img1, img2, mask)`` with the same dtypes and shapes as the inputs; the
        mask is ``int16``.
    """
    which = _resolve_which(which)

    a, b = _as_uint8_rgb(img1), _as_uint8_rgb(img2)
    h, w = a.shape[:2]
    m = _jitter_matrix(h, w, translate, rotate, scale, corner_jitter, _rng_of(rng))

    def _warp_img(x: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            x, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    if which == "img1":
        a = _warp_img(a)
    else:
        b = _warp_img(b)

    out_mask = np.asarray(mask)
    if out_mask.ndim == 3:
        out_mask = out_mask[..., 0]
    out_mask = out_mask.astype(np.int16, copy=True)
    if warp_mask:
        out_mask = cv2.warpPerspective(
            out_mask,
            m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(IGNORE_INDEX),
        ).astype(np.int16)

    # Pixels the warp pulled from outside the source frame: unannotatable either way.
    valid = cv2.warpPerspective(
        np.ones((h, w), dtype=np.uint8),
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    out_mask[~valid] = IGNORE_INDEX
    return a, b, out_mask


# --------------------------------------------------------------------------------------
# The two call patterns
# --------------------------------------------------------------------------------------


#: ``research/05`` §4.2 writes the one-frame eval signature as
#: ``apply_nuisance(img1, img2, corruption_fn, which="t2")`` with ``which in {"t1","t2"}``.
#: Our own naming follows the frozen contract's ``img1``/``img2``. Rather than make P5
#: discover the mismatch at runtime, both spellings are accepted.
_WHICH_ALIASES: dict[str, str] = {"t1": "img1", "t2": "img2", "img1": "img1", "img2": "img2"}


def _resolve_which(which: str) -> str:
    try:
        return _WHICH_ALIASES[which]
    except KeyError:
        raise ValueError(
            f"which must be one of {sorted(_WHICH_ALIASES)}, got {which!r} "
            "('t1'/'t2' are research/05's spelling of 'img1'/'img2')"
        ) from None


def apply_nuisance(
    img1: np.ndarray,
    img2: np.ndarray,
    corruption_fn: Callable[..., np.ndarray],
    which: str = "img2",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one corruption to exactly one frame — P5's evaluation call pattern.

    ``research/05`` §4.1, restated because it is the whole design of the benchmark:
    corrupting *both* frames identically lets a Siamese difference cancel the
    corruption as common-mode, so the measurement would report generic image-quality
    robustness rather than nuisance robustness. The failure mode we care about is a
    detector that fires because *one* camera pass had different auto-exposure,
    motion blur or compression than the other.

    P1's training path is deliberately different: it corrupts *both* frames with
    *independently sampled* parameters (:class:`cdlib.data.transforms.
    IndependentPhotometric`). Both are correct; they answer different questions. Do
    not unify them.

    Args:
        img1: Reference frame, ``HWC uint8`` RGB.
        img2: Edited frame, ``HWC uint8`` RGB.
        corruption_fn: Any single-image function in this module.
        which: The frame to corrupt. ``"img1"``/``"img2"``, or ``research/05``'s
            equivalent ``"t1"``/``"t2"`` — both spellings are accepted.
        **kwargs: Forwarded to ``corruption_fn``.

    Returns:
        ``(img1, img2)`` with exactly one frame corrupted.

    Raises:
        ValueError: If ``which`` is not one of the four accepted spellings.
    """
    which = _resolve_which(which)
    if which == "img2":
        return _as_uint8_rgb(img1), corruption_fn(img2, **kwargs)
    if which == "img1":
        return corruption_fn(img1, **kwargs), _as_uint8_rgb(img2)
    raise ValueError(f"which must be one of {sorted(_WHICH_ALIASES)}, got {which!r}")


#: Name -> single-image corruption, for callers that iterate the suite. Keys match
#: :data:`CORRUPTION_NUISANCE`. ``viewpoint`` is absent on purpose: it is the one
#: corruption that also rewrites the mask, so it has to go through
#: :func:`viewpoint_jitter_pair` and cannot be driven by :func:`apply_nuisance`.
CORRUPTION_FNS: dict[str, Callable[..., np.ndarray]] = {
    "gamma": adjust_gamma,
    "brightness_contrast": adjust_brightness_contrast,
    "colour_grade": random_colour_grade,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg": jpeg_compress,
    "shadow": add_shadow,
    "occlusion": occlude,
    "sensor_noise": sensor_noise,
    "video_codec": video_codec_roundtrip,
}


__all__ = [
    "CORRUPTION_FNS",
    "CORRUPTION_NUISANCE",
    "SEVERITY_REFERENCE",
    "add_shadow",
    "adjust_brightness_contrast",
    "adjust_gamma",
    "apply_nuisance",
    "colour_grade",
    "gaussian_blur",
    "has_ffmpeg",
    "jpeg_compress",
    "motion_blur",
    "occlude",
    "random_colour_grade",
    "sensor_noise",
    "video_codec_roundtrip",
    "viewpoint_jitter",
    "viewpoint_jitter_pair",
]
