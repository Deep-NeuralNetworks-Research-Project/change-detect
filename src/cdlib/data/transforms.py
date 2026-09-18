"""Augmentation — shared geometry, independent photometry. Two code paths, on purpose.

Root ``CLAUDE.md`` rule 4: **geometric augmentation is shared across the pair;
photometric augmentation is independent per frame.** Both halves are load-bearing and
they fail in opposite directions:

* Sample the geometry *independently* per frame and you manufacture misregistration
  that is not in the data. P4's alignment module then spends its capacity learning to
  undo an augmentation bug, and the ablation that is supposed to show bounded
  alignment helps will show it helping for the wrong reason.
* Sample the photometry *jointly* and every nuisance you apply appears identically in
  both branches, where a Siamese difference cancels it as common-mode
  (``research/05`` §4.1). The model never sees the lighting/blur/codec mismatch that
  is the entire point of the project, and the robustness table measures nothing.

So there are two classes, each owning one ``A.Compose``, and
:class:`PairedTransform` chains them. The P1 role file is explicit: "Two different
code paths. Do not let a single ``Compose`` handle both."

Mask handling in the geometric path
-----------------------------------
Rotation, affine and perspective all pad. Those padded pixels are outside the
annotated frame, so their label is *unknown*, and the fill value must be
:data:`~cdlib.data.contract.IGNORE_INDEX`, never ``0``. Filling with ``0`` fabricates
"unchanged" ground truth: it inflates the true-negative count in every metric, it
feeds a confident background gradient into the loss over pixels nobody annotated, and
because the padded area scales with rotation angle it does so *non-uniformly across
the dataset*. In albumentations 2.0.8 this is the per-transform ``fill_mask``
argument, and it only takes effect under ``border_mode=cv2.BORDER_CONSTANT`` — see
:data:`GEOMETRIC_BORDER_MODE` for why the images therefore also get a constant border
rather than the reflected one that would otherwise be preferable.

Interpolation of the mask is pinned to ``cv2.INTER_NEAREST`` at the ``Compose``
level. Anything else averages neighbouring labels and produces values like ``0.5`` or
``-0.5``, which are outside the ``{0,1,-1}`` alphabet
:func:`cdlib.data.contract.validate_sample` enforces and which
:func:`cdlib.data.base.mask_to_tensor` would silently round into the wrong class.

RNG
---
Nothing here touches global RNG state. Albumentations 2.x gives each ``Compose`` its
own ``np.random.default_rng``; ``np.random.seed`` / ``random.seed`` have no effect on
it. Reproducibility comes from the ``seed`` argument and :meth:`PairedTransform.reseed`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import albumentations as A
import cv2
import numpy as np

from cdlib.data.contract import IGNORE_INDEX, SPLITS

#: Border mode for every padding geometric transform.
#:
#: It is ``BORDER_CONSTANT`` and not ``BORDER_REFLECT_101`` because albumentations
#: 2.0.8 exposes **one** ``border_mode`` per transform, shared by the image and mask
#: targets (verified with ``inspect.signature(A.Affine.__init__)``: ``border_mode``,
#: ``fill``, ``fill_mask`` — there is no ``border_mode_mask``). Choosing a reflected
#: border for the images would reflect the *mask* too, mirroring real change labels
#: into the padding. That is strictly worse than a black corner: a fabricated ``1`` is
#: worse than a fabricated ``0``, and both are worse than a correct ``-1``. So the
#: constant border wins and ``fill_mask=IGNORE_INDEX`` does its job.
#:
#: If the black corners ever prove to matter, the fix is to reflect-pad the images and
#: ignore-pad the mask *before* the warp and centre-crop after — per-target border
#: rules we own, rather than one the library shares.
GEOMETRIC_BORDER_MODE: int = cv2.BORDER_CONSTANT

#: Default augmentation policy per split, overridable through ``cfg["splits"]``.
#:
#: ``train`` gets both paths. ``val`` / ``test`` / ``cal`` get neither: augmenting at
#: evaluation time makes the number depend on the augmentation seed, and ``cal`` in
#: particular is the conformal split (``research/04`` §7) whose whole job is to be a
#: fixed, untouched reference.
#:
#: ``val_shift`` is the exception. It exists (``contract.SPLITS``) to be *more
#: nuisance-shift-heavy than training*, because temperature scaling fitted on clean
#: ``val`` under-corrects under shift (``research/04`` §4). Geometry stays off — a
#: shared flip is not a distribution shift, it is a relabelling of the same pair — but
#: photometry stays on, because independent per-frame photometry *is* the shift being
#: held out.
SPLIT_AUGMENTATION_POLICY: dict[str, dict[str, bool]] = {
    "train": {"geometric": True, "photometric": True},
    "val": {"geometric": False, "photometric": False},
    "test": {"geometric": False, "photometric": False},
    "val_shift": {"geometric": False, "photometric": True},
    "cal": {"geometric": False, "photometric": False},
}


# --------------------------------------------------------------------------------------
# Config access — plain dict or OmegaConf node, no hydra import
# --------------------------------------------------------------------------------------


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one key from a dict, a ``DictConfig`` or anything with attributes.

    ``build_transforms`` is called from ``build_dataset(cfg, split)`` with whatever
    Hydra produced, but the data layer must stay importable and testable without
    Hydra installed (``research/06`` §4 dry-constructs every config in CI), so this
    duck-types instead of importing ``omegaconf``.
    """
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        value = cfg.get(key, default)
    elif hasattr(cfg, "get"):  # omegaconf.DictConfig
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    """Coerce a scalar or 2-sequence into a ``(lo, hi)`` tuple."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (-float(value), float(value))
    lo, hi = (float(v) for v in value)
    return (lo, hi)


# --------------------------------------------------------------------------------------
# Path 1 — geometry, shared across the pair
# --------------------------------------------------------------------------------------


class PairedGeometric:
    """One ``A.Compose`` applied once to ``img1``, ``img2`` and ``mask`` together.

    ``additional_targets={"img2": "image"}`` makes albumentations route the second
    frame through the *same* sampled parameters as the first. The transform is
    sampled once per call, so the pair stays pixel-aligned and the mask stays aligned
    with both — which is the only reason the change labels remain meaningful after
    augmentation.

    Args:
        hflip: Probability of a horizontal flip.
        vflip: Probability of a vertical flip. Off by default: our frames are
            gravity-oriented video, and a vertical flip takes them off-distribution
            in a way a horizontal flip does not.
        rot90: Probability of a random 90-degree rotation. Free — no interpolation, no
            padding — but only meaningful for square, orientation-free imagery.
        affine_p: Probability of the affine transform.
        rotate: Rotation half-range in degrees.
        translate: Translation half-range as a fraction of W/H.
        scale: Zoom range as ``(lo, hi)`` multipliers.
        shear: Shear half-range in degrees.
        perspective_p: Probability of a perspective warp.
        perspective_scale: Corner displacement scale for the perspective warp.
        crop_size: ``(h, w)`` random resized crop, or ``None`` to leave the frame
            size alone. Loaders that already crop (LEVIR-CD's 256x256 non-overlapping
            grid, rule 3) must leave this ``None`` — a second random crop would
            silently break the fixed crop protocol.
        crop_scale: Area range for the random resized crop.
        image_fill: Constant border fill for the images. The mask's fill is always
            :data:`~cdlib.data.contract.IGNORE_INDEX` and is not configurable.
        seed: Seed for this pipeline's private RNG. ``None`` draws from entropy.
    """

    def __init__(
        self,
        *,
        hflip: float = 0.5,
        vflip: float = 0.0,
        rot90: float = 0.0,
        affine_p: float = 0.5,
        rotate: float = 10.0,
        translate: float = 0.05,
        scale: tuple[float, float] = (0.9, 1.1),
        shear: float = 0.0,
        perspective_p: float = 0.0,
        perspective_scale: float = 0.04,
        crop_size: tuple[int, int] | None = None,
        crop_scale: tuple[float, float] = (0.8, 1.0),
        image_fill: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.seed = seed
        self.image_fill = float(image_fill)
        self.mask_fill = float(IGNORE_INDEX)

        transforms: list[A.BasicTransform] = []
        if hflip > 0:
            transforms.append(A.HorizontalFlip(p=float(hflip)))
        if vflip > 0:
            transforms.append(A.VerticalFlip(p=float(vflip)))
        if rot90 > 0:
            transforms.append(A.RandomRotate90(p=float(rot90)))
        if crop_size is not None:
            transforms.append(
                A.RandomResizedCrop(
                    size=(int(crop_size[0]), int(crop_size[1])),
                    scale=(float(crop_scale[0]), float(crop_scale[1])),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                )
            )
        if affine_p > 0:
            transforms.append(
                A.Affine(
                    rotate=(-float(rotate), float(rotate)),
                    translate_percent=(-float(translate), float(translate)),
                    scale=(float(scale[0]), float(scale[1])),
                    shear=(-float(shear), float(shear)),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=GEOMETRIC_BORDER_MODE,
                    fill=self.image_fill,
                    # The one line this whole module exists to get right.
                    fill_mask=self.mask_fill,
                    p=float(affine_p),
                )
            )
        if perspective_p > 0:
            transforms.append(
                A.Perspective(
                    scale=(0.0, float(perspective_scale)),
                    keep_size=True,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=GEOMETRIC_BORDER_MODE,
                    fill=self.image_fill,
                    fill_mask=self.mask_fill,
                    p=float(perspective_p),
                )
            )

        self.transforms = transforms
        self._compose = A.Compose(
            transforms,
            # This is what makes the geometry shared rather than sampled twice.
            additional_targets={"img2": "image"},
            # Belt and braces over the per-transform setting: a nearest-neighbour mask
            # is the difference between {0,1,-1} and a continuum.
            mask_interpolation=cv2.INTER_NEAREST,
            seed=seed,
        )

    def __call__(
        self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply one sampled geometry to all three targets.

        Args:
            img1: ``HWC uint8`` RGB.
            img2: ``HWC uint8`` RGB.
            mask: ``HW`` with values in ``{0, 1, IGNORE_INDEX}``.

        Returns:
            ``(img1, img2, mask)``; the mask is ``int16``.
        """
        out = self._compose(image=img1, img2=img2, mask=_normalise_mask(mask))
        return out["image"], out["img2"], out["mask"].astype(np.int16)

    def reseed(self, seed: int | None) -> None:
        """Reset this pipeline's private RNG. Does not touch global RNG state."""
        self.seed = seed
        self._compose.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"PairedGeometric(seed={self.seed}, transforms={names})"


# --------------------------------------------------------------------------------------
# Path 2 — photometry, independent per frame
# --------------------------------------------------------------------------------------


class IndependentPhotometric:
    """A second ``A.Compose``, invoked once per frame with two independent RNGs.

    No ``additional_targets`` and no ``mask`` target: photometry must not be shared
    across the pair, and it never moves a pixel, so the mask is not involved at all.
    Two separately-seeded ``Compose`` objects rather than one called twice, so that
    "frame 1's draw" and "frame 2's draw" are reproducible independently of call
    order.

    The corruption families mirror ``research/05`` §4.2 so training nuisances and
    P5's evaluation nuisances are the same families. They are deliberately **not** the
    same *parameters*: §4.2's discrete severity grid belongs to P5, and training on it
    would make the retention table train-on-test. See
    :mod:`cdlib.data.corruptions` for the full statement of that constraint. The
    defaults below sit inside, and straddle, severities 1-3.

    Args:
        brightness_contrast_p: Probability of the brightness/contrast transform.
        brightness_limit: Additive brightness half-range in ``[0,1]`` units.
        contrast_limit: Contrast half-range.
        gamma_p: Probability of the gamma transform.
        gamma_limit: Gamma range in albumentations' percent units (``(70, 140)`` is
            ``gamma`` 0.7-1.4, §4.2's severity-2 row).
        colour_p: Probability of a hue/saturation/value shift, standing in for the
            colour-grading / white-balance-drift family.
        hue_limit: Hue shift half-range, OpenCV units.
        saturation_limit: Saturation shift half-range.
        blur_p: Probability of a Gaussian (defocus) blur.
        blur_sigma: ``(lo, hi)`` sigma in pixels.
        motion_blur_p: Probability of a motion blur.
        motion_blur_limit: ``(lo, hi)`` odd kernel lengths in pixels.
        jpeg_p: Probability of a JPEG round-trip.
        jpeg_quality: ``(lo, hi)`` JPEG quality.
        noise_p: Probability of additive Gaussian sensor noise.
        noise_std: ``(lo, hi)`` noise std in ``[0,1]`` units.
        shadow_p: Probability of a synthetic cast shadow.
        shadow_intensity: ``(lo, hi)`` shadow darkening fraction.
        seed: Seed for frame 1's pipeline; frame 2 gets a derived, distinct seed.
    """

    def __init__(
        self,
        *,
        brightness_contrast_p: float = 0.8,
        brightness_limit: float = 0.25,
        contrast_limit: float = 0.25,
        gamma_p: float = 0.5,
        gamma_limit: tuple[float, float] = (70, 140),
        colour_p: float = 0.3,
        hue_limit: float = 8.0,
        saturation_limit: float = 20.0,
        blur_p: float = 0.3,
        blur_sigma: tuple[float, float] = (0.3, 2.5),
        motion_blur_p: float = 0.2,
        motion_blur_limit: tuple[int, int] = (3, 17),
        jpeg_p: float = 0.3,
        jpeg_quality: tuple[int, int] = (35, 95),
        noise_p: float = 0.2,
        noise_std: tuple[float, float] = (0.02, 0.12),
        shadow_p: float = 0.15,
        shadow_intensity: tuple[float, float] = (0.15, 0.55),
        seed: int | None = None,
    ) -> None:
        self.seed = seed

        transforms: list[A.BasicTransform] = []
        if brightness_contrast_p > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=_pair(brightness_limit, (-0.25, 0.25)),
                    contrast_limit=_pair(contrast_limit, (-0.25, 0.25)),
                    p=float(brightness_contrast_p),
                )
            )
        if gamma_p > 0:
            transforms.append(
                A.RandomGamma(
                    gamma_limit=(float(gamma_limit[0]), float(gamma_limit[1])),
                    p=float(gamma_p),
                )
            )
        if colour_p > 0:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=_pair(hue_limit, (-8.0, 8.0)),
                    sat_shift_limit=_pair(saturation_limit, (-20.0, 20.0)),
                    val_shift_limit=(0, 0),  # value lives in the brightness transform
                    p=float(colour_p),
                )
            )
        if blur_p > 0:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=0,  # 0 => derive the kernel from sigma, as research/05 does
                    sigma_limit=(float(blur_sigma[0]), float(blur_sigma[1])),
                    p=float(blur_p),
                )
            )
        if motion_blur_p > 0:
            transforms.append(
                A.MotionBlur(
                    blur_limit=(int(motion_blur_limit[0]), int(motion_blur_limit[1])),
                    p=float(motion_blur_p),
                )
            )
        if jpeg_p > 0:
            transforms.append(
                A.ImageCompression(
                    compression_type="jpeg",
                    quality_range=(int(jpeg_quality[0]), int(jpeg_quality[1])),
                    p=float(jpeg_p),
                )
            )
        if noise_p > 0:
            transforms.append(
                A.GaussNoise(
                    std_range=(float(noise_std[0]), float(noise_std[1])),
                    p=float(noise_p),
                )
            )
        if shadow_p > 0:
            transforms.append(
                A.RandomShadow(
                    shadow_roi=(0.0, 0.0, 1.0, 1.0),
                    num_shadows_limit=(1, 2),
                    shadow_intensity_range=(
                        float(shadow_intensity[0]),
                        float(shadow_intensity[1]),
                    ),
                    p=float(shadow_p),
                )
            )

        self.transforms = transforms
        self._pipe1 = A.Compose(transforms, seed=seed)
        self._pipe2 = A.Compose(transforms, seed=_derive_seed(seed))

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corrupt each frame with its own independently sampled parameters.

        Args:
            img1: ``HWC uint8`` RGB.
            img2: ``HWC uint8`` RGB.

        Returns:
            ``(img1, img2)``, both ``HWC uint8``.
        """
        return self._pipe1(image=img1)["image"], self._pipe2(image=img2)["image"]

    def reseed(self, seed: int | None) -> None:
        """Reset both frames' private RNGs. Does not touch global RNG state."""
        self.seed = seed
        self._pipe1.set_random_seed(seed)
        self._pipe2.set_random_seed(_derive_seed(seed))

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"IndependentPhotometric(seed={self.seed}, transforms={names})"


# --------------------------------------------------------------------------------------
# The composed transform
# --------------------------------------------------------------------------------------


class PairedTransform:
    """Geometry first (shared), then photometry (independent).

    Conforms to :class:`cdlib.data.base.PairTransform`, so
    ``PairedChangeDataset.__getitem__`` can call it blind.

    The order matters. Geometry before photometry means the photometric corruption is
    applied to the frame the network will actually see, at its final resolution. The
    other order would blur/compress at the pre-crop scale and then resample the
    artifacts, which changes their spatial frequency and makes a "sigma 2 blur"
    training sample not comparable with P5's "sigma 2 blur" evaluation sample.

    Args:
        geometric: The shared-geometry path, or ``None`` to skip it.
        photometric: The independent-photometry path, or ``None`` to skip it.
        seed: When given, both sub-pipelines are reseeded from it so one seed
            reproduces the whole transform.
    """

    def __init__(
        self,
        geometric: PairedGeometric | None = None,
        photometric: IndependentPhotometric | None = None,
        seed: int | None = None,
    ) -> None:
        self.geometric = geometric
        self.photometric = photometric
        self.seed = seed
        if seed is not None:
            self.reseed(seed)

    def __call__(
        self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run both paths.

        Args:
            img1: ``HWC uint8`` RGB.
            img2: ``HWC uint8`` RGB.
            mask: ``HW`` with values in ``{0, 1, IGNORE_INDEX}``.

        Returns:
            ``(img1 HWC uint8, img2 HWC uint8, mask HW int16 in {0,1,-1})``.
        """
        out_mask = _normalise_mask(mask)
        if self.geometric is not None:
            img1, img2, out_mask = self.geometric(img1, img2, out_mask)
        if self.photometric is not None:
            img1, img2 = self.photometric(img1, img2)
        return img1, img2, out_mask

    def reseed(self, seed: int | None) -> None:
        """Reseed every sub-pipeline from one seed.

        Useful to P5 and to ``val_shift``: reseeding per sample index makes an
        augmented evaluation set reproducible pair-by-pair, independent of iteration
        order, worker count or epoch. Without it a seeded pipeline is only
        reproducible as a *sequence*.
        """
        self.seed = seed
        if self.geometric is not None:
            self.geometric.reseed(seed)
        if self.photometric is not None:
            self.photometric.reseed(_derive_seed(seed, salt=0x5EED))

    def __repr__(self) -> str:
        return (
            f"PairedTransform(seed={self.seed}, geometric={self.geometric!r}, "
            f"photometric={self.photometric!r})"
        )


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def build_transforms(cfg: Any, split: str) -> PairedTransform:
    """Build the transform for one split.

    Accepts either the full dataset config (with a ``transforms`` sub-node) or the
    ``transforms`` node itself, as a plain ``dict`` or an OmegaConf ``DictConfig``.
    Hydra is never imported here.

    Recognised keys (all optional)::

        transforms:
          seed: 0
          geometric:     {hflip: 0.5, rotate: 10.0, ...}   # PairedGeometric kwargs
          photometric:   {gamma_p: 0.5, jpeg_quality: [35, 95], ...}
          splits:
            val_shift: {geometric: false, photometric: true}

    ``splits`` overrides :data:`SPLIT_AUGMENTATION_POLICY` per split, which is how
    ``val_shift`` gets its photometric perturbation without that decision being
    hardcoded in a branch.

    Args:
        cfg: Config mapping, or ``None`` for defaults.
        split: One of :data:`cdlib.data.contract.SPLITS`.

    Returns:
        A configured :class:`PairedTransform`. Both paths may be ``None``, in which
        case the transform is a pass-through that only normalises the mask.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")

    node = _get(cfg, "transforms", cfg)
    policy = dict(SPLIT_AUGMENTATION_POLICY[split])
    policy.update({k: bool(v) for k, v in dict(_get(_get(node, "splits", {}), split, {})).items()})

    seed = _get(node, "seed", None)
    seed = None if seed is None else int(seed)

    geometric = None
    if policy.get("geometric", False):
        geometric = PairedGeometric(**_kwargs(_get(node, "geometric", {})), seed=seed)

    photometric = None
    if policy.get("photometric", False):
        photometric = IndependentPhotometric(**_kwargs(_get(node, "photometric", {})), seed=seed)

    return PairedTransform(geometric=geometric, photometric=photometric, seed=seed)


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _kwargs(node: Any) -> dict[str, Any]:
    """Turn a config sub-node into plain kwargs, converting OmegaConf lists to tuples."""
    if node is None:
        return {}
    items = node.items() if hasattr(node, "items") else dict(node).items()
    out: dict[str, Any] = {}
    for key, value in items:
        if key == "seed":  # owned by build_transforms, not by the sub-node
            continue
        out[str(key)] = tuple(value) if isinstance(value, (list, tuple)) else value
    return out


def _derive_seed(seed: int | None, salt: int = 0x9E3779B1) -> int | None:
    """A second, distinct seed from one seed. ``None`` stays ``None`` (full entropy).

    The two photometric pipelines must not share a stream — that is the whole point
    of the independent path — and must not be merely offset by one either, since
    consecutive seeds of a counter-based generator are a documented way to get
    correlated streams.
    """
    if seed is None:
        return None
    return int((int(seed) * 0x2545F491 + salt) % (2**32))


def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce any incoming mask to ``HW int16`` with values exactly in ``{0, 1, -1}``.

    Loaders hand over masks as ``uint8`` ``{0,255}``, ``bool``, or already-normalised
    ``int16``. Normalising on the way *in* is what lets the geometric path guarantee
    the output alphabet: with ``fill_mask=-1`` and nearest-neighbour interpolation,
    an input drawn from ``{0,1,-1}`` cannot leave it.
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    out = np.zeros(m.shape, dtype=np.int16)
    out[m > 0] = 1
    # Any negative value is an ignore marker; only -1 is contractual, but a loader
    # emitting -255 should not silently become "changed".
    out[m < 0] = IGNORE_INDEX
    return out


__all__ = [
    "GEOMETRIC_BORDER_MODE",
    "SPLIT_AUGMENTATION_POLICY",
    "IndependentPhotometric",
    "PairedGeometric",
    "PairedTransform",
    "build_transforms",
]
