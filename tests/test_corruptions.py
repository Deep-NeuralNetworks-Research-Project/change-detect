from __future__ import annotations

import numpy as np
import pytest

from cdlib.metrics.corruptions import (
    EVAL_CORRUPTIONS,
    apply_corruption,
    apply_to_pair,
)
from cdlib.metrics.robustness import retention, run_corruption_eval


def _img(seed=0, h=32, w=32):
    rng = np.random.default_rng(seed)
    return rng.random((3, h, w), dtype=np.float32)


def test_one_frame_only_t2():
    a, b = _img(1), _img(2)
    c1, c2 = apply_to_pair(a, b, "jpeg", 3, which="t2", rng=np.random.default_rng(0))
    assert np.allclose(c1, a)
    assert not np.allclose(c2, b)


def test_one_frame_only_t1():
    a, b = _img(1), _img(2)
    c1, c2 = apply_to_pair(a, b, "gaussian_blur", 2, which="t1", rng=np.random.default_rng(0))
    assert np.allclose(c2, b)
    assert not np.allclose(c1, a)


def test_both_frames_rejected():
    with pytest.raises(ValueError, match="t1"):
        apply_to_pair(_img(), _img(), "jpeg", 1, which="both")  # type: ignore[arg-type]


def test_eval_corruptions_stay_in_range():
    img = _img()
    rng = np.random.default_rng(0)
    for name in EVAL_CORRUPTIONS:
        out = apply_corruption(img, name, 3, rng=rng)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5


def test_severity_monotonic_jpeg_filesize_proxy():
    img = _img(h=48, w=48)
    outs = [apply_corruption(img, "jpeg", s, rng=np.random.default_rng(0)) for s in range(1, 6)]
    # Higher severity should move further from the original (not a strict
    # pixel-L2 guarantee for every seed, but q=90 vs q=10 is).
    d1 = np.mean(np.abs(outs[0] - img))
    d5 = np.mean(np.abs(outs[4] - img))
    assert d5 >= d1


def test_retention_formula():
    assert retention(0.4, 0.8) == pytest.approx(0.5)
    assert retention(0.1, 0.0) == 0.0


def test_run_corruption_eval_both_directions():
    img1, img2 = _img(1), _img(2)
    mask = np.zeros((1, 32, 32), dtype=np.float32)
    mask[0, 4:8, 4:8] = 1

    def predict(a, b):
        # Dummy: fire where |a-b| is large on luminance.
        d = np.abs(a.mean(0) - b.mean(0))
        logits = np.where(d > 0.05, 4.0, -4.0).astype(np.float32)
        return logits[None]

    rows = run_corruption_eval(
        predict, img1, img2, mask, names=("jpeg", "gamma"), severities=(1, 5), directions=("t1", "t2")
    )
    assert "clean_f1" in rows
    assert "jpeg/s1/t1/f1" in rows and "jpeg/s1/t2/f1" in rows
    assert "mean_retention" in rows


@pytest.mark.ffmpeg
def test_h264_roundtrip_changes_pixels(tmp_path):
    img = _img(h=64, w=64)
    out = apply_corruption(img, "h264", 5, cache_dir=tmp_path)
    assert out.shape == img.shape
    assert not np.allclose(out, img)
