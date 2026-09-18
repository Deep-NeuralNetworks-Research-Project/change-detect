from __future__ import annotations

import math

import pytest
import torch

from cdlib.cli._dummy import DummyPairCNN
from cdlib.metrics.efficiency import benchmark_model, count_params


def test_param_count_shared_siamese_not_doubled():
    m = DummyPairCNN()
    total, trainable = count_params(m)
    assert total == trainable
    assert total > 0
    # Encoder is applied to a concatenated batch, so parameters exist once.
    n_enc = sum(p.numel() for p in m.encoder.parameters())
    n_head = sum(p.numel() for p in m.head.parameters())
    assert total == n_enc + n_head


def test_benchmark_cpu_smoke():
    m = DummyPairCNN()
    r = benchmark_model(m, batch_size=1, img_size=32, warmup=1, iters=3, device="cpu")
    assert r.gpu == "cpu"
    assert r.params_total_M > 0
    assert r.latency_median_ms > 0
    assert r.latency_iqr_q25_ms <= r.latency_median_ms <= r.latency_iqr_q75_ms
    assert r.flop_tool.startswith("torch.utils.flop_counter")
    assert math.isnan(r.peak_mem_reserved_MB) or r.peak_mem_reserved_MB >= 0
    d = r.as_float_dict()
    assert "latency_median_ms" in d and "FLOPs_G" in d


def test_cli_benchmark_dummy(capsys):
    from cdlib.cli.benchmark import main

    rc = main(["--model", "dummy", "--img-size", "32", "--warmup", "1", "--iters", "2", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gpu=" in out
    assert "median_ms=" in out
    assert "dummy" in out


def test_cli_builds_siamese_resnet18():
    from cdlib.cli.benchmark import build_named_model

    m = build_named_model("siamese_resnet18")
    assert m is not None
    img = torch.rand(1, 3, 32, 32)
    m.eval()
    with torch.no_grad():
        out = m(img, img)
    assert out["logits"].shape[0] == 1


def test_cli_rejects_unknown_model():
    from cdlib.cli.benchmark import build_named_model

    with pytest.raises(FileNotFoundError, match="not registered"):
        build_named_model("not_a_real_model")


@pytest.mark.gpu
def test_benchmark_cuda_events():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    m = DummyPairCNN()
    r = benchmark_model(m, batch_size=1, img_size=32, warmup=5, iters=10, device="cuda")
    assert r.gpu != "cpu"
    assert r.peak_mem_reserved_MB >= r.peak_mem_allocated_MB or math.isnan(r.peak_mem_allocated_MB)
