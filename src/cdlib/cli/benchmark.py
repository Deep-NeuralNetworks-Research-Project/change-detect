"""Efficiency benchmark — P3 is blocked on this.

Warm-up excluded, CUDA events + synchronize, median + IQR, GPU logged
every run. Pair-input FLOPs via torch.utils.flop_counter. Peak memory
headline is max_memory_reserved.

This supersedes P3's scripts/benchmark_encoders.py disclosure fields
(params, pair GFLOPs, median ms, IQR, peak mem, gpu name).

Usage:
    python -m cdlib.cli.benchmark --model dummy --batch-size 1 --img-size 64
    python -m cdlib.cli.benchmark model=dummy   # Hydra-shaped leftover
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from cdlib.cli._dummy import DummyPairCNN
from cdlib.metrics.efficiency import benchmark_model

KNOWN_MODELS = ("dummy", "rgb_ssim", "fc_siam_diff", "siamese_resnet18", "proposed", "proposed_effnet")


def _parse_hydra_leftovers(unknown: list[str]) -> dict[str, str]:
    out = {}
    for tok in unknown:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.lstrip("+")] = v
    return out


def build_named_model(name: str) -> torch.nn.Module:
    name = name.strip()
    if name == "dummy":
        return DummyPairCNN()
    from cdlib.models._model_registry import MODEL_REGISTRY
    from cdlib.models.baselines import _import_all_baselines

    _import_all_baselines()
    from cdlib.models import proposed as _proposed  # noqa: F401

    kwargs: dict = {}
    if name == "siamese_resnet18":
        kwargs = {"pretrained": False}
    elif name == "fc_siam_diff":
        kwargs = {"encoder_weights": None}
    elif name in ("proposed", "proposed_effnet"):
        kwargs = {
            "encoder": {"name": "resnet18", "pretrained": False},
            "fusion": {"name": "signed_fusion", "mode": "signed"},
            "alignment": {"name": "identity"},
            "heads": {"confidence": True, "directional": True},
            "compute_swap": False,
        }
    if name not in MODEL_REGISTRY:
        raise FileNotFoundError(
            f"model={name!r} is not registered. Known names: {KNOWN_MODELS}. "
            f"Registry keys: {MODEL_REGISTRY.keys()}"
        )
    return MODEL_REGISTRY.build(name, **kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dummy", help="comma-separated model keys")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", action="store_true")
    args, unknown = p.parse_known_args(argv)
    extra = _parse_hydra_leftovers(unknown)
    model_spec = extra.get("model", args.model)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(
        f"device={args.device} gpu={gpu!r} cuda={torch.version.cuda} "
        f"amp={args.amp} tf32={bool(torch.backends.cuda.matmul.allow_tf32) if use_cuda else False} "
        f"batch={args.batch_size} img={args.img_size} "
        f"flop_tool=torch.utils.flop_counter convention=true_FLOPs_mul+add"
    )
    if not use_cuda:
        print("WARNING: not running on CUDA — smoke test, not a reportable latency number.")

    reports = []
    for name in model_spec.split(","):
        model = build_named_model(name)
        report = benchmark_model(
            model,
            batch_size=args.batch_size,
            img_size=args.img_size,
            warmup=args.warmup,
            iters=args.iters,
            use_amp=args.amp,
            device=args.device,
        )
        reports.append((name, report))
        print(
            f"{name:<20} params_M={report.params_total_M:.3f} "
            f"(trainable={report.params_trainable_M:.3f}) "
            f"FLOPs_G={report.FLOPs_G:.4f} MACs_G={report.MACs_G:.4f} "
            f"median_ms={report.latency_median_ms:.3f} "
            f"IQR=[{report.latency_iqr_q25_ms:.3f}, {report.latency_iqr_q75_ms:.3f}] "
            f"mem_reserved_MB={report.peak_mem_reserved_MB:.1f} "
            f"mem_alloc_MB={report.peak_mem_allocated_MB:.1f} gpu={report.gpu!r}"
        )

    if args.json:
        blob = {n: r.as_dict() for n, r in reports}
        json.dump(blob, sys.stdout, indent=2, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
