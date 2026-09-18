"""Encoder speed benchmark — P3 Week 3 checklist item.

Measures step time for every encoder in ENCODER_REGISTRY, not just
FLOPs: research/01 and CLAUDE.md both flag that EfficientNet's FLOP
advantage does not reliably translate into wall-clock speed on T4/P100
(depthwise convs have low arithmetic intensity; SiLU is often not
kernel-fused). This script produces the number that finding is actually
based on, on whatever GPU it's run on — report the result to the team,
it decides the default encoder and the number itself goes in the paper.

Usage:
    python scripts/benchmark_encoders.py --batch-size 8 --img-size 256

Run this on the actual Colab/Kaggle GPU, not locally on CPU — the CPU
fallback below exists only so the script is runnable as a smoke test.

Coordinate with P5's `cdlib.cli.benchmark` once it exists (see
CLAUDE.local.md: "I depend on ... P5's benchmark.py") — this is a
standalone starter, not meant to fork into a second permanent harness.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from cdlib.models.encoders import ENCODER_REGISTRY


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _pair_flops(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor) -> int:
    """Pair-input FLOPs (both frames through the Siamese forward), via
    torch's built-in flop counter — no extra dependency (fvcore/thop)
    needed on torch>=2.2.
    """
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model.forward_pair(img1, img2)
    return fc.get_total_flops()


def _bench_one(
    key: str, batch_size: int, img_size: int, n_warmup: int, n_iters: int, device: str
) -> dict:
    model = ENCODER_REGISTRY.build(key, pretrained=False).to(device).eval()
    img1 = torch.rand(batch_size, 3, img_size, img_size, device=device)
    img2 = torch.rand(batch_size, 3, img_size, img_size, device=device)
    use_cuda = device == "cuda" and torch.cuda.is_available()

    gflops = _pair_flops(model, img1, img2) / 1e9

    with torch.no_grad():
        for _ in range(n_warmup):
            model.forward_pair(img1, img2)
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times_ms = []
        for _ in range(n_iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model.forward_pair(img1, img2)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model.forward_pair(img1, img2)
                times_ms.append((time.perf_counter() - t0) * 1000)

        peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else float("nan")

    times_ms.sort()
    n = len(times_ms)
    median = statistics.median(times_ms)
    q1 = times_ms[n // 4]
    q3 = times_ms[(3 * n) // 4]
    return {
        "key": key,
        "params_M": _param_count(model) / 1e6,
        "pair_gflops": gflops,
        "median_ms": median,
        "iqr_ms": (q1, q3),
        "peak_mem_MB": peak_mem_mb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"device={args.device} gpu={gpu_name!r} batch={args.batch_size} img={args.img_size}")
    if not use_cuda:
        print("WARNING: not running on CUDA — this is a smoke test, not the reportable measurement.")

    header = (
        f"{'encoder':<20}{'params(M)':>12}{'pair GFLOPs':>13}"
        f"{'median(ms)':>14}{'IQR(ms)':>22}{'peak_mem(MB)':>16}"
    )
    print(header)
    for key in ENCODER_REGISTRY.keys():
        r = _bench_one(key, args.batch_size, args.img_size, args.warmup, args.iters, args.device)
        iqr_str = f"[{r['iqr_ms'][0]:.2f}, {r['iqr_ms'][1]:.2f}]"
        print(
            f"{r['key']:<20}{r['params_M']:>12.2f}{r['pair_gflops']:>13.2f}"
            f"{r['median_ms']:>14.2f}{iqr_str:>22}{r['peak_mem_MB']:>16.1f}"
        )


if __name__ == "__main__":
    main()
