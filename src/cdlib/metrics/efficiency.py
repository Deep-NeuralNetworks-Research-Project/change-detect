"""Efficiency measurement: params, pair-input FLOPs, peak memory, latency.

Primary FLOP tool: ``torch.utils.flop_counter.FlopCounterMode`` (true FLOPs,
mul+add counted separately). We also report MACs ≈ FLOPs/2. fvcore is an
optional cross-check — never the silent default.

Peak memory headline = ``max_memory_reserved`` (closer to nvidia-smi);
``max_memory_allocated`` is secondary. GPU name is logged every run.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

ForwardFn = Callable[[torch.Tensor, torch.Tensor], Any]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def pair_flops(model: nn.Module, t1: torch.Tensor, t2: torch.Tensor) -> int:
    """True FLOPs for one pair forward (both frames). Convention: mul+add."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        _call_model(model, t1, t2)
    return int(fc.get_total_flops())


def _call_model(model: nn.Module, t1: torch.Tensor, t2: torch.Tensor) -> Any:
    if hasattr(model, "forward_pair"):
        return model.forward_pair(t1, t2)
    try:
        return model(t1, t2)
    except TypeError:
        return model({"img1": t1, "img2": t2})


def _latency_ms(
    model: nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    warmup: int,
    iters: int,
    use_amp: bool,
) -> np.ndarray:
    device = t1.device
    use_cuda = device.type == "cuda"
    model.eval()
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and use_cuda)
        else nullcontext()
    )
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            _call_model(model, t1, t2)
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _call_model(model, t1, t2)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _call_model(model, t1, t2)
                times.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(times, dtype=np.float64)


@dataclass
class EfficiencyReport:
    params_total_M: float
    params_trainable_M: float
    FLOPs_G: float
    MACs_G: float
    flop_tool: str
    flop_convention: str
    peak_mem_reserved_MB: float
    peak_mem_allocated_MB: float
    latency_median_ms: float
    latency_iqr_q25_ms: float
    latency_iqr_q75_ms: float
    gpu: str
    cuda_version: str
    amp: bool
    tf32: bool
    batch_size: int
    img_size: int
    device: str

    def as_dict(self) -> dict[str, float | str | bool | int]:
        return asdict(self)

    def as_float_dict(self) -> dict[str, float]:
        """Metric-contract subset (floats only)."""
        return {
            "params_total_M": self.params_total_M,
            "params_trainable_M": self.params_trainable_M,
            "FLOPs_G": self.FLOPs_G,
            "MACs_G": self.MACs_G,
            "peak_mem_reserved_MB": self.peak_mem_reserved_MB,
            "peak_mem_allocated_MB": self.peak_mem_allocated_MB,
            "latency_median_ms": self.latency_median_ms,
            "latency_iqr_q25_ms": self.latency_iqr_q25_ms,
            "latency_iqr_q75_ms": self.latency_iqr_q75_ms,
            "amp": 1.0 if self.amp else 0.0,
            "tf32": 1.0 if self.tf32 else 0.0,
            "batch_size": float(self.batch_size),
            "img_size": float(self.img_size),
        }


def benchmark_model(
    model: nn.Module,
    *,
    batch_size: int = 1,
    img_size: int = 256,
    warmup: int = 30,
    iters: int = 100,
    use_amp: bool = False,
    device: str | None = None,
) -> EfficiencyReport:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # CI / smoke: fewer iters on CPU.
    if device == "cpu":
        warmup = min(warmup, 2)
        iters = min(iters, 5)
    model = model.to(device).eval()
    t1 = torch.rand(batch_size, 3, img_size, img_size, device=device)
    t2 = torch.rand(batch_size, 3, img_size, img_size, device=device)
    total, trainable = count_params(model)
    try:
        flops = pair_flops(model, t1, t2)
    except Exception:
        flops = 0
    macs = flops / 2.0  # FlopCounterMode counts mul+add; MACs ≈ FLOPs/2
    use_cuda = device == "cuda" and torch.cuda.is_available()
    peak_alloc = float("nan")
    peak_reserv = float("nan")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _call_model(model, t1, t2)
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserv = torch.cuda.max_memory_reserved(device) / 1024**2
    times = _latency_ms(model, t1, t2, warmup=warmup, iters=iters, use_amp=use_amp)
    q25, median, q75 = np.percentile(times, [25, 50, 75])
    gpu = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if use_cuda else False
    return EfficiencyReport(
        params_total_M=total / 1e6,
        params_trainable_M=trainable / 1e6,
        FLOPs_G=flops / 1e9,
        MACs_G=macs / 1e9,
        flop_tool="torch.utils.flop_counter.FlopCounterMode",
        flop_convention="true_FLOPs_mul_and_add; MACs=FLOPs/2",
        peak_mem_reserved_MB=float(peak_reserv),
        peak_mem_allocated_MB=float(peak_alloc),
        latency_median_ms=float(median),
        latency_iqr_q25_ms=float(q25),
        latency_iqr_q75_ms=float(q75),
        gpu=gpu,
        cuda_version=str(torch.version.cuda or ""),
        amp=use_amp,
        tf32=tf32,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
    )
