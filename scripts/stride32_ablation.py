"""Stride-32 ablation — the cheap half.

Weeks 5+ checklist item: "Stride-32 ablation (with/without C5, dilated
vs not)". The half that needs trained weights and real data (does
dropping/dilating C5 actually improve thin-boundary quality, measured
via P5's Boundary IoU) is blocked on P1's loaders + P2's trainer + a GPU
session. This script covers the half that doesn't need any of that: for
each of the three `ResNet18Encoder` configurations, report param count,
pair-input GFLOPs, and the decoder's output resolution relative to input
— i.e. exactly how much it costs and how much resolution it buys, before
spending a single training step on the question.

Usage:
    python scripts/stride32_ablation.py --img-size 256
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

from cdlib.models.baselines.siamese_resnet18 import SiameseResNet18

_CONFIGS = [
    ("use_c5=True,  dilate_last=False (default, stride 32)", dict(use_c5=True, dilate_last=False)),
    ("use_c5=False, dilate_last=False (drop C5, decode to stride 16)", dict(use_c5=False, dilate_last=False)),
    ("use_c5=True,  dilate_last=True  (atrous C5, stride 16)", dict(use_c5=True, dilate_last=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    B, S = args.batch_size, args.img_size
    img1 = torch.rand(B, 3, S, S)
    img2 = torch.rand(B, 3, S, S)

    print(f"input: {B}x3x{S}x{S} (pair)\n")
    header = f"{'config':<62}{'params(M)':>11}{'pair GFLOPs':>13}{'C5 stride':>11}"
    print(header)
    for label, kwargs in _CONFIGS:
        model = SiameseResNet18(pretrained=False, **kwargs).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(img1, img2)
        gflops = fc.get_total_flops() / 1e9

        c5_stride = model.encoder.out_strides[-1]
        print(f"{label:<62}{params_m:>11.2f}{gflops:>13.2f}{c5_stride:>11}")

    print(
        "\nNext step once P1/P2 land: train each config for >=3 seeds on the "
        "same budget and compare Boundary IoU / thin-structure recall (P5's "
        "metric), not just cost — that's the actual ablation result."
    )


if __name__ == "__main__":
    main()
