"""Overfit-one-batch sanity check for the P3 baseline.

Per research/06's test strategy: on a tiny fixed synthetic batch, loss
must drop sharply within ~50-100 steps. Catches broken gradients or
shape mismatches in seconds on CPU, well before any real data pipeline
(P1) or trainer (P2) exists to run this against real data.

The target is a deterministic vertical-split pattern, not pixel-random
noise. Tried noise first (`mask = torch.rand(...) > 0.5`) and it
plateaus around BCE's ln(2) with gradients decaying to ~0 by step
~100 — not a broken-gradient bug (checked: gradients are healthy and
non-zero through encoder/fusion/decoder in early steps), just that a
target spatially independent of the input and of every other pixel
gives BCE no learnable signal to climb, so the optimiser gets stuck in
a "predict the marginal frequency" local minimum. A deterministic
spatial pattern is what this test is actually meant to catch shape/
gradient bugs against.
"""
from __future__ import annotations

import torch

from cdlib.models.baselines.siamese_resnet18 import SiameseResNet18

B, H, W = 2, 64, 64
N_STEPS = 80


def test_siamese_resnet18_overfits_one_batch():
    torch.manual_seed(0)
    model = SiameseResNet18(pretrained=False)
    model.train()

    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, W // 2 :] = 1.0  # deterministic vertical-split target

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(N_STEPS):
        opt.zero_grad()
        out = model(img1, img2)
        loss = loss_fn(out["logits"], mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop enough in {N_STEPS} steps: "
        f"start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
