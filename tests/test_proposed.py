"""Proposed-model contract: shapes, swap tensors, identity vs bounded alignment."""

from __future__ import annotations

import torch

from cdlib.models.proposed import ProposedModel

B, H, W = 2, 64, 64


def _tiny_proposed(**kwargs):
    defaults = dict(
        encoder={"name": "resnet18", "pretrained": False},
        fusion={"name": "signed_fusion", "mode": "signed"},
        alignment={"name": "identity"},
        decoder={"name": "unet_decoder"},
        heads={"confidence": True, "directional": True},
        compute_swap=True,
    )
    defaults.update(kwargs)
    return ProposedModel(**defaults)


def test_proposed_forward_contract_eval():
    model = _tiny_proposed(compute_swap=False)
    model.eval()
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    out = model(img1, img2)
    assert set(out) >= {"logits", "confidence", "aux"}
    assert out["logits"].shape == (B, 1, H, W)
    assert out["confidence"] is not None and out["confidence"].shape == (B, 1, H, W)
    assert out["aux"]["directional_logits"].shape[1] == 2
    assert out["aux"]["alignment_offset"].shape[1] == 2
    assert "logits_swapped" not in out


def test_proposed_emits_swap_in_train():
    model = _tiny_proposed()
    model.train()
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    out = model(img1, img2)
    assert out["logits_swapped"].shape == out["logits"].shape
    assert "directional_logits_swapped" in out["aux"]


def test_proposed_bounded_alignment_runs():
    model = _tiny_proposed(alignment={"name": "bounded", "max_disp": 2.0, "gated": True})
    model.eval()
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    with torch.no_grad():
        out = model(img1, img2)
    assert out["logits"].shape == (B, 1, H, W)
    assert torch.isfinite(out["logits"]).all()


def test_pairorder_loss_zero_without_swap():
    from cdlib.losses.pair_order_consistency import BCEDicePairOrderLoss

    model = _tiny_proposed(compute_swap=False)
    model.eval()
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, W // 2 :] = 1.0
    with torch.no_grad():
        out = model(img1, img2)
    loss = BCEDicePairOrderLoss().compute(out, {"mask": mask})
    assert torch.isfinite(loss["loss"])
    assert float(loss["loss/pairorder"]) == 0.0


def test_pairorder_loss_nonzero_with_signed_train_swap():
    from cdlib.losses.pair_order_consistency import BCEDicePairOrderLoss

    torch.manual_seed(0)
    model = _tiny_proposed()
    model.train()
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, W // 2 :] = 1.0
    out = model(img1, img2)
    loss = BCEDicePairOrderLoss(pairorder_weight=1.0).compute(out, {"mask": mask})
    assert torch.isfinite(loss["loss"])
    assert float(loss["loss/pairorder"].detach()) > 0.0
