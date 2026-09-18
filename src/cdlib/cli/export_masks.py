"""Export colour-blind-safe overlays for best/median/worst pairs.

    python -m cdlib.cli.export_masks exp_id=demo --overlays best,median,worst
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cdlib.cli._dummy import DummyPairCNN
from cdlib.metrics._tensor import as_numpy
from cdlib.metrics.overlays import overlay_tp_fp_fn, pick_best_median_worst
from cdlib.metrics.segmentation import confusion_counts, prf1_from_counts
from cdlib.utils.reproducibility import set_seed


def _parse_leftovers(unknown: list[str]) -> dict[str, str]:
    out = {}
    for tok in unknown:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.lstrip("+")] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlays", default="best,median,worst")
    p.add_argument("--out-dir", default="results/overlays")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args, unknown = p.parse_known_args(argv)
    extra = _parse_leftovers(unknown)
    exp_id = extra.get("exp_id", "standalone")
    set_seed(args.seed)

    model = DummyPairCNN().eval()
    g = torch.Generator().manual_seed(args.seed)
    n, h = args.n, args.img_size
    img1 = torch.rand(n, 3, h, h, generator=g)
    img2 = torch.rand(n, 3, h, h, generator=g)
    mask = (torch.rand(n, 1, h, h, generator=g) > 0.85).float()
    with torch.no_grad():
        logits = model(img1, img2)["logits"]
    pred = (logits.sigmoid() >= 0.5).cpu().numpy()[:, 0]
    gt = mask.cpu().numpy()[:, 0] > 0.5
    scores = []
    for i in range(n):
        tp, fp, fn, _ = confusion_counts(pred[i], gt[i], np.ones_like(gt[i], dtype=bool))
        _, _, f1, _ = prf1_from_counts(tp, fp, fn)
        scores.append(f1)
    idx = pick_best_median_worst(np.asarray(scores))
    wanted = [s.strip() for s in args.overlays.split(",") if s.strip()]
    out_dir = Path(args.out_dir) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        i = idx[name]
        overlay = overlay_tp_fp_fn(as_numpy(img2[i]), pred[i], gt[i])
        path = out_dir / f"{name}_{i}.npy"
        np.save(path, overlay)
        print(f"wrote {path}  pair_f1={scores[i]:.4f}")
    print(f"exp_id={exp_id} indices={idx}")
    print("Note: .npy overlays; convert to PDF in the paper pipeline (vector figures).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
