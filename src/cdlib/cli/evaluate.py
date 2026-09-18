"""Evaluate stub. Real Hydra driver is P2-owned; this runs metrics on dummy tensors.

    python -m cdlib.cli.evaluate +metrics=all
    python -m cdlib.cli.evaluate +robustness=full
    python -m cdlib.cli.evaluate exp_id=demo +metrics=segmentation,calibration
"""
from __future__ import annotations

import argparse
import json

import torch

from cdlib.cli._dummy import DummyPairCNN
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric
from cdlib.utils.reproducibility import set_seed

ALL_METRICS = ("segmentation", "boundary", "region", "calibration", "robustness")


def _parse_leftovers(unknown: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in unknown:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.lstrip("+")] = v
    return out


def dummy_batch(n: int = 4, h: int = 32, w: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(n, 1, h, w, generator=g) > 0.85).float()
    # sprinkle a true-negative pair
    mask[0].zero_()
    labels = torch.zeros(n, dtype=torch.int64)
    labels[1] = 1  # brightness-stratum dummy
    return {
        "img1": torch.rand(n, 3, h, w, generator=g),
        "img2": torch.rand(n, 3, h, w, generator=g),
        "mask": mask,
        "nuisance_label": labels,
        "meta": {"pair_id": "dummy", "dataset": "synthetic"},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=32)
    args, unknown = p.parse_known_args(argv)
    extra = _parse_leftovers(unknown)
    set_seed(args.seed)

    metrics_spec = extra.get("metrics", "all")
    robustness = extra.get("robustness")
    exp_id = extra.get("exp_id", "standalone")
    if robustness == "full":
        keys = list(ALL_METRICS)
    elif metrics_spec == "all":
        keys = list(ALL_METRICS)
    else:
        keys = [k.strip() for k in metrics_spec.split(",") if k.strip()]

    model = DummyPairCNN().eval()
    batch = dummy_batch(args.batch_size, args.img_size, args.img_size, seed=args.seed)
    with torch.no_grad():
        outputs = model(batch["img1"], batch["img2"])

    results = {"exp_id": exp_id, "metrics": {}}
    for key in keys:
        if key not in METRIC_REGISTRY:
            raise SystemExit(f"unknown metric {key!r}; available: {METRIC_REGISTRY.keys()}")
        m = build_metric(key)
        m.update(outputs, batch)
        results["metrics"][key] = m.compute()

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
