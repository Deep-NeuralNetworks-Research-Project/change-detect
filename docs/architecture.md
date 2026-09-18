# Architecture — Frozen Contracts

> **Locked in Week 1.** Changing a contract requires a **team-wide PR review**, not a quiet edit.
> These contracts are enforced by `tests/test_shapes.py` and `tests/test_registries.py`.

---

## 1. Dataset `__getitem__` Contract

Every dataset in `src/cdlib/data/datasets/*` must return exactly this:

```python
def __getitem__(self, idx: int) -> dict:
    return {
        "img1": Tensor[C, H, W],          # float32, range [0, 1]
        "img2": Tensor[C, H, W],          # float32, range [0, 1]
        "mask": Tensor[1, H, W],           # float32, values in {0, 1, -1}
                                           # -1 = ignore region
        "nuisance_label": Tensor[],        # int64
                                           # 0 = clean
                                           # 1..k = nuisance type
                                           # -1 = unknown
        "meta": {
            "source_video": str,
            "scene_id": str,
            "frame_idx": tuple[int, int],
            "pair_id": str,
            "dataset": str,
        },
    }
```

### Rules
- `img1` and `img2` are **always** float32 in `[0, 1]` — normalisation to ImageNet stats happens in the model or transform, not the dataset.
- `mask` uses `-1` for ignore regions. The loss must mask these out.
- `nuisance_label` is per-pair, not per-pixel.
- `meta["scene_id"]` is used for scene-disjoint splits — it must be unique per physical location. Splits must never leak `scene_id` across train/val/test.

---

## 2. Model `forward` Contract

Every baseline and the proposed model must implement:

```python
def forward(self, img1: Tensor, img2: Tensor) -> dict:
    return {
        "logits": Tensor[B, 1, H, W],                    # change-mask logits (pre-sigmoid)
        "confidence": Tensor[B, 1, H, W] | None,          # confidence-head output
        "aux": {                                           # experimental heads only ADD keys
            "directional_logits": Tensor[B, 2, H, W] | None,  # appeared/disappeared
            "alignment_offset": Tensor[B, 2, H, W] | None,
        },
    }
```

### Rules
- `logits` are **pre-sigmoid** — the loss applies the activation.
- `confidence` is `None` for baselines that don't have a confidence head.
- `aux` keys are **additive only** — new heads add new keys, never remove existing ones.
- Siamese models must concatenate `[I1; I2]` along the **batch** dim into one encoder forward pass for correct BatchNorm statistics, then split. Two sequential forwards is a known accuracy bug.

---

## 3. Loss Contract

```python
def compute(self, outputs: dict, batch: dict) -> dict:
    return {
        "loss": Tensor[],          # scalar, backpropagated by trainer
        "loss/bce": Tensor[],      # component losses for logging
        "loss/dice": Tensor[],
        "loss/pairorder": Tensor[],
        # Additional loss/* keys are allowed
    }
```

### Rules
- `trainer.py` backprops **only** `out["loss"]`.
- `trainer.py` auto-logs **every** `loss/*` key to W&B. It never inspects what's inside.
- If a loss component is not applicable (e.g., `loss/pairorder` for baselines), return `torch.tensor(0.0)`.
- The loss must handle `mask == -1` (ignore regions) — zero those pixels' contributions.

---

## 4. Metric Contract

Torchmetrics-style interface. Consider literally subclassing `torchmetrics.Metric` for free DDP-safety.

```python
class Metric:
    def reset(self) -> None:
        """Reset internal state (called at epoch start)."""
        ...

    def update(self, outputs: dict, batch: dict) -> None:
        """Accumulate predictions and targets for one batch."""
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric values from accumulated state."""
        ...
```

### Rules
- `update()` receives the same `outputs` dict from the model and `batch` dict from the dataloader.
- `compute()` returns a flat dict of `{metric_name: float}` values.
- Use **aggregate (corpus) F1**, not per-image averaged. Sum TP/FP/FN across the whole test set, then compute F1 once.

---

## 5. Registry Builders Contract

The **only four entrypoints** the CLI ever calls:

```python
def build_model(cfg) -> nn.Module
def build_dataset(cfg, split: str) -> Dataset
def build_loss(cfg) -> Loss
def build_optimizer(cfg, params) -> Optimizer
```

### Rules
- All builders live in `src/cdlib/models/build.py`.
- Builders resolve component names via registries — they never import a specific model/dataset/loss class directly.
- The CLI calls builders; it never instantiates components itself.
- Adding a new component means: (1) write a new file, (2) register a key, (3) add a config YAML. **Never edit build.py to add a component.**

---

## Registry Instances

| Registry | Location | Purpose |
|---|---|---|
| `MODEL_REGISTRY` | `models/_model_registry.py` | Complete CD models (forward contract) |
| `ENCODER_REGISTRY` | `models/encoders/registry.py` | Backbone encoders |
| `FUSION_REGISTRY` | `models/fusion/registry.py` | Feature fusion strategies |
| `ALIGNMENT_REGISTRY` | `models/alignment/registry.py` | Spatial alignment modules |
| `LOSS_REGISTRY` | `losses/registry.py` | Loss functions |
| `DATASET_REGISTRY` | `data/registry.py` | Dataset implementations |

---

## Non-Negotiable Rules (from CLAUDE.md)

1. **Aggregate (corpus) F1**, not per-image averaged.
2. **Scene/source-disjoint splits.** Never split at frame level.
3. **LEVIR-CD: 256×256 non-overlapping crops** → 7,120 / 1,024 / 2,048.
4. **Geometric aug shared; photometric aug independent per frame.**
5. **Siamese BN: batch-dim concat, one forward, split.**
6. **Checkpoint every 10–15 min wall-clock**, including RNG state.
7. **Notebooks: bootstrap + CLI call + plot. Nothing else.**
8. **≥3 seeds** for principal models.
9. **Never commit** video footage, `.env`, or W&B keys.

---

*Last updated: 2026-09-08. Any proposed change to these contracts requires a team-wide PR review.*
