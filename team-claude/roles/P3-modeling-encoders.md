# My role: P3 — Modeling A (Encoders, Fusion, ResNet-18 Baseline)

> Copy this to `CLAUDE.local.md` at the repo root. It layers on top of the shared `CLAUDE.md`.

**I own:** `src/cdlib/models/encoders/**`, `src/cdlib/models/fusion/**`, `src/cdlib/models/decoders/**`, `src/cdlib/models/baselines/siamese_resnet18.py`
**CODEOWNERS:** `/src/cdlib/models/encoders/ /src/cdlib/models/fusion/ /src/cdlib/models/decoders/ /src/cdlib/models/baselines/siamese_resnet18.py @me`
**I build the components P4 composes into the proposed model.** My fusion module is where the project's cheapest novelty claim lives — treat `signed_fusion.py` as a contribution, not plumbing.

## What I must not break

The model `forward` contract. Encoders return a list of multi-scale features; fusion takes two feature lists and returns one; decoders take the fused list and return `[B,1,H,W]` logits. New components are **one file + one registry line** — I never edit `build.py` or `proposed.py`.

## Gotchas

**ResNet-18 taps.** `layer1`/C2 @ stride 4 (64ch), `layer2`/C3 @ 8 (128), `layer3`/C4 @ 16 (256), `layer4`/C5 @ 32 (512). Backbone 11.7M params.

**Stride-32 is too coarse for thin change boundaries.** Standard finding in the CD literature (Changer, SARAS-Net). Two options: drop `layer4` and decode from C2–C4, or make the last stage(s) stride-1/dilated. Costs FLOPs, recovers thin structures — exactly the small inserted/removed objects we care about. Ablate this; don't just pick one.

**Siamese BatchNorm** — concat `[I1;I2]` along the **batch** dim, one forward, then split. Not two sequential forwards.

**ImageNet normalisation is genuinely correct for us** (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) because we use natural video RGB, not satellite radiometry. Most CD papers have to caveat this and we don't — say so in the paper, it's a small free point.

**EfficientNet is probably slower than ResNet-18 in wall-clock despite ~4× fewer FLOPs.** Depthwise separable convs have low arithmetic intensity; SiLU often isn't kernel-fused. On T4/P100 this can invert the FLOP ranking entirely. **Benchmark step time before committing to an encoder** — and when I do, that measurement is a reportable result, not just an implementation detail.

**timm channel counts must be read at runtime**, not copied from a blog: `timm.create_model('efficientnet_b0', features_only=True, out_indices=(0,1,2,3,4))` then `model.feature_info.channels()` / `.reduction()`. B2's rounding (divisor-8 rule) makes published tables unreliable.

**Fusion taxonomy — this is the interesting part.** `|a−b|` is order-invariant but destroys direction (appeared and disappeared collapse). `concat` keeps direction at 2–3× decoder width. **Signed `a−b` + product is the middle ground: cheaper than concat, strictly more informative than abs-diff.** Citable precedent: FCCDN (arXiv:2105.10860) dual sum/difference branches. My `signed_fusion.py` must be configurable across all four variants because P4's ablation B6 sweeps them.

---

## Checklist

### Week 2 — encoders
- [ ] Read `research/01-baselines-architectures.md` §2, §3, §5
- [ ] `encoders/resnet.py` — ResNet-18, torchvision, ImageNet weights, C2–C5 taps, batch-dim-concat BN pattern
- [ ] Make the stride-32 handling a config flag (`use_c5: bool`, `dilate_last: bool`) so it's ablatable
- [ ] `encoders/efficientnet.py` — timm `features_only`, channels read at runtime, B0 and B2
- [ ] Register both in `ENCODER_REGISTRY`; `configs/model/` entries
- [ ] Shape tests pass for every encoder key

### Week 3 — the encoder speed measurement (needs P5's `benchmark.py`, due end of week 2)
- [ ] Benchmark ResNet-18 vs EfficientNet-B0 vs B2 **step time** on the actual Colab/Kaggle GPU — warm-up excluded, `torch.cuda.synchronize()`, CUDA events, median + IQR, log the GPU model (Colab's varies run to run)
- [ ] Record params, pair-input FLOPs, peak memory alongside (coordinate with P5's `benchmark.py` — don't build a second harness)
- [ ] **Report the finding to the team**: if EfficientNet is slower, we default to ResNet-18 and the measurement goes in the paper

### Week 3 — fusion (my main contribution)
- [ ] `fusion/absdiff.py` — `|a−b|`, the FC-Siam-Diff baseline behaviour
- [ ] `fusion/signed_fusion.py` — configurable: `signed` (`a−b`), `signed_product` (`a−b` ⊕ `a·b`), `concat`, `signed_concat`. All four selectable by config, because B6 sweeps them
- [ ] Register in `FUSION_REGISTRY`; shape tests over every key
- [ ] Write the docstring explaining *why* signed preserves direction — P4 and P5 will both cite this reasoning

### Week 3–4 — decoder and ResNet-18 baseline
- [ ] `decoders/unet_decoder.py` — U-Net/FPN style over the 4 skip levels
- [ ] `models/baselines/siamese_resnet18.py` composing encoder + abs-diff fusion + decoder
- [ ] **Tune it properly.** LR sweep, same budget the proposed model gets. Per the Reality Check paper a well-tuned ResNet-18 baseline may rival BIT — if it does, that's a legitimate and publishable result, not a failure
- [ ] ≥3 seeds; log to the shared benchmark table

### Weeks 5+ — support the ablations
- [ ] Make sure every fusion variant is swappable by a one-line config override for P4's B6
- [ ] Help P4 wire encoder + fusion into `proposed.py`
- [ ] Stride-32 ablation (with/without C5, dilated vs not) — cheap, and it speaks directly to thin-boundary quality that P5 measures with Boundary IoU
- [ ] Draft the Experiments/ablations paper section (my writing assignment)

## Commands I'll use

```bash
python -c "import timm; m=timm.create_model('efficientnet_b0',features_only=True); print(m.feature_info.channels(), m.feature_info.reduction())"
pytest tests/test_shapes.py -k encoder -v
python -m cdlib.cli.benchmark model=siamese_resnet18,proposed_effnet   # step time, params, FLOPs, memory
python -m cdlib.cli.train -m model=siamese_resnet18 train.lr=1e-3,3e-4,1e-4 seed=0,1,2
python -m cdlib.cli.train -m model.fusion=absdiff,signed,signed_product,concat   # B6 sweep
```

## Hand-offs

| To | What | When |
|---|---|---|
| P4 | encoders + fusion registered and shape-tested | end of week 3 |
| team | encoder speed benchmark → which encoder we default to | week 3 |
| P5 | ResNet-18 baseline numbers, ≥3 seeds | week 4 |
| P4 | all four fusion variants config-swappable for B6 | week 5 |

**I depend on:** P2's registries/builders (week 1), P1's loaders (week 2), **P5's `benchmark.py` (end of week 2)** for the encoder speed measurement.
