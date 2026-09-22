# Architecture diagrams

Diagrams of `cdlib` as the code is wired today. Frozen tensor contracts stay in [`architecture.md`](architecture.md). The paper figure `paper/figures/proposed_architecture.pdf` is the publication drawing of the proposed net; this file is the codebase map.

Root Hydra defaults are `data=sysu_cd`, `model=fc_siam_diff`, `loss=bce_dice`. The proposed net is opt-in (`model=proposed_effnet`, `loss=bce_dice_pairorder`).

---

## 1. How a run starts

Three launchers. None of them contain model, loss, or data logic. The GPU process is always `python -m cdlib.cli.train`.

```mermaid
flowchart LR
  subgraph launch [Launchers]
    local["python -m cdlib.cli.train"]
    nb["notebooks/colab and kaggle drivers"]
    modal["modal run train.py or modal_app.py"]
  end

  local --> hydra
  nb --> hydra
  modal --> vol
  vol["Volumes cdlib-data and cdlib-results"] --> hydra
  hydra["Hydra configs/config.yaml"] --> cli["cdlib.cli.train"]
  cli --> builders["build_model / build_dataset / build_loss / build_optimizer"]
  builders --> trainer["engine.trainer.Trainer"]
  trainer --> results["results/experiment_name/"]
  trainer --> wandb["W and B, optional"]
```

`train.py` at the repo root is only the short Modal command. It re-exports `modal_app` and, under `python train.py`, execs `modal run`. Checkpoints land on the `cdlib-results` volume at `/opt/cdlib/results` when Modal launches the job, and under `./results/<experiment_name>/` locally.

---

## 2. Package map

```mermaid
flowchart TB
  subgraph cli [cli]
    train["train.py Hydra"]
    evaluate["evaluate.py stub"]
    exportm["export_masks.py stub"]
    bench["benchmark.py"]
    valid["validate_configs.py"]
    mrun["modal_runtime.py job catalog"]
  end

  subgraph build [The only four builders]
    bm["build_model"]
    bd["build_dataset"]
    bl["build_loss"]
    bo["build_optimizer"]
  end

  subgraph models [models]
    regM["MODEL_REGISTRY"]
    enc["encoders"]
    aln["alignment"]
    fus["fusion"]
    dec["decoders"]
    heads["heads"]
    prop["proposed.py"]
    base["baselines"]
  end

  subgraph rest [Rest of the library]
    data["data datasets, splits, transforms"]
    loss["losses"]
    eng["engine.trainer"]
    met["metrics"]
    util["utils registry, checkpoint, seed, capture"]
  end

  train --> bm
  train --> bd
  train --> bl
  train --> bo
  bm --> regM
  regM --> prop
  regM --> base
  prop --> enc
  prop --> aln
  prop --> fus
  prop --> dec
  prop --> heads
  bd --> data
  bl --> loss
  train --> eng
  evaluate --> met
  bench --> regM
  bench --> met
  mrun --> train
```

Adding a component is one new file plus one registry line plus one YAML. `build.py` is not edited to add a model, dataset, or loss.

---

## 3. Hydra composition

```mermaid
flowchart TB
  root["configs/config.yaml"]
  root --> dataG["data: sysu_cd | levir_cd | pcd | video_pairs | videosham | dvi"]
  root --> modelG["model: fc_siam_diff | siamese_resnet18 | rgb_ssim | proposed_effnet"]
  root --> lossG["loss: bce_dice | bce_dice_pairorder"]
  root --> trainG["train: default"]
  root --> exp["optional +experiment= ablation or baseline override"]

  modelG --> encG["model/encoder resnet18, efficientnet_b0, efficientnet_b2"]
  modelG --> fusG["model/fusion signed, signed_product, concat, signed_concat, absdiff"]

  dataG --> rootPath["root = data.root, else data_root/name, else data/name"]
```

`configs/model/proposed_effnet.yaml` composes encoder `efficientnet_b0` (ImageNet), fusion `signed_fusion` (mode defaults to `signed_product` because the YAML does not set one), alignment `bounded` with `max_offset: 4`, decoder `unet_decoder`, and both heads on.

Named Modal jobs live in `cdlib.cli.modal_runtime.JOBS`. Suites are `baselines`, `proposed`, `ablations`, and `all`.

---

## 4. Training step

`cdlib.cli.train` does not pass `metric_fns`, so the validation pass inside the trainer is loss-only. Corpus F1 and the other metrics run from `cdlib.cli.evaluate` and `cdlib.cli.benchmark`, not from this loop.

```mermaid
sequenceDiagram
  participant H as Hydra
  participant C as cli.train
  participant B as builders
  participant T as Trainer
  participant M as model
  participant L as loss
  participant K as CheckpointSaver

  H->>C: composed cfg
  C->>B: build_model, build_loss, build_optimizer, build_dataset
  B-->>C: module, loss, AdamW, train and val loaders
  C->>T: Trainer(cfg, ...)
  T->>T: set_seed, capture metadata, resume_from=auto
  loop each epoch
    loop each batch
      T->>M: forward(img1, img2)
      M-->>T: logits, confidence, aux
      T->>L: compute(outputs, batch)
      L-->>T: loss and loss/* keys
      T->>T: backward on out loss only, AdamW step
      T->>K: maybe_save every 15 min wall clock
    end
    T->>T: cosine scheduler step
    T->>M: eval forward on val
    T->>L: val loss
    T->>K: save best on primary metric, else minus val loss
  end
  T->>K: final checkpoint
```

One step is contract-blind. The trainer backprops `out["loss"]` and logs every `loss/*` key. It does not inspect BCE, Dice, or pair-order internals.

AMP is on when the device is CUDA. The default recipe is 200 epochs, batch 8, AdamW `3e-4`, weight decay `0.01`, cosine to `1e-6`.

A checkpoint stores model, optimizer, scheduler, and RNG state. `capture_run_metadata` writes `config.yaml`, `commit.txt`, `command.txt`, and `env.txt` into the run directory. The saver keeps the last 2 checkpoints plus 1 best.

---

## 5. Data pipeline

```mermaid
flowchart TB
  cfg["cfg.data.name + split"] --> resolve["resolve root"]
  resolve --> cls["DATASET_REGISTRY"]
  cls --> ds["PairedChangeDataset"]
  policy["SPLIT_AUGMENTATION_POLICY"] --> tf["build_transforms"]
  tf --> ds

  ds --> item["__getitem__"]
  item --> img1["img1 float32 C,H,W in 0 to 1"]
  item --> img2["img2 float32 C,H,W in 0 to 1"]
  item --> mask["mask 1,H,W in 0, 1, or -1"]
  item --> nui["nuisance_label int64"]
  item --> meta["meta source_video, scene_id, frame_idx, pair_id, dataset"]
```

Registered datasets:

| Key | Role |
|---|---|
| `levir_cd` | 256 non-overlapping crops, 7120 / 1024 / 2048 |
| `sysu_cd` | Official 12000 / 4000 / 4000 of 256 pairs. Root default. |
| `pcd` | PCD / TSUNAMI. GSV is off the critical path. |
| `video_pairs` | In-house pairs. Scene split still falls back to a hash partition. |
| `videosham` | Fallback video set. Registered so the switch is a config change. |
| `dvi` | Registered dataset. |

Splits accepted by `build_dataset`: `train`, `val`, `test`, `val_shift`, `cal`. A missing root yields an empty dataset so CI can construct every config with no media on disk. `scene_id` must not cross a split.

Augmentation policy:

| Split | Geometric, shared across the pair | Photometric, independent per frame |
|---|---|---|
| `train` | on | on |
| `val`, `test`, `cal` | off | off |
| `val_shift` | off | on |

`nuisance_label`: `-1` unknown, `0` clean, then `1` camera displacement, `2` lighting, `3` colour grading, `4` blur, `5` shadow, `6` occlusion, `7` codec, `8` sensor noise, `9` mixed. Public CD sets emit unknown. Ignore pixels (`mask == -1`) are dropped from loss and from counts.

ImageNet mean and std constants live on the ResNet encoder module. The dataset contract and `transforms.py` leave pixels in `[0, 1]`. Nothing in the data path applies those constants.

---

## 6. Proposed model

Registered twice, as `proposed` and `proposed_effnet`, both `ProposedModel`. `compute_swap` defaults to true. In train mode the module runs a second forward on `(img2, img1)` and stores `logits_swapped` plus `aux.directional_logits_swapped`. Eval mode does not.

```mermaid
flowchart TB
  i1["I1 reference B,3,H,W"] --> cat
  i2["I2 edited B,3,H,W"] --> cat
  cat["batch-concat to 2B, one encoder forward, split"] --> f1["feats1 shallow to deep"]
  cat --> f2["feats2"]

  f1 --> align["BoundedAlignment per scale"]
  f2 --> align
  align --> f1u["feats1 unchanged"]
  align --> f2u["feats2 warped and gated"]
  align --> off["alignment_offset from the deepest scale"]

  f1u --> fuse["SignedFusion per scale"]
  f2u --> fuse
  fuse --> skips["fused list, same order"]

  skips --> dec["UNetDecoder coarsest to finest"]
  dec --> logits0["logits at stride-4 skip"]
  logits0 --> up["bilinear to H,W"]
  up --> logits["logits B,1,H,W pre-sigmoid"]

  logits --> confH["ConfidenceHead 1x1"]
  confH --> conf["confidence B,1,H,W"]

  skips --> dirIn["fused deepest map"]
  dirIn --> dirH["DirectionalHead 1x1 to 2 channels"]
  dirH --> dir["directional_logits B,2,H,W"]

  logits --> out["forward dict"]
  conf --> out
  dir --> out
  off --> out
```

Return value, matching the frozen contract:

```text
{
  logits:              [B, 1, H, W]   pre-sigmoid
  confidence:          [B, 1, H, W] | None
  aux.directional_logits: [B, 2, H, W] | None
  aux.alignment_offset:   [B, 2, h, w]   deepest scale, not resized to the image
  logits_swapped:      train and compute_swap only
  aux.directional_logits_swapped: same
}
```

### Encoder taps

Both encoders implement `forward_pair`: concatenate `[I1; I2]` on the batch axis, one forward, then split. Two sequential forwards are not used.

| Encoder | Key | Taps | Typical channels | Strides |
|---|---|---|---|---|
| ResNet-18 | `resnet18` | C2–C5, `layer1`–`layer4` | 64 / 128 / 256 / 512 | 4 / 8 / 16 / 32 |
| EfficientNet-B0 | `efficientnet_b0` | timm `features_only`, `out_indices (1,2,3,4)` | read at runtime, usually 24 / 40 / 112 / 320 | 4 / 8 / 16 / 32 |
| EfficientNet-B2 | `efficientnet_b2` | same indices | runtime `feature_info` | same four strides |

`use_c5: false` drops the last tap. ResNet `dilate_last: true` keeps four taps and emits the last one at stride 16. `ProposedModel` falls back to ResNet-18 if the encoder spec is missing. The proposed YAML asks for EfficientNet-B0 with ImageNet weights. Nothing in the encoder is frozen.

### Bounded alignment, one scale

Applied to `feats2` only. The last conv of the offset head is zero-initialised, so the module starts as identity. `max_offset` in the YAML is `max_disp` in the module (default 4 pixels).

```mermaid
flowchart LR
  f1["f1"] --> cat2["concat f1, f2"]
  f2["f2"] --> cat2
  cat2 --> head["Conv 3x3, ReLU, Conv 3x3 to 2"]
  head --> tanh["tanh times max_disp"]
  tanh --> warp["grid_sample bilinear, zeros pad, align_corners true"]
  f2 --> warp
  f1 --> ad["abs f1 minus f2"]
  f2 --> ad
  ad --> gate["1x1 then sigmoid"]
  warp --> mix["g * warped + (1-g) * f2"]
  gate --> mix
  f2 --> mix
  mix --> out2["aligned f2"]
```

`identity` is the ablation: features pass through and the offset is zeros at the deepest map. The gate is taken from absolute difference, not from the change head.

### Fusion modes

Same op at every scale. Output width is what the decoder is built against.

| Mode | Op | Width |
|---|---|---|
| `signed` | `a - b` | C |
| `signed_product` | `concat(a - b, a * b)` | 2C |
| `concat` | `concat(a, b)` | 2C |
| `signed_concat` | `concat(a - b, a, b)` | 3C |
| `absdiff` | `\|a - b\|` | C |

`absdiff` is order-invariant for any downstream head. Signed difference is not: `h(b - a) = h(-(a - b))` equals `h(a - b)` only when `h` is even. That is why the pair-order term exists. The proposed config uses `signed_product`.

### Decoder

`UNetDecoder` walks coarsest to finest. At each finer scale it bilinear-upsamples, concatenates the skip, and applies Conv–BN–ReLU. A 1×1 head emits one channel at the finest skip (stride 4). `ProposedModel` then bilinear-upsamples logits, confidence, and directional logits to the input size. `align_corners` is false on those interpolations.

---

## 7. Baselines

Every model, including the non-learned floor, returns the same dict. Missing heads are `None`.

```mermaid
flowchart TB
  subgraph ssim [rgb_ssim]
    s1["mean |I1-I2| minus threshold"] --> sL["logits, no parameters that train the mask"]
  end

  subgraph fc [fc_siam_diff]
    fstack["stack to B,2,C,H,W"] --> torchgeo["TorchGeo FCSiamDiff over SMP"]
    torchgeo --> fL["logits"]
  end

  subgraph sia [siamese_resnet18]
    se["ResNet-18 forward_pair"] --> sa["AbsDiffFusion"]
    sa --> su["UNetDecoder"]
    su --> sL2["logits"]
  end

  subgraph pr [proposed / proposed_effnet]
    pe["shared encoder forward_pair"] --> pa["bounded or identity"]
    pa --> pf["signed fusion"]
    pf --> pd["UNetDecoder"]
    pd --> ph["confidence and directional heads"]
    ph --> swap["train-time swapped forward"]
  end
```

`fc_siam_diff` is TorchGeo's FC-Siam-Diff on segmentation-models-pytorch, default encoder ResNet-18 ImageNet. It is not Daudt's original shallow 16/32/64/128 net. `encoder_name` can be swapped. `siamese_resnet18` builds the encoder, abs-diff, and decoder directly and does not go through the alignment or fusion registries.

---

## 8. Losses

`build_loss` injects `pos_weight = (1 - π) / π` when the dataset class publishes `published_changed_pixel_ratio`.

```mermaid
flowchart TB
  out["model outputs"] --> bce["BCE with logits, ignore mask == -1"]
  batch["batch mask"] --> bce
  out --> dice["soft Dice"]
  batch --> dice
  bce --> seg["0.5 * bce + 0.5 * dice"]
  dice --> seg

  out --> po["pair order"]
  po --> bin["mean |sigmoid logits - sigmoid swapped| on valid pixels"]
  po --> dterm["MSE of directional vs swapped channels flipped 0 and 1"]
  bin --> pot["binary + directional"]
  dterm --> pot

  seg --> total["loss = seg + 0.1 * pair-order"]
  pot --> total

  miss["no logits_swapped"] --> zero["pair-order term is 0"]
  zero --> total
```

| Registry key | What it is |
|---|---|
| `bce_dice` | Segmentation term only. Root default. |
| `pair_order` | Swap term only. |
| `bce_dice_pairorder` | Sum above. Weight `0.1` on the swap term. Used by the proposed Modal jobs. |
| `calibration` | Registered placeholder. `compute` returns zero. |

Post-hoc calibration is `metrics/temperature.py` (`TemperatureScaler`). It is not fitted inside the trainer.

Under a swap, binary probabilities should match, and the two directional channels should exchange.

---

## 9. Metrics

`METRIC_REGISTRY` keys: `segmentation`, `boundary`, `region`, `calibration`, `robustness`, `swap_consistency`.

```mermaid
flowchart LR
  pred["outputs + batch"] --> seg["segmentation: corpus P, R, F1, IoU"]
  pred --> bnd["boundary: boundary IoU, HD95, ASD"]
  pred --> reg["region: component match F1"]
  pred --> cal["calibration: foreground ECE, Brier, NLL, risk-coverage"]
  pred --> rob["robustness: F1 retention under corruptions"]
  pred --> sw["swap_consistency: agreement after order flip"]
  pred --> ov["overlays: TP / FP / FN, best median worst"]
  pred --> eff["efficiency: params, pair FLOPs, latency, peak memory"]
```

`segmentation` sums TP, FP, and FN over the evaluated set, then computes F1 once. Ignore pixels are excluded. `cdlib.cli.evaluate` and `cdlib.cli.export_masks` still score `DummyPairCNN` on synthetic tensors. `cdlib.cli.benchmark` can construct the real registered models. Corruption functions live in `metrics/corruptions.py` (brightness, gamma, colour grade, blur, motion blur, JPEG, H.264, HEVC, shadow, viewpoint, occlusion).

---

## 10. Modal

```mermaid
flowchart TB
  user["modal run --detach train.py --suite all --gpu L4 --seeds 0,1,2"] --> app["modal_app.py"]
  app --> img["Debian 3.12 image, CUDA torch, repo mounted at /opt/cdlib"]
  app --> dataV["volume cdlib-data at /vol/data"]
  app --> resV["volume cdlib-results at /opt/cdlib/results"]
  app --> gpu["one GPU call per job: python -m cdlib.cli.train"]
  gpu --> hydraO["Hydra overrides from JOBS plus data_root and results_dir"]
  hydraO --> ckpt["checkpoints/*.pt on the results volume"]
```

`modal run --detach` keeps the GPU call alive after the laptop disconnects for a single triggered job. A `--suite` fan-out of many `spawn` calls does not have that property if the local entrypoint dies: only the last job stays. Resume is `train.checkpoint.resume_from=auto`.

Download helpers, used to fill the data volume, are `scripts/download_levir_cd.sh`, `scripts/download_sysu_cd.sh`, and `scripts/download_pcd_tsunami.sh`.

---

## Registry catalogue

| Registry | Module | Keys |
|---|---|---|
| `MODEL_REGISTRY` | `models/_model_registry.py` | `rgb_ssim`, `fc_siam_diff`, `siamese_resnet18`, `proposed`, `proposed_effnet` |
| `ENCODER_REGISTRY` | `models/encoders/registry.py` | `resnet18`, `efficientnet_b0`, `efficientnet_b2` |
| `FUSION_REGISTRY` | `models/fusion/registry.py` | `signed_fusion`, `absdiff` |
| `ALIGNMENT_REGISTRY` | `models/alignment/registry.py` | `bounded`, `identity` |
| `DECODER_REGISTRY` | `models/decoders/registry.py` | `unet_decoder` |
| `LOSS_REGISTRY` | `losses/registry.py` | `bce_dice`, `pair_order`, `bce_dice_pairorder`, `calibration` |
| `DATASET_REGISTRY` | `data/registry.py` | `sysu_cd`, `levir_cd`, `pcd`, `video_pairs`, `videosham`, `dvi` |
| `METRIC_REGISTRY` | `metrics/registry.py` | `segmentation`, `boundary`, `region`, `calibration`, `robustness`, `swap_consistency` |

Heads are not registered. `ProposedModel` constructs `ConfidenceHead` and `DirectionalHead` when `heads.confidence` and `heads.directional` are set. Optimizer names are a dict inside `build_optimizer`: `adam`, `adamw`, `sgd`, `rmsprop`.
