# Corruption severity (ImageNet-C philosophy, re-tuned for RGB scene pairs)

Severity 1 = barely perceptible. Severity 5 = clearly degraded, still the same scene. Do **not** compound two corruptions in the main sweep.

Applied to **one frame only**. Run `which=t1` and `which=t2`.

| name | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| brightness (additive V) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| gamma | 0.85–1.18 | 0.7–1.4 | 0.55–1.8 | 0.4–2.2 | 0.25–3.0 |
| colour_grading (ASC-CDL jitter) | ±3% | ±6% | ±10% | ±15% | ±25% |
| gaussian_blur σ (px) | 0.5 | 1 | 2 | 3 | 4 |
| motion_blur length (px) | 3 | 7 | 15 | 25 | 35 |
| jpeg quality | 90 | 70 | 50 | 30 | 10 |
| h264 CRF | 18 | 23 | 28 | 35 | 45 |
| hevc CRF | 20 | 25 | 30 | 37 | 47 |
| shadow intensity | 0.1 | 0.2 | 0.35 | 0.5 | 0.7 |
| viewpoint translate / rotate | 1% / 0.5° | 2% / 1° | 4% / 2° | 8% / 4° | 16% / 8° |
| occlusion area | 2% | 5% | 10% | 20% | 35% |

Codec path: still → 10-frame static clip → extract middle frame. **Cache** the result (`cache_dir`); never ffmpeg inside the training loop.

Visual calibration: dump one example per (name, severity) from a held-out pair and check that s1 is subtle and s5 is ugly-but-recognisable. Tick this off in week 2–3 before quoting retention numbers.
