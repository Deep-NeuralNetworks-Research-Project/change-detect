# `nuisance_label` taxonomy — P5 proposal for P1 (week 2)

P1 populates `nuisance_label` on every sample. P5's stratified evaluation bins on the same ids. **Agree this before either of us ships a loader or a heatmap.**

| id | name | who uses it |
|---|---|---|
| 0 | clean | both |
| 1 | brightness / gamma / exposure | P5 eval corruptions `brightness`, `gamma` |
| 2 | colour_grading (ASC-CDL / white-balance) | P5 `colour_grading` |
| 3 | gaussian_blur / defocus | P5 `gaussian_blur` |
| 4 | motion_blur | P5 `motion_blur` |
| 5 | jpeg | P5 `jpeg` |
| 6 | codec_h264 | P5 `h264` |
| 7 | codec_hevc | P5 `hevc` |
| 8 | shadow | P5 `shadow` |
| 9 | viewpoint / homography | P5 `viewpoint` |
| 10 | occlusion | P5 `occlusion` |
| -1 | unknown | default when we do not know — **never silently write 0** |

Constants live in `cdlib.metrics.corruptions.NUISANCE_ID`.

P1's training-negative mining and P5's evaluation suite must call the **same** `apply_to_pair` implementation. Training uses it as augmentation; evaluation uses it as a frozen test-time transform with a fixed RNG seed per pair.
