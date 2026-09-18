# Dice vs calibration — raise with P2 and P4 in week 4

The planned **weighted BCE + Dice** loss fights the calibration deliverable.

- Mehrtash et al., IEEE TMI 2020: Dice-trained FCNs are more overconfident than CE-trained ones.
- Yeung et al., DSC++: "the DSC loss is poorly calibrated, resulting in overconfident predictions."
- Mukhoti et al., NeurIPS 2020: Focal *improves* calibration relative to CE. The problem is specifically the Dice term, not imbalance handling in general.

**Decision P5 owns:**

1. Default fix (near-zero cost): post-hoc temperature scaling fitted on **shift-representative** validation (Ovadia et al., NeurIPS 2019 — a T fitted on clean val does not transfer). Implemented in `cdlib.metrics.temperature.TemperatureScaler` (`fitted_on="clean_val"` is rejected).
2. Only if class-wise / foreground ECE still fails after (1): a calibration-aware auxiliary term (P4's `losses/calibration.py`).

P2/P4 implement against this decision; P5 measures whether it worked.
