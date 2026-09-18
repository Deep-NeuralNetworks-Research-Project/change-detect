# Figure standards (P5 owns these)

- Line plots / diagrams: vector PDF (`savefig(..., format='pdf')`).
- **No red-green.** Validate with a CVD simulator before freezing a figure.
- Change-mask overlay (Okabe-Ito):
  - TP: blue `(0, 114, 178)`
  - FP: vermillion/orange `(213, 94, 0)`
  - FN: magenta/purple `(204, 121, 167)`
- Confidence / signed-change maps: diverging **blue ↔ orange**, not red ↔ green.
- Qualitative export: best / median / worst pair per split (`cdlib.cli.export_masks`).
- Overlay implementation: `cdlib.metrics.overlays.overlay_tp_fp_fn`.
