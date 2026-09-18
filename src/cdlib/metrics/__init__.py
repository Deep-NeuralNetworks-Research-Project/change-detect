"""P5-owned metrics. Contract: reset() / update(outputs, batch) / compute() -> dict[str, float]."""

from cdlib.metrics.base import Metric
from cdlib.metrics.boundary import BoundaryMetric
from cdlib.metrics.calibration import CalibrationMetric
from cdlib.metrics.efficiency import EfficiencyReport, benchmark_model
from cdlib.metrics.overlays import overlay_tp_fp_fn
from cdlib.metrics.region import RegionMetric
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric
from cdlib.metrics.robustness import RobustnessMetric
from cdlib.metrics.segmentation import SegmentationMetric
from cdlib.metrics.stats import ablation_table, bootstrap_ci, summarize_seeds
from cdlib.metrics.temperature import TemperatureScaler

__all__ = [
    "METRIC_REGISTRY",
    "BoundaryMetric",
    "CalibrationMetric",
    "EfficiencyReport",
    "Metric",
    "RegionMetric",
    "RobustnessMetric",
    "SegmentationMetric",
    "TemperatureScaler",
    "ablation_table",
    "benchmark_model",
    "bootstrap_ci",
    "build_metric",
    "overlay_tp_fp_fn",
    "summarize_seeds",
]
