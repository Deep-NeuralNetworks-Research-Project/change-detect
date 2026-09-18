"""METRIC_REGISTRY — one new metric file + one register() line."""
from __future__ import annotations

from cdlib.metrics.base import Metric
from cdlib.utils.registry import Registry

METRIC_REGISTRY: Registry = Registry("metric")


def build_metric(key: str, **kwargs) -> Metric:
    return METRIC_REGISTRY.build(key, **kwargs)


def _register_builtins() -> None:
    # Imported here to avoid circular imports at module load.
    from cdlib.metrics.boundary import BoundaryMetric
    from cdlib.metrics.calibration import CalibrationMetric
    from cdlib.metrics.region import RegionMetric
    from cdlib.metrics.robustness import RobustnessMetric
    from cdlib.metrics.segmentation import SegmentationMetric
    from cdlib.metrics.swap import SwapConsistencyMetric

    METRIC_REGISTRY.register("segmentation")(SegmentationMetric)
    METRIC_REGISTRY.register("boundary")(BoundaryMetric)
    METRIC_REGISTRY.register("region")(RegionMetric)
    METRIC_REGISTRY.register("calibration")(CalibrationMetric)
    METRIC_REGISTRY.register("robustness")(RobustnessMetric)
    METRIC_REGISTRY.register("swap_consistency")(SwapConsistencyMetric)


_register_builtins()
