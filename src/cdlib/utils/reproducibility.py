"""Compatibility alias for the P5 slice. Use ``cdlib.utils.seed.set_seed``."""

from cdlib.utils.seed import get_rng_state, set_rng_state, set_seed

__all__ = ["get_rng_state", "set_rng_state", "set_seed"]
