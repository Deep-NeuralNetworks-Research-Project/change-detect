"""CLI entrypoints. Hydra drivers live in ``train`` / ``validate_configs``.

``evaluate``, ``export_masks``, and ``benchmark`` accept argparse plus
Hydra-shaped leftovers (``+metrics=all``, ``model=dummy``) so they stay
runnable without a composed config.
"""
