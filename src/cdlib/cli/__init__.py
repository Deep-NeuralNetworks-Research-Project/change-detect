"""CLI entrypoints. Hydra drivers live in ``train`` / ``validate_configs``.

``modal_runtime`` is imported by the repo-root ``modal_app.py`` driver; it is
not a training entrypoint.

``evaluate``, ``export_masks``, and ``benchmark`` accept argparse plus
Hydra-shaped leftovers (``+metrics=all``, ``model=dummy``) so they stay
runnable without a composed config.
"""
