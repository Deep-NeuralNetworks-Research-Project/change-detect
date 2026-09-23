"""Train cdlib on Modal.

The GPU container, Volumes, and Hydra invocation live in ``modal_app.py``.
This file is the short command for that same app: it never contains model,
loss, or data logic. The remote process is ``python -m cdlib.cli.train``.

    modal run train.py --list-jobs
    modal run train.py --smoke --gpu T4
    modal run --detach train.py
    modal run --detach train.py --overrides "data=levir_cd"
    modal run --detach train.py --job proposed-levir

A launch with no ``--job`` and no ``--suite`` trains the proposed
EfficientNet-B0 model on an A100. The speed recipe is 100 epochs, batch
32, AdamW 6e-4, validation every 4 epochs, aimed at about 6 hours on
SYSU-CD. Checkpoints are written under ``effnet-sysu_cd``. Pass
``data=levir_cd`` for LEVIR-CD (``effnet-levir_cd``). An explicit
``model=`` or ``+experiment=`` replaces that default. ``--gpu T4`` forces
the slower GPU.

``python train.py`` forwards to ``modal run`` on this file. Put ``--detach``
anywhere in that command; it is hoisted onto ``modal run``.

Datasets must already be on the ``cdlib-data`` Volume (``modal volume put``
or ``modal run train.py --upload ...``). Checkpoints resume by default
(``train.checkpoint.resume_from=auto``) onto ``cdlib-results``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# So ``modal run train.py`` works without a prior ``pip install -e .``.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modal_app import app, main  # noqa: E402

from cdlib.cli.modal_runtime import modal_cli_argv  # noqa: E402

__all__ = ["app", "main", "modal_cli_argv"]


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(modal_cli_argv(sys.argv[1:], str(Path(__file__).resolve()))))
