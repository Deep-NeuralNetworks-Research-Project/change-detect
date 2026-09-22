"""Train cdlib on Modal.

The GPU container, Volumes, and Hydra invocation live in ``modal_app.py``.
This file is the short command for that same app: it never contains model,
loss, or data logic. The remote process is ``python -m cdlib.cli.train``.

    modal run train.py --list-jobs
    modal run train.py --smoke --gpu T4
    modal run --detach train.py --job proposed-levir --gpu T4
    modal run --detach train.py --suite all --gpu L4 --seeds 0,1,2
    python train.py --job siamese-levir --gpu T4

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
