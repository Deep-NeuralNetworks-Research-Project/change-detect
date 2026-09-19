"""Helpers for the Modal training driver.

The Modal app is a thin Colab/Kaggle-style wrapper: it never contains
model, loss, or data logic. It only (a) places datasets and results on
Volumes and (b) execs ``python -m cdlib.cli.train`` with Hydra overrides.

This module is deliberately free of ``import modal`` so unit tests and CI
can cover the argv / path / job-catalog contract without the Modal SDK.
"""

from __future__ import annotations

import shlex
from pathlib import Path

# Mount points inside the Modal container. Passed as Hydra overrides /
# cwd, never hardcoded into dataset or trainer code (same rule as the
# pre-commit hook that bans /content/ and /kaggle/ outside notebooks/).
DATA_MOUNT = "/vol/data"
RESULTS_MOUNT = "/opt/cdlib/results"
REPO_MOUNT = "/opt/cdlib"
CONFIG_DIR = "/opt/cdlib/configs"

DATA_VOLUME = "cdlib-data"
RESULTS_VOLUME = "cdlib-results"

DEFAULT_DATASET = "sysu_cd"
DEFAULT_GPU = "T4"
DEFAULT_TIMEOUT_HOURS = 12.0

# Named jobs → extra Hydra overrides. Shared with ``modal_app.py --job``.
# Dataset name is inferred from ``data=`` / ``+experiment=`` (experiments
# pin data via Hydra defaults).
JOBS: dict[str, list[str]] = {
    "fcsiam-sysu": ["+experiment=baseline_fcsiamdiff_sysu"],
    "fcsiam-levir": ["data=levir_cd", "model=fc_siam_diff"],
    "fcsiam-pcd": ["data=pcd", "model=fc_siam_diff"],
    "siamese-sysu": ["data=sysu_cd", "model=siamese_resnet18"],
    "siamese-levir": ["data=levir_cd", "model=siamese_resnet18"],
    "siamese-pcd": ["data=pcd", "model=siamese_resnet18"],
    "proposed-levir": [
        "data=levir_cd",
        "model=proposed_effnet",
        "loss=bce_dice_pairorder",
    ],
    "proposed-sysu": [
        "data=sysu_cd",
        "model=proposed_effnet",
        "loss=bce_dice_pairorder",
    ],
    "ablation-no-align": ["+experiment=ablation_no_alignment"],
    "ablation-no-pairorder": ["+experiment=ablation_no_pairorder"],
    "ablation-no-conf": ["+experiment=ablation_no_confidence_gate"],
    "rgb-ssim-levir": ["data=levir_cd", "model=rgb_ssim", "train.epochs=1"],
}

SUITES: dict[str, list[str]] = {
    "baselines": [
        "rgb-ssim-levir",
        "fcsiam-sysu",
        "fcsiam-levir",
        "siamese-sysu",
        "siamese-levir",
    ],
    "proposed": ["proposed-levir", "proposed-sysu"],
    "ablations": [
        "ablation-no-align",
        "ablation-no-pairorder",
        "ablation-no-conf",
    ],
    "all": list(JOBS),
}

EXTRACT_SCRIPTS: dict[str, str] = {
    "levir_cd": "scripts/download_levir_cd.sh",
    "sysu_cd": "scripts/download_sysu_cd.sh",
    "pcd": "scripts/download_pcd_tsunami.sh",
}


def parse_overrides(overrides: str | list[str] | None) -> list[str]:
    """Split a user-supplied Hydra override string into argv tokens."""
    if overrides is None:
        return []
    if isinstance(overrides, list):
        return [str(tok) for tok in overrides if str(tok).strip()]
    text = str(overrides).strip()
    if not text:
        return []
    return shlex.split(text)


def parse_seeds(seeds: str | list[int] | None) -> list[int]:
    """``\"0,1,2\"`` or ``[0, 1, 2]``. Empty → no seed override."""
    if seeds is None or seeds == "":
        return []
    if isinstance(seeds, list):
        return [int(s) for s in seeds]
    return [int(p.strip()) for p in str(seeds).split(",") if p.strip()]


def _override_key(token: str) -> str | None:
    if "=" not in token:
        return None
    return token.split("=", 1)[0].lstrip("+")


def dataset_name_from_overrides(overrides: list[str], default: str = DEFAULT_DATASET) -> str:
    """Last ``data=...`` / ``data.name=...`` wins; else the Hydra default."""
    name = default
    for token in overrides:
        key = _override_key(token)
        if key is None:
            continue
        value = token.split("=", 1)[1]
        if key in {"data", "data.name"}:
            name = value
    return name


def build_train_argv(
    overrides: list[str] | None = None,
    *,
    data_root: str = DATA_MOUNT,
    results_dir: str = RESULTS_MOUNT,
    resume: bool = True,
    experiment_name: str | None = None,
    epochs: int | None = None,
) -> list[str]:
    """Hydra argv for ``python -m cdlib.cli.train``.

    Always injects ``data_root`` and ``results_dir`` so datasets and
    checkpoints resolve onto the Modal Volumes. User-supplied keys win.
    """
    user = list(overrides or [])
    keys = {_override_key(t) for t in user}
    injected: list[str] = []
    if "data_root" not in keys:
        injected.append(f"data_root={data_root}")
    if "results_dir" not in keys:
        injected.append(f"results_dir={results_dir}")
    if resume and "train.checkpoint.resume_from" not in keys:
        injected.append("train.checkpoint.resume_from=auto")
    if experiment_name and "experiment_name" not in keys:
        injected.append(f"experiment_name={experiment_name}")
    if epochs is not None and "train.epochs" not in keys:
        injected.append(f"train.epochs={int(epochs)}")
    return injected + user


def job_overrides(job: str) -> list[str]:
    if job not in JOBS:
        known = ", ".join(sorted(JOBS))
        raise KeyError(f"unknown job {job!r}. Known: {known}")
    return list(JOBS[job])


def expand_suite(
    suite: str | None = None,
    job: str | None = None,
    *,
    seeds: list[int] | None = None,
    extra: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Return ``(run_name, hydra_overrides)`` pairs for one job or a suite.

    ``run_name`` is used as ``experiment_name`` so checkpoints don't collide.
    """
    if suite and job:
        raise ValueError("pass only one of suite= or job=")
    if suite:
        if suite not in SUITES:
            known = ", ".join(sorted(SUITES))
            raise KeyError(f"unknown suite {suite!r}. Known: {known}")
        names = list(SUITES[suite])
    elif job:
        names = [job]
    else:
        raise ValueError("pass job= or suite=")

    extra = list(extra or [])
    seed_list = list(seeds or [])
    out: list[tuple[str, list[str]]] = []
    for name in names:
        base = job_overrides(name) + extra
        if not seed_list:
            out.append((name, base))
            continue
        for seed in seed_list:
            run = f"{name}_s{seed}"
            out.append((run, base + [f"train.seed={seed}"]))
    return out


def train_command(hydra_args: list[str], *, python: str | None = None) -> list[str]:
    """Full subprocess argv, including Hydra's config-path flags.

    ``cdlib.cli.train`` sets ``config_path`` relative to its source file.
    An absolute ``--config-path`` keeps that working when Modal mounts the
    package somewhere other than the repo-layout ``src/cdlib/cli/``.
    """
    exe = python or "python"
    return [
        exe,
        "-m",
        "cdlib.cli.train",
        "--config-path",
        CONFIG_DIR,
        "--config-name",
        "config",
        *hydra_args,
    ]


def dataset_root(data_root: str, name: str) -> Path:
    return Path(data_root) / name


def preflight_dataset(root: Path, name: str) -> None:
    """Fail fast if the Volume is missing this dataset.

    ``build_dataset`` constructs an *empty* dataset when the root is absent
    so CI can dry-run. Training on that empty set would look like a run
    and produce nothing useful — catch it here, in the driver.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {root}. Upload it first:\n"
            f"  modal volume put {DATA_VOLUME} <local-{name}-dir> /{name}\n"
            f"  # or: modal run modal_app.py --upload <local-{name}-dir> --dest {name}"
        )
    if not any(root.rglob("*")):
        raise FileNotFoundError(
            f"Dataset {name!r} at {root} exists but is empty. "
            f"Upload the split directories (train/val/test) into that path."
        )


def parse_gpu(gpu: str) -> str | list[str]:
    """``T4`` or comma-separated fallbacks ``T4,L4`` (Modal respects order)."""
    parts = [p.strip() for p in str(gpu).split(",") if p.strip()]
    if not parts:
        return DEFAULT_GPU
    if len(parts) == 1:
        return parts[0]
    return parts
