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
# EfficientNet launches (bare ``modal run train.py`` and the proposed-* jobs)
# request an A100. Batch 8 leaves that GPU idle; the speed overrides below
# raise the batch and shorten the schedule so a SYSU run lands near 6 hours.
# That figure is an estimate from the T4/A100 notes in the research briefs,
# not a measured step time.
TRAIN_GPU = "A100"
DEFAULT_TIMEOUT_HOURS = 12.0

# Paper recipe is 200 epochs, batch 8, AdamW 3e-4. This speed recipe keeps
# the same optimizer and scales the learning rate by sqrt(32/8) = 2.
# 100 epochs and validation every 4 epochs are what bring the wall clock
# down; cosine T_max follows train.epochs.
EFFNET_SPEED: tuple[tuple[str, str], ...] = (
    ("train.epochs", "100"),
    ("train.batch_size", "32"),
    ("train.optimizer.lr", "6e-4"),
    ("train.val_interval", "4"),
    ("data.num_workers", "8"),
)

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
        *(f"{key}={value}" for key, value in EFFNET_SPEED),
    ],
    "proposed-sysu": [
        "data=sysu_cd",
        "model=proposed_effnet",
        "loss=bce_dice_pairorder",
        *(f"{key}={value}" for key, value in EFFNET_SPEED),
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


# Bare ``modal run train.py`` (no --job / --suite) trains this model.
# EfficientNet-B0 ImageNet, signed fusion, bounded alignment, pair-order loss.
# An explicit ``model=`` or ``+experiment=`` on the command line replaces it.
EFFNET_MODEL = "proposed_effnet"
EFFNET_LOSS = "bce_dice_pairorder"


def _has_model_choice(overrides: list[str]) -> bool:
    for token in overrides:
        key = _override_key(token)
        if key in {"model", "model.name", "experiment"}:
            return True
    return False


def default_effnet_overrides(extra: list[str] | None = None) -> list[str]:
    """Hydra overrides for an EfficientNet-only launch.

    Dataset and other tokens in ``extra`` are kept. ``model=`` and
    ``+experiment=`` already in ``extra`` are left unchanged.
    """
    user = list(extra or [])
    if _has_model_choice(user):
        return user
    keys = {_override_key(t) for t in user}
    injected: list[str] = [f"model={EFFNET_MODEL}"]
    if "loss" not in keys:
        injected.append(f"loss={EFFNET_LOSS}")
    for key, value in EFFNET_SPEED:
        if key not in keys:
            injected.append(f"{key}={value}")
    return injected + user


def select_train_gpu(gpu: str = "", *, job: str = "", suite: str = "") -> str:
    """GPU for a training launch.

    An explicit ``--gpu`` is kept. A bare EfficientNet launch and the
    ``proposed-*`` jobs use :data:`TRAIN_GPU`. Every other job stays on
    :data:`DEFAULT_GPU`.
    """
    chosen = str(gpu).strip()
    if chosen:
        return chosen
    if not suite and (not job or job.startswith("proposed-")):
        return TRAIN_GPU
    return DEFAULT_GPU


def default_effnet_run(extra: list[str] | None = None) -> tuple[str, list[str]]:
    """``(experiment_name, overrides)`` for a bare EfficientNet launch.

    Checkpoints land in ``effnet-<dataset>`` on the results Volume.
    """
    overrides = default_effnet_overrides(extra)
    dataset = dataset_name_from_overrides(overrides)
    return f"effnet-{dataset}", overrides


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


def modal_cli_argv(argv: list[str], script: str) -> list[str]:
    """Turn ``python train.py ...`` into ``modal run [--detach] <script> ...``.

    ``--detach`` belongs to the ``modal run`` CLI, so it is hoisted in front
    of the script. Every other token is an entrypoint argument.
    """
    detach = False
    rest: list[str] = []
    for tok in argv:
        if tok == "--detach":
            detach = True
        else:
            rest.append(tok)
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd.append(script)
    cmd.extend(rest)
    return cmd


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
