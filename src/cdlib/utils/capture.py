"""Auto-capture run metadata at training start for reproducibility.

Saves everything needed to reproduce a run:
    - Fully resolved Hydra config
    - Git commit hash + dirty diff
    - Exact CLI command
    - Environment snapshot (pip freeze, CUDA, GPU, Python)

All files are written to the run directory (e.g., ``results/exp_001/``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def capture_run_metadata(cfg: DictConfig, run_dir: str | Path) -> None:
    """Save all reproducibility metadata to the run directory.

    Args:
        cfg: Fully resolved Hydra config.
        run_dir: Directory to save metadata files into.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_config(cfg, run_dir)
    _save_git_info(run_dir)
    _save_command(run_dir)
    _save_environment(run_dir)

    logger.info(f"Run metadata captured in: {run_dir}")


def _save_config(cfg: DictConfig, run_dir: Path) -> None:
    """Save the fully resolved Hydra config."""
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    logger.debug(f"Config saved: {config_path}")


def _save_git_info(run_dir: Path) -> None:
    """Save git commit hash and dirty diff."""
    commit_path = run_dir / "commit.txt"
    lines = []

    try:
        # Current commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit.returncode == 0:
            lines.append(f"commit: {commit.stdout.strip()}")
        else:
            lines.append("commit: (not a git repo or git not available)")

        # Check if working tree is dirty
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode == 0:
            dirty_files = status.stdout.strip()
            if dirty_files:
                lines.append(f"dirty: yes\n{dirty_files}")
            else:
                lines.append("dirty: no")

        # Capture the diff for dirty files
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if diff.returncode == 0 and diff.stdout.strip():
            lines.append(f"\n--- git diff --stat ---\n{diff.stdout.strip()}")

    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines.append("git: not available")

    with open(commit_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.debug(f"Git info saved: {commit_path}")


def _save_command(run_dir: Path) -> None:
    """Save the exact CLI command used to start this run."""
    command_path = run_dir / "command.txt"
    with open(command_path, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    logger.debug(f"Command saved: {command_path}")


def _save_environment(run_dir: Path) -> None:
    """Save environment snapshot: pip freeze, CUDA, GPU, Python."""
    env_path = run_dir / "env.txt"
    lines = []

    # Python version
    lines.append(f"python: {sys.version}")
    lines.append(f"executable: {sys.executable}")

    # PyTorch + CUDA info
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version: {torch.version.cuda}")
            lines.append(f"cudnn_version: {torch.backends.cudnn.version()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"gpu_{i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        lines.append("torch: not installed")

    # pip freeze
    lines.append("\n--- pip freeze ---")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if freeze.returncode == 0:
            lines.append(freeze.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines.append("(pip freeze unavailable)")

    # Key environment variables (sanitized — no secrets)
    lines.append("\n--- environment variables ---")
    safe_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    for var in safe_vars:
        val = os.environ.get(var, "(not set)")
        lines.append(f"{var}={val}")

    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.debug(f"Environment saved: {env_path}")
