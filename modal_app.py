"""Modal training driver — bootstrap + one CLI call, nothing else.

Same contract as ``notebooks/colab_train_driver.ipynb``:
the GPU container only execs ``python -m cdlib.cli.train`` with Hydra
overrides. Model / loss / data logic stays in ``cdlib``.

Setup (once)::

    pip install -e ".[modal]"
    modal setup
    modal secret create wandb WANDB_API_KEY=$WANDB_API_KEY   # optional

Upload a dataset (LEVIR-CD shown)::

    modal volume put cdlib-data data/levir_cd /levir_cd

Train one named job, or the whole paper matrix::

    modal run modal_app.py --job siamese-levir --gpu T4
    modal run --detach modal_app.py --suite all --gpu L4 --seeds 0,1,2

Resume after a kill is the default (``train.checkpoint.resume_from=auto``).
Results live on Volume ``cdlib-results``::

    modal volume get cdlib-results / ./modal-results
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import modal

# So ``modal run modal_app.py`` works without a prior ``pip install -e .``.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cdlib.cli.modal_runtime import (  # noqa: E402
    CONFIG_DIR,
    DATA_MOUNT,
    DATA_VOLUME,
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_HOURS,
    EXTRACT_SCRIPTS,
    JOBS,
    REPO_MOUNT,
    RESULTS_MOUNT,
    RESULTS_VOLUME,
    SUITES,
    build_train_argv,
    dataset_name_from_overrides,
    dataset_root,
    default_effnet_run,
    expand_suite,
    parse_gpu,
    parse_overrides,
    parse_seeds,
    preflight_dataset,
    select_train_gpu,
    train_command,
)

REPO_ROOT = Path(__file__).resolve().parent

_IGNORE = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".grok",
    "data",
    "results",
    "checkpoints",
    "wandb",
    "*.egg-info",
    ".DS_Store",
    "files (9)",
    "modal-results",
]

# CUDA wheels first; remaining deps from pyproject without re-pinning torch.
_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
_REST = [
    "torchgeo>=0.6",
    "torchmetrics>=1.0",
    "segmentation-models-pytorch>=0.3",
    "timm>=0.9",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "wandb>=0.15",
    "numpy>=1.24",
    "Pillow>=9.0",
    "opencv-python-headless>=4.8",
    "scikit-image>=0.21",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "albumentations>=1.4",
    "pyyaml>=6.0",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "libgomp1", "ffmpeg", "unzip", "wget", "ca-certificates")
    .uv_pip_install(
        "torch>=2.2",
        "torchvision>=0.17",
        extra_options=f"--index-url {_CUDA_INDEX}",
    )
    .uv_pip_install(*_REST)
    .env(
        {
            "PYTHONPATH": f"{REPO_MOUNT}/src",
            "HYDRA_FULL_ERROR": "1",
            "CDLIB_DATA_ROOT": DATA_MOUNT,
            "CDLIB_RESULTS_ROOT": RESULTS_MOUNT,
        }
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REPO_MOUNT,
        ignore=_IGNORE,
    )
)

data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

app = modal.App("cdlib-train", image=image)

_VOLUMES = {
    DATA_MOUNT: data_vol,
    RESULTS_MOUNT: results_vol,
}


def _wandb_secrets() -> list[modal.Secret]:
    """Optional. Trainer already continues without W&B if the key is absent."""
    named = os.environ.get("MODAL_WANDB_SECRET")
    if named:
        return [modal.Secret.from_name(named, required_keys=["WANDB_API_KEY"])]
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return [modal.Secret.from_dict({"WANDB_API_KEY": key})]
    return []


def _commit_loop(stop: threading.Event, interval: float = 60.0) -> None:
    """Flush the results Volume while training so a preemption keeps checkpoints."""
    while not stop.wait(interval):
        results_vol.commit()


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=int(DEFAULT_TIMEOUT_HOURS * 3600),
    volumes=_VOLUMES,
    memory=32768,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
    single_use_containers=True,
)
def train_remote(hydra_args: list[str]) -> int:
    """Run the existing train CLI against Volumes. No training logic here."""
    os.chdir(REPO_MOUNT)
    Path(RESULTS_MOUNT).mkdir(parents=True, exist_ok=True)
    if not Path(CONFIG_DIR).is_dir():
        raise FileNotFoundError(f"Hydra configs not mounted at {CONFIG_DIR}")

    name = dataset_name_from_overrides(hydra_args)
    root = dataset_root(DATA_MOUNT, name)
    preflight_dataset(root, name)

    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[modal] device={gpu_name}  data_root={DATA_MOUNT}  dataset={name}  root={root}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The image must install a CUDA torch wheel "
            f"(index {_CUDA_INDEX})."
        )

    cmd = train_command(hydra_args, python=sys.executable)
    print("[modal] " + " ".join(cmd), flush=True)

    stop = threading.Event()
    flusher = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
    flusher.start()
    try:
        result = subprocess.run(cmd, cwd=REPO_MOUNT)
    finally:
        stop.set()
        flusher.join(timeout=5)
        results_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"cdlib.cli.train exited {result.returncode}")
    return result.returncode


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=15 * 60,
    memory=8192,
)
def smoke_remote() -> dict[str, str]:
    """GPU + CUDA torch + one forward/backward. No dataset required."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from cdlib.models.build import build_model

    info: dict[str, str] = {
        "cuda": str(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("smoke expected a GPU but torch.cuda.is_available() is False")

    configs_dir = str(Path(REPO_MOUNT) / "configs")
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=configs_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["model=fc_siam_diff", "model.encoder_weights=null"],
            )
        model = build_model(cfg).cuda()
        img1 = torch.rand(2, 3, 32, 32, device="cuda")
        img2 = torch.rand(2, 3, 32, 32, device="cuda")
        out = model(img1, img2)
        out["logits"].sum().backward()
        info["forward"] = "ok"
        info["logits"] = "x".join(str(s) for s in out["logits"].shape)
    finally:
        GlobalHydra.instance().clear()
    print(f"[modal] smoke ok: {info}", flush=True)
    return info


@app.function(image=image, volumes={DATA_MOUNT: data_vol, RESULTS_MOUNT: results_vol}, timeout=120)
def ls_remote() -> dict[str, list[str]]:
    """List dataset and results Volume contents."""

    def _listing(path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            return []
        return sorted(str(c.relative_to(p)) for c in p.iterdir())

    listing = {"data": _listing(DATA_MOUNT), "results": _listing(RESULTS_MOUNT)}
    print(listing, flush=True)
    return listing


@app.function(image=image, volumes={DATA_MOUNT: data_vol}, timeout=3600, cpu=2.0)
def extract_remote(dataset: str) -> str:
    """Run the repo's extract/verify script against an archive already on the Volume."""
    if dataset not in EXTRACT_SCRIPTS:
        known = ", ".join(sorted(EXTRACT_SCRIPTS))
        raise KeyError(f"no extract script for {dataset!r}. Known: {known}")
    script = Path(REPO_MOUNT) / EXTRACT_SCRIPTS[dataset]
    target = dataset_root(DATA_MOUNT, dataset)
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(script), str(target)]
    print("[modal] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_MOUNT)
    data_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} exited {result.returncode}")
    return str(target)


@app.local_entrypoint()
def main(
    gpu: str = "",
    overrides: str = "",
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    resume: bool = True,
    smoke: bool = False,
    dry: bool = False,
    upload: str = "",
    dest: str = "",
    ls: bool = False,
    retries: int = 3,
    job: str = "",
    suite: str = "",
    seeds: str = "",
    epochs: int = 0,
    list_jobs: bool = False,
    extract: str = "",
) -> None:
    """Launch training, a GPU smoke test, a Volume upload, or a listing.

    Examples::

        modal run modal_app.py --smoke --gpu T4
        modal run modal_app.py --list-jobs
        modal run modal_app.py --upload data/levir_cd --dest levir_cd
        modal run modal_app.py --job siamese-levir --gpu T4
        modal run --detach modal_app.py --suite all --gpu L4 --seeds 0,1,2
    """
    if list_jobs:
        print("jobs:")
        for name, ov in JOBS.items():
            print(f"  {name:24s} {' '.join(ov)}")
        print("suites:")
        for name, members in SUITES.items():
            print(f"  {name:24s} {', '.join(members)}")
        return
    if upload:
        _upload_local(upload, dest)
        return
    if extract:
        print(extract_remote.remote(extract))
        return
    if ls:
        print(ls_remote.remote())
        return
    if smoke:
        print(smoke_remote.with_options(gpu=parse_gpu(gpu.strip() or DEFAULT_GPU)).remote())
        return

    extra = parse_overrides(overrides)
    seed_list = parse_seeds(seeds)
    epochs_arg = int(epochs) if int(epochs) > 0 else None

    if suite or job:
        runs = expand_suite(
            suite=suite or None,
            job=job or None,
            seeds=seed_list or None,
            extra=extra,
        )
    else:
        # No --job and no --suite: proposed EfficientNet-B0, not the
        # Hydra default (FC-Siam-Diff / ResNet-18).
        runs = [default_effnet_run(extra)]

    planned: list[tuple[str, list[str]]] = []
    for run_name, ov in runs:
        argv = build_train_argv(
            ov,
            resume=resume,
            experiment_name=run_name,
            epochs=epochs_arg,
        )
        planned.append((run_name, argv))
        print(f"[modal] {run_name}: {' '.join(argv)}")

    if dry:
        print(f"[modal] dry-run: {len(planned)} job(s) not launched")
        return

    chosen_gpu = select_train_gpu(gpu, job=job, suite=suite)
    kwargs: dict = {
        "gpu": parse_gpu(chosen_gpu),
        "cpu": 8.0,
        "timeout": int(float(timeout_hours) * 3600),
        "secrets": _wandb_secrets(),
    }
    if retries > 0:
        kwargs["retries"] = modal.Retries(max_retries=int(retries), initial_delay=0.0)

    fn = train_remote.with_options(**kwargs)
    if len(planned) == 1:
        fn.remote(planned[0][1])
        return

    # Fan-out. --detach keeps these alive if you close the laptop.
    handles = [fn.spawn(argv) for _, argv in planned]
    print(f"[modal] spawned {len(handles)} jobs")
    for (run_name, _), handle in zip(planned, handles, strict=True):
        print(f"[modal] waiting {run_name} …", flush=True)
        handle.get()


def _upload_local(local_path: str, dest: str) -> None:
    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"local path does not exist: {src}")
    remote = dest.strip() or src.name
    if not remote.startswith("/"):
        remote = "/" + remote
    print(f"[modal] uploading {src} -> {DATA_VOLUME}:{remote}")
    with data_vol.batch_upload() as batch:
        if src.is_dir():
            batch.put_directory(str(src), remote)
        else:
            batch.put_file(str(src), remote)
    print("[modal] upload complete")
