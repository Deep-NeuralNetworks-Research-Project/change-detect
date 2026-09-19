"""Modal driver helpers — no Modal SDK, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from cdlib.cli.modal_runtime import (
    CONFIG_DIR,
    DATA_MOUNT,
    DEFAULT_DATASET,
    JOBS,
    RESULTS_MOUNT,
    SUITES,
    build_train_argv,
    dataset_name_from_overrides,
    dataset_root,
    expand_suite,
    parse_gpu,
    parse_overrides,
    parse_seeds,
    preflight_dataset,
    train_command,
)


def test_parse_overrides_splits_hydra_tokens():
    assert parse_overrides("data=levir_cd model=siamese_resnet18") == [
        "data=levir_cd",
        "model=siamese_resnet18",
    ]
    assert parse_overrides('train.epochs=2 "model=fc_siam_diff"') == [
        "train.epochs=2",
        "model=fc_siam_diff",
    ]
    assert parse_overrides("") == []
    assert parse_overrides(None) == []
    assert parse_overrides(["data=sysu_cd"]) == ["data=sysu_cd"]


def test_dataset_name_from_overrides_last_wins():
    assert dataset_name_from_overrides([]) == DEFAULT_DATASET
    assert dataset_name_from_overrides(["model=fc_siam_diff"]) == DEFAULT_DATASET
    assert dataset_name_from_overrides(["data=levir_cd"]) == "levir_cd"
    assert dataset_name_from_overrides(["data=levir_cd", "data=pcd", "model=rgb_ssim"]) == "pcd"
    assert dataset_name_from_overrides(["data.name=videosham"]) == "videosham"


def test_build_train_argv_injects_data_root_and_resume():
    argv = build_train_argv(["data=levir_cd", "model=siamese_resnet18"])
    assert f"data_root={DATA_MOUNT}" in argv
    assert f"results_dir={RESULTS_MOUNT}" in argv
    assert "train.checkpoint.resume_from=auto" in argv
    assert argv[-2:] == ["data=levir_cd", "model=siamese_resnet18"]


def test_build_train_argv_user_data_root_wins():
    argv = build_train_argv(["data_root=/mnt/custom", "data=levir_cd"])
    assert "data_root=/mnt/custom" in argv
    assert f"data_root={DATA_MOUNT}" not in argv


def test_build_train_argv_resume_false_skips_injection():
    argv = build_train_argv(["data=levir_cd"], resume=False)
    assert all(not t.startswith("train.checkpoint.resume_from") for t in argv)


def test_build_train_argv_user_resume_wins():
    argv = build_train_argv(["train.checkpoint.resume_from=/vol/ckpt.pt"])
    assert argv.count("train.checkpoint.resume_from=auto") == 0
    assert "train.checkpoint.resume_from=/vol/ckpt.pt" in argv


def test_dataset_root_joins_volume_mount():
    assert dataset_root("/vol/data", "levir_cd") == Path("/vol/data/levir_cd")


def test_preflight_dataset_missing(tmp_path: Path):
    missing = tmp_path / "levir_cd"
    with pytest.raises(FileNotFoundError, match="levir_cd"):
        preflight_dataset(missing, "levir_cd")


def test_preflight_dataset_empty_dir(tmp_path: Path):
    empty = tmp_path / "sysu_cd"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="empty"):
        preflight_dataset(empty, "sysu_cd")


def test_preflight_dataset_ok(tmp_path: Path):
    root = tmp_path / "sysu_cd"
    (root / "train" / "time1").mkdir(parents=True)
    (root / "train" / "time1" / "00001.png").write_bytes(b"x")
    preflight_dataset(root, "sysu_cd")


def test_parse_gpu_fallbacks():
    assert parse_gpu("T4") == "T4"
    assert parse_gpu("T4,L4") == ["T4", "L4"]
    assert parse_gpu("  ") == "T4"


def test_train_command_pins_absolute_config_path():
    argv = train_command(["data=levir_cd"], python="/usr/bin/python")
    assert argv[:6] == [
        "/usr/bin/python",
        "-m",
        "cdlib.cli.train",
        "--config-path",
        CONFIG_DIR,
        "--config-name",
    ]
    assert argv[-2:] == ["config", "data=levir_cd"]


def test_parse_seeds():
    assert parse_seeds("") == []
    assert parse_seeds("0,1,2") == [0, 1, 2]
    assert parse_seeds([0, 2]) == [0, 2]


def test_expand_suite_all_covers_catalog():
    runs = expand_suite(suite="all")
    names = [n for n, _ in runs]
    assert set(names) == set(JOBS)
    assert "baselines" in SUITES and "all" in SUITES


def test_expand_suite_seeds_suffixes_run_name():
    runs = expand_suite(job="siamese-levir", seeds=[0, 1])
    assert [n for n, _ in runs] == ["siamese-levir_s0", "siamese-levir_s1"]
    assert "train.seed=0" in runs[0][1]
    assert "data=levir_cd" in runs[0][1]


def test_expand_suite_rejects_both():
    with pytest.raises(ValueError, match="only one"):
        expand_suite(suite="all", job="siamese-levir")


def test_job_overrides_unknown():
    with pytest.raises(KeyError, match="unknown job"):
        expand_suite(job="not-a-job")
