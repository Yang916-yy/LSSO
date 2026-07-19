from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
import torch

from experiments.imagenet_wds_train import (
    apply_virtual_group_mixup,
    checkpoint_metadata,
    complete_hf_split_cache,
    create_model_ema,
    create_official_optimizer,
    create_training_model,
    make_loaders,
    parse_args,
    resolve_run_mode,
    resize_position_embedding,
    shard_commands,
    validate_initialization_checkpoint,
    validate_resume_checkpoint,
)


def test_stage_defaults_cover_low_resolution_and_refinement() -> None:
    pretrain = parse_args(["--stage", "pretrain", "--no-resume"])
    assert (pretrain.epochs, pretrain.image_size) == (800, 192)
    assert (pretrain.lr, pretrain.warmup_epochs) == (3e-3, 5)
    assert pretrain.optimizer == "fusedlamb"
    assert pretrain.drop_path_rate == 0.2
    assert pretrain.bce_loss and pretrain.repeated_aug == 3
    assert pretrain.batch_size * pretrain.grad_accum == 2048
    assert pretrain.augmentation_group_size == 256
    assert pretrain.runtime_augmentation_group_size == 256
    assert pretrain.batch_size // pretrain.augmentation_group_size == 2
    assert not pretrain.rrlsso_extended_diagnostics

    finetune = parse_args(
        [
            "--stage",
            "finetune224",
            "--init-checkpoint",
            "pretrain/best.pt",
            "--no-resume",
        ]
    )
    assert (finetune.epochs, finetune.image_size) == (20, 224)
    assert (finetune.lr, finetune.warmup_epochs) == (1e-5, 5)
    assert finetune.min_lr == 1e-5
    assert finetune.optimizer == "adamw"
    assert finetune.augmentation_group_size == 64
    assert finetune.output.endswith("finetune224")


def test_large_profile_preserves_physical_batch_with_virtual_groups() -> None:
    args = parse_args(["--model", "deit3_large_patch16_rrlsso"])
    assert (args.batch_size, args.grad_accum) == (128, 16)
    assert args.augmentation_group_size == 64
    assert args.batch_size // args.augmentation_group_size == 2


def test_extended_solve_diagnostics_are_explicitly_opt_in() -> None:
    args = parse_args(["--rrlsso-extended-diagnostics"])
    assert args.rrlsso_extended_diagnostics


def test_virtual_group_mixup_draws_once_per_official_local_batch() -> None:
    class RecordingMixup:
        def __init__(self) -> None:
            self.group_sizes = []

        def __call__(self, images, labels):
            self.group_sizes.append(images.shape[0])
            images.add_(len(self.group_sizes))
            return images, labels[:, None].repeat(1, 2) + len(self.group_sizes)

    images = torch.zeros(8, 1)
    labels = torch.arange(8)
    mixup = RecordingMixup()
    mixed_images, mixed_labels = apply_virtual_group_mixup(
        images, labels, mixup, group_size=4
    )
    assert mixup.group_sizes == [4, 4]
    assert mixed_images[:4].eq(1).all() and mixed_images[4:].eq(2).all()
    assert mixed_labels.shape == (8, 2)


def test_ema_is_fresh_for_refinement_and_restored_only_for_resume() -> None:
    model = torch.nn.Linear(2, 2)
    fresh = create_model_ema(model, enabled=True, decay=0.9)
    assert fresh is not None
    assert torch.equal(fresh.module.weight, model.weight)

    saved = {key: value.clone() for key, value in fresh.module.state_dict().items()}
    saved["weight"].fill_(7.0)
    resumed = create_model_ema(
        model,
        enabled=True,
        decay=0.9,
        resume_checkpoint={"model_ema": saved},
    )
    assert resumed is not None
    assert resumed.module.weight.eq(7.0).all()


def test_position_embedding_is_resized_from_192_to_224() -> None:
    source_args = parse_args(["--model", "deit3_base_patch16_rrlsso"])
    target_args = parse_args(
        [
            "--model", "deit3_base_patch16_rrlsso", "--stage", "finetune224",
            "--init-checkpoint", "dummy.pt", "--no-resume",
        ]
    )
    source = create_training_model(source_args)
    target = create_training_model(target_args)
    resized = resize_position_embedding(source.state_dict(), target)
    assert source.pos_embed.shape[1] == 12 * 12
    assert target.pos_embed.shape[1] == 14 * 14
    assert resized["pos_embed"].shape == target.pos_embed.shape
    target.load_state_dict(resized)


def test_hf_shard_commands_have_stable_non_overlapping_names(tmp_path) -> None:
    train = shard_commands("train", tmp_path, "timm/imagenet-1k-wds")
    validation = shard_commands("validation", tmp_path, "timm/imagenet-1k-wds")
    assert len(train) == 1024
    assert len(validation) == 64
    assert len(set(train + validation)) == 1088
    assert "imagenet1k-train-0000.tar" in train[0]
    assert "imagenet1k-validation-63.tar" in validation[-1]


def test_remote_train_workers_do_not_persist_into_validation(tmp_path) -> None:
    args = parse_args(
        [
            "--cache-dir",
            str(tmp_path),
            "--workers",
            "2",
            "--eval-workers",
            "1",
            "--shard-limit",
            "1",
            "--steps-per-epoch",
            "1",
        ]
    )
    train_loader, val_loader = make_loaders(args)
    assert not train_loader.pipeline[0].persistent_workers
    assert not val_loader.pipeline[0].persistent_workers


def test_complete_remote_train_cache_enables_persistent_workers(tmp_path) -> None:
    (tmp_path / "imagenet1k-train-0000.tar").write_bytes(b"cached")
    args = parse_args(
        [
            "--cache-dir",
            str(tmp_path),
            "--workers",
            "2",
            "--eval-workers",
            "1",
            "--shard-limit",
            "1",
            "--steps-per-epoch",
            "1",
        ]
    )
    train_loader, val_loader = make_loaders(args, train_seed_offset=1)
    assert train_loader.pipeline[0].persistent_workers
    assert not val_loader.pipeline[0].persistent_workers


def test_hf_cache_barrier_is_a_noop_when_bounded_split_is_complete(tmp_path) -> None:
    (tmp_path / "imagenet1k-train-0000.tar").write_bytes(b"cached")
    complete_hf_split_cache(
        "train",
        tmp_path,
        "owner/repo",
        workers=1,
        shard_limit=1,
    )


def test_hf_cache_barrier_fills_every_missing_shard_before_return(tmp_path) -> None:
    downloaded = []

    def fake_run(command, **kwargs):
        filename = command[command.index("--filename") + 1]
        cache_dir = command[command.index("--cache-dir") + 1]
        downloaded.append(filename)
        (tmp_path / filename).write_bytes(b"complete")
        assert cache_dir == str(tmp_path)
        assert kwargs["stdout"] is subprocess.DEVNULL
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("experiments.imagenet_wds_train.subprocess.run", side_effect=fake_run):
        complete_hf_split_cache(
            "train",
            tmp_path,
            "owner/repo",
            workers=0,
            shard_limit=3,
        )
    assert sorted(downloaded) == [
        "imagenet1k-train-0000.tar",
        "imagenet1k-train-0001.tar",
        "imagenet1k-train-0002.tar",
    ]


def test_registered_model_configuration_is_trace_normalized() -> None:
    args = parse_args(
        [
            "--model",
            "deit3_small_patch16_rrlsso",
            "--rank",
            "8",
        ]
    )
    model = create_training_model(args)
    assert model.rrlsso_config["basis_normalization"] == "trace"
    assert model.rrlsso_config["rank_rotary"] == "ordinary-1d"
    assert model.rrlsso_config["alpha_init"] == 1.2
    assert model.rrlsso_config["layerscale_init"] == 1e-4
    assert torch.isfinite(model.pos_embed).all()


def test_pretraining_never_silently_drops_fused_lamb(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    args = parse_args(["--model", "deit3_base_patch16_rrlsso", "--epochs", "1"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA and NVIDIA Apex FusedLAMB"):
        create_official_optimizer(model, args)

    args.allow_unfused_lamb = True
    _, implementation = create_official_optimizer(model, args)
    assert implementation == "lamb"


def test_solve_scalars_are_excluded_from_ordinary_weight_decay() -> None:
    args = parse_args(
        ["--model", "deit3_small_patch16_rrlsso", "--rank", "8", "--allow-unfused-lamb"]
    )
    model = create_training_model(args)
    optimizer, _ = create_official_optimizer(model, args)
    decay_by_parameter = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    solve_scalars = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("theta_gain", "theta_alpha"))
    ]
    assert solve_scalars
    assert all(decay_by_parameter[id(parameter)] == 0.0 for parameter in solve_scalars)


def test_checkpoint_modes_are_explicit_and_non_overwriting(tmp_path) -> None:
    pretrain = parse_args(["--output", str(tmp_path / "pretrain")])
    pretrain_last = tmp_path / "pretrain" / "last.pt"
    assert resolve_run_mode(pretrain, pretrain_last) == "new_pretrain"
    pretrain_last.parent.mkdir(parents=True)
    pretrain_last.touch()
    assert resolve_run_mode(pretrain, pretrain_last) == "resume_pretrain"

    init = tmp_path / "pretrain-best.pt"
    init.touch()
    finetune = parse_args(
        [
            "--stage", "finetune224", "--output", str(tmp_path / "finetune"),
            "--init-checkpoint", str(init), "--no-resume",
        ]
    )
    finetune_last = tmp_path / "finetune" / "last.pt"
    assert resolve_run_mode(finetune, finetune_last) == "init_finetune"
    finetune_last.parent.mkdir(parents=True)
    finetune_last.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        resolve_run_mode(finetune, finetune_last)

    resumed = parse_args(
        ["--stage", "finetune224", "--output", str(tmp_path / "finetune")]
    )
    assert resolve_run_mode(resumed, finetune_last) == "resume_finetune"


def test_checkpoint_cli_rejects_ambiguous_initialization() -> None:
    with pytest.raises(ValueError, match="requires --no-resume"):
        parse_args(["--stage", "finetune224", "--init-checkpoint", "pretrain.pt"])
    with pytest.raises(ValueError, match="only valid for refinement"):
        parse_args(["--stage", "pretrain", "--init-checkpoint", "pretrain.pt", "--no-resume"])


def test_checkpoint_metadata_preserves_gain_reference_contract() -> None:
    pretrain = parse_args([])
    metadata = checkpoint_metadata(pretrain, gain_reference_origin="pretrain_zero")
    checkpoint = {"run_metadata": metadata, "rrlsso_gain_reference": {"x": torch.zeros(1)}}
    assert validate_resume_checkpoint(checkpoint, pretrain) == metadata

    bad = {**checkpoint, "rrlsso_gain_reference": None}
    with pytest.raises(RuntimeError, match="persistent gain reference"):
        validate_resume_checkpoint(bad, pretrain)

    finetune = parse_args(["--stage", "finetune224"])
    validate_initialization_checkpoint({"model": {}, "run_metadata": metadata}, finetune)
    wrong_stage = {**metadata, "training_stage": "finetune224"}
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validate_initialization_checkpoint(
            {"model": {}, "run_metadata": wrong_stage}, finetune
        )
