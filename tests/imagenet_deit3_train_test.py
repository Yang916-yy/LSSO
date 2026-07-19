from __future__ import annotations

import torch
import pytest

from experiments.imagenet_wds_train import (
    apply_virtual_group_mixup,
    create_model_ema,
    create_official_optimizer,
    create_training_model,
    parse_args,
    resize_position_embedding,
    shard_commands,
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
            "--init-checkpoint", "dummy.pt",
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


def test_pretraining_never_silently_drops_fused_lamb() -> None:
    model = torch.nn.Linear(2, 2)
    args = parse_args(["--model", "deit3_base_patch16_rrlsso", "--epochs", "1"])
    try:
        import apex  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="requires NVIDIA Apex FusedLAMB"):
            create_official_optimizer(model, args)
    else:
        _, implementation = create_official_optimizer(model, args)
        assert implementation == "apex_fusedlamb"

    args.allow_unfused_lamb = True
    _, implementation = create_official_optimizer(model, args)
    assert implementation in {"apex_fusedlamb", "lamb"}
