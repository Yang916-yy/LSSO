from __future__ import annotations

import torch
import pytest

from experiments.imagenet_wds_train import (
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
    assert finetune.output.endswith("finetune224")


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
