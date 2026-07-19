from __future__ import annotations

import torch

from experiments.imagenet_wds_train import (
    create_training_model,
    parse_args,
    resize_position_embedding,
    shard_commands,
)


def test_stage_defaults_cover_low_resolution_and_refinement() -> None:
    pretrain = parse_args(["--stage", "pretrain", "--no-resume"])
    assert (pretrain.epochs, pretrain.image_size) == (800, 192)
    assert (pretrain.lr, pretrain.warmup_epochs) == (0.0, 20)

    finetune = parse_args(
        [
            "--stage",
            "finetune",
            "--init-checkpoint",
            "pretrain/best.pt",
            "--no-resume",
        ]
    )
    assert (finetune.epochs, finetune.image_size) == (20, 224)
    assert (finetune.lr, finetune.warmup_epochs) == (1e-5, 5)
    assert finetune.output.endswith("finetune224")


def test_position_embedding_is_resized_from_192_to_224() -> None:
    source_args = parse_args(
        ["--model", "deit3_small_patch16_192_rrlsso", "--image-size", "192"]
    )
    target_args = parse_args(
        ["--model", "deit3_small_patch16_192_rrlsso", "--image-size", "224"]
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
            "deit3_small_patch16_192_rrlsso",
            "--image-size",
            "32",
            "--rank",
            "8",
        ]
    )
    model = create_training_model(args)
    assert model.rrlsso_config["basis_normalization"] == "trace"
    assert model.rrlsso_config["rank_rotary"] == "ordinary-1d"
    assert model.rrlsso_config["alpha_init"] == 1.2
    assert torch.isfinite(model.pos_embed).all()
