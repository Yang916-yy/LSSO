from __future__ import annotations

import sys
import runpy
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.data import TensorDataset

from experiments.imagenet import (
    DEFAULT_CONFIG,
    RepeatedAugmentationSampler,
    _recipe_fidelity,
    build_optimizer,
    build_scheduler,
    checkpoint_contract_digest,
    interpolate_position_embedding,
    load_finetune_checkpoint,
    load_run,
    parse_args,
)


pytestmark = pytest.mark.experiment


def test_direct_imagenet_launcher_imports_repository_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    original_path = sys.path.copy()
    try:
        namespace = runpy.run_path(
            str(root / "experiments" / "train_imagenet.py"),
            run_name="lsso_direct_launcher_test",
        )
    finally:
        sys.path[:] = original_path
    assert callable(namespace["main"])


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--tier",
            "small",
            "--data-root",
            str(tmp_path / "imagenet"),
            "--output",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def test_small_recipe_preserves_deit3_geometry_and_800_epoch_contract(tmp_path: Path) -> None:
    run = load_run(_args(tmp_path))
    assert run.model == {
        "image_size": 224,
        "patch_size": 16,
        "num_classes": 1000,
        "mlp_ratio": 4.0,
        "layer_scale_init_value": 1.0e-4,
        "norm_eps": 1.0e-6,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "rank": 32,
        "drop_path_rate": 0.05,
    }
    assert (run.train["epochs"], run.train["optimizer"], run.train["augmentation"]) == (
        800,
        "fusedlamb",
        "three_augment",
    )
    assert run.train["repeated_aug"] and run.train["bce_loss"]
    assert run.operator == {
        "core_mode": "dynamic",
        "rank_rotary": True,
        "bias": True,
        "implementation": "cuda",
    }


@pytest.mark.parametrize(
    ("tier", "expected"),
    (
        ("base", (768, 12, 12, 48, 192, 0.20)),
        ("large", (1024, 24, 16, 64, 192, 0.45)),
    ),
)
def test_base_and_large_pretraining_recipes(tmp_path: Path, tier: str, expected: tuple[int, ...]) -> None:
    args = _args(tmp_path, "--tier", tier)
    run = load_run(args)
    assert (
        run.model["embed_dim"],
        run.model["depth"],
        run.model["num_heads"],
        run.model["rank"],
        run.model["image_size"],
        run.model["drop_path_rate"],
    ) == expected
    assert run.train["optimizer"] == "fusedlamb"


@pytest.mark.parametrize("tier", ("base", "large"))
def test_224_finetuning_requires_a_pretraining_checkpoint(tmp_path: Path, tier: str) -> None:
    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        load_run(_args(tmp_path, "--tier", tier, "--phase", "finetune_224"))

    run = load_run(
        _args(
            tmp_path,
            "--tier",
            tier,
            "--phase",
            "finetune_224",
            "--init-checkpoint",
            str(tmp_path / "pretrain.pt"),
        )
    )
    assert (run.model["image_size"], run.train["epochs"], run.train["optimizer"]) == (
        224,
        20,
        "adamw",
    )
    assert not run.train["repeated_aug"]
    assert not run.train["bce_loss"]


def test_repeated_augmentation_sampler_matches_deit_selected_length() -> None:
    dataset = TensorDataset(torch.arange(512))
    sampler = RepeatedAugmentationSampler(
        dataset,
        num_replicas=2,
        rank=0,
        num_repeats=3,
    )
    sampler.set_epoch(7)
    first = list(sampler)
    sampler.set_epoch(7)
    assert first == list(sampler)
    assert len(first) == 256


def test_finetune_position_interpolation_keeps_a_no_cls_patch_table() -> None:
    source = torch.arange(1 * 9 * 4, dtype=torch.float32).reshape(1, 9, 4)
    target = torch.empty(1, 16, 4)
    resized = interpolate_position_embedding(source, target)
    assert resized.shape == target.shape
    assert torch.isfinite(resized).all()


def test_finetune_loader_interpolates_the_shared_encoder_position_key(tmp_path: Path) -> None:
    class Encoder(torch.nn.Module):
        def __init__(self, tokens: int) -> None:
            super().__init__()
            self.pos_embed = torch.nn.Parameter(torch.randn(1, tokens, 4))

    class Model(torch.nn.Module):
        def __init__(self, tokens: int) -> None:
            super().__init__()
            self.encoder = Encoder(tokens)

    source = Model(9)
    target = Model(16)
    checkpoint = tmp_path / "pretrain.pt"
    run = load_run(_args(tmp_path))
    contract = run.checkpoint_contract()
    torch.save(
        {
            "format_version": 2,
            "contract": contract,
            "contract_digest": checkpoint_contract_digest(contract),
            "model": source.state_dict(),
        },
        checkpoint,
    )
    load_finetune_checkpoint(target, checkpoint, run)
    assert target.encoder.pos_embed.shape == (1, 16, 4)


def test_finetune_loader_rejects_a_checkpoint_without_current_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrain.pt"
    torch.save({"model": {}}, checkpoint)
    with pytest.raises(ValueError, match="current ImageNet contract"):
        load_finetune_checkpoint(torch.nn.Linear(2, 2), checkpoint, load_run(_args(tmp_path)))


def test_scheduler_uses_epoch_zero_warmup_then_advances_to_epoch_one(tmp_path: Path) -> None:
    pytest.importorskip("timm")
    run = load_run(_args(tmp_path))
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=float(run.train["lr"]))
    scheduler = build_scheduler(optimizer, run)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)

    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        1.0e-6 + (0.004 - 1.0e-6) / 5
    )


def test_explicit_lamb_fallback_permission_is_never_canonical(tmp_path: Path) -> None:
    run = load_run(_args(tmp_path))
    assert _recipe_fidelity(
        run,
        resolved_optimizer="apex.fused_lamb",
        allow_lamb_fallback=False,
    ) == "deit3-derived"
    assert _recipe_fidelity(
        run,
        resolved_optimizer="apex.fused_lamb",
        allow_lamb_fallback=True,
    ) == "explicitly-modified"


def test_fused_lamb_uses_the_fixed_deit3_epsilon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("timm")
    captured: dict[str, object] = {}

    class FusedLAMB:
        def __init__(self, _groups: object, **kwargs: object) -> None:
            captured.update(kwargs)

    apex = ModuleType("apex")
    optimizers = ModuleType("apex.optimizers")
    optimizers.FusedLAMB = FusedLAMB  # type: ignore[attr-defined]
    apex.optimizers = optimizers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "apex", apex)
    monkeypatch.setitem(sys.modules, "apex.optimizers", optimizers)

    build_optimizer(
        torch.nn.Linear(4, 2),
        load_run(_args(tmp_path)),
        allow_lamb_fallback=False,
    )
    assert captured["eps"] == 1.0e-8
