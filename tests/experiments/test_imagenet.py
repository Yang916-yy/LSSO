from __future__ import annotations

import copy
import sys
import runpy
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import experiments.imagenet as imagenet
from experiments.imagenet import (
    BatchingPlan,
    DEFAULT_CONFIG,
    DistributedState,
    ImageNetRun,
    VirtualGroupSampler,
    _checkpoint,
    _load_resume,
    _recipe_fidelity,
    apply_virtual_group_mixup,
    build_optimizer,
    build_scheduler,
    checkpoint_contract_digest,
    interpolate_position_embedding,
    load_finetune_checkpoint,
    load_run,
    parse_args,
    resolve_batching_plan,
    train_epoch,
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


def _cpu_state(*, world_size: int = 1, rank: int = 0) -> DistributedState:
    return DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        device=torch.device("cpu"),
    )


def _checkpoint_batching_plan() -> BatchingPlan:
    return BatchingPlan(
        world_size=1,
        physical_batch_size=256,
        effective_batch_size=2048,
        augmentation_group_size=256,
        grad_accum=8,
        samples_per_epoch=1_280_000,
        updates_per_epoch=625,
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
    assert (
        run.train["batch_size"],
        run.train["effective_batch"],
        run.train["augmentation_group_size"],
    ) == (256, 2048, 256)
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


def test_batching_plan_preserves_the_effective_deit3_update(tmp_path: Path) -> None:
    dataset_size = 1_281_167
    small_single = resolve_batching_plan(
        load_run(_args(tmp_path, "--batch-size", "512")),
        _cpu_state(),
        dataset_size=dataset_size,
        requested_grad_accum=None,
    )
    assert small_single.as_dict() == {
        "world_size": 1,
        "physical_batch_size": 512,
        "effective_batch_size": 2048,
        "augmentation_group_size": 256,
        "grad_accum": 4,
        "samples_per_epoch": 1_280_000,
        "updates_per_epoch": 625,
    }

    small_eight_gpu = resolve_batching_plan(
        load_run(_args(tmp_path)),
        _cpu_state(world_size=8),
        dataset_size=dataset_size,
        requested_grad_accum=None,
    )
    assert (small_eight_gpu.physical_batch_size, small_eight_gpu.grad_accum) == (256, 1)

    large_single = resolve_batching_plan(
        load_run(_args(tmp_path, "--tier", "large", "--batch-size", "128")),
        _cpu_state(),
        dataset_size=dataset_size,
        requested_grad_accum=None,
    )
    assert (
        large_single.physical_batch_size,
        large_single.augmentation_group_size,
        large_single.grad_accum,
        large_single.updates_per_epoch,
    ) == (128, 64, 16, 625)

    finetune = resolve_batching_plan(
        load_run(
            _args(
                tmp_path,
                "--tier",
                "base",
                "--phase",
                "finetune_224",
                "--init-checkpoint",
                str(tmp_path / "pretrain.pt"),
            )
        ),
        _cpu_state(),
        dataset_size=dataset_size,
        requested_grad_accum=None,
    )
    assert (
        finetune.effective_batch_size,
        finetune.augmentation_group_size,
        finetune.grad_accum,
        finetune.updates_per_epoch,
    ) == (512, 64, 8, 2502)


def test_batching_plan_rejects_non_equivalent_physical_schedules(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="physical batch_size"):
        resolve_batching_plan(
            load_run(_args(tmp_path, "--batch-size", "128")),
            _cpu_state(),
            dataset_size=1_281_167,
            requested_grad_accum=None,
        )
    with pytest.raises(ValueError, match="must equal effective_batch"):
        resolve_batching_plan(
            load_run(_args(tmp_path, "--batch-size", "512", "--grad-accum", "3")),
            _cpu_state(),
            dataset_size=1_281_167,
            requested_grad_accum=3,
        )


def test_virtual_group_sampler_replays_whole_groups_without_duplicates() -> None:
    dataset = TensorDataset(torch.arange(48))
    sampler = VirtualGroupSampler(
        dataset,
        num_replicas=2,
        rank=0,
        samples_per_rank=12,
        group_size=4,
        num_repeats=3,
    )
    sampler.set_epoch(7)
    first = list(sampler)
    sampler.set_epoch(7)
    assert first == list(sampler)
    assert len(first) == 12

    peer = VirtualGroupSampler(
        dataset,
        num_replicas=2,
        rank=1,
        samples_per_rank=12,
        group_size=4,
        num_repeats=3,
    )
    peer.set_epoch(7)
    local_groups = [tuple(first[offset : offset + 4]) for offset in range(0, 12, 4)]
    peer_values = list(peer)
    peer_groups = [
        tuple(peer_values[offset : offset + 4]) for offset in range(0, 12, 4)
    ]
    assert all(len(set(group)) == 4 for group in (*local_groups, *peer_groups))

    global_groups = [
        group
        for rank_groups in zip(local_groups, peer_groups, strict=True)
        for group in rank_groups
    ]
    assert sorted(Counter(global_groups).values()) == [3, 3]


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
    contract = run.checkpoint_contract(_checkpoint_batching_plan())
    torch.save(
        {
            "format_version": 3,
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
        batching_plan=_checkpoint_batching_plan(),
        resolved_optimizer="apex.fused_lamb",
        allow_lamb_fallback=False,
    ) == "deit3-derived"
    assert _recipe_fidelity(
        run,
        batching_plan=_checkpoint_batching_plan(),
        resolved_optimizer="apex.fused_lamb",
        allow_lamb_fallback=True,
    ) == "explicitly-modified"

    large_physical_run = load_run(_args(tmp_path, "--batch-size", "512"))
    large_physical_plan = resolve_batching_plan(
        large_physical_run,
        _cpu_state(),
        dataset_size=1_281_167,
        requested_grad_accum=None,
    )
    assert _recipe_fidelity(
        large_physical_run,
        batching_plan=large_physical_plan,
        resolved_optimizer="apex.fused_lamb",
        allow_lamb_fallback=False,
    ) == "deit3-derived"
    assert run.checkpoint_contract_digest(_checkpoint_batching_plan()) != (
        large_physical_run.checkpoint_contract_digest(large_physical_plan)
    )


def test_virtual_group_mixup_never_crosses_group_boundaries() -> None:
    class RecordingMixup:
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def __call__(
            self,
            images: torch.Tensor,
            targets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append(targets.tolist())
            images.add_(1.0)
            return images, targets.to(dtype=torch.float32).unsqueeze(1)

    mixup = RecordingMixup()
    images = torch.zeros(8, 1)
    targets = torch.arange(8)
    mixed_images, mixed_targets = apply_virtual_group_mixup(
        images,
        targets,
        mixup,
        group_size=4,
    )
    assert mixup.calls == [[0, 1, 2, 3], [4, 5, 6, 7]]
    torch.testing.assert_close(mixed_images, torch.ones_like(images))
    torch.testing.assert_close(mixed_targets[:, 0], targets.to(torch.float32))
    with pytest.raises(ValueError, match="divisible"):
        apply_virtual_group_mixup(torch.zeros(7, 1), torch.arange(7), mixup, 4)


def test_gradient_accumulation_matches_one_effective_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingScaler:
        def __init__(self) -> None:
            self.steps = 0
            self.updates = 0

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            self.steps += 1
            optimizer.step()

        def update(self) -> None:
            self.updates += 1

    class RecordingEma:
        def __init__(self) -> None:
            self.updates = 0

        def update(self, _model: torch.nn.Module) -> None:
            self.updates += 1

    features = torch.tensor(
        ((1.0, -1.0), (0.5, 1.0), (-1.0, 0.25), (1.5, 0.5))
    )
    targets = torch.tensor((0, 1, 1, 0))
    dataset = TensorDataset(features, targets)
    model = torch.nn.Linear(2, 2, bias=False)
    reference = copy.deepcopy(model)
    criterion = torch.nn.CrossEntropyLoss()

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    criterion(reference(features), targets).backward()
    reference_optimizer.step()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = RecordingScaler()
    ema = RecordingEma()
    run = ImageNetRun(
        config_path=tmp_path / "test.toml",
        tier="test",
        phase="test",
        model={},
        operator={},
        train={"bce_loss": False},
        overrides=(),
    )
    batching_plan = BatchingPlan(
        world_size=1,
        physical_batch_size=2,
        effective_batch_size=4,
        augmentation_group_size=2,
        grad_accum=2,
        samples_per_epoch=4,
        updates_per_epoch=1,
    )
    monkeypatch.setattr(imagenet, "_autocast", lambda: nullcontext())
    train_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        SequentialSampler(dataset),
        criterion,
        optimizer,
        scaler,  # type: ignore[arg-type]
        None,
        ema,
        epoch=0,
        state=_cpu_state(),
        run=run,
        batching_plan=batching_plan,
        print_freq=0,
    )
    torch.testing.assert_close(model.weight, reference.weight)
    assert (scaler.steps, scaler.updates, ema.updates) == (1, 1, 1)


def test_checkpoint_round_trip_uses_timm_model_ema(
    tmp_path: Path,
) -> None:
    pytest.importorskip("timm")
    from timm.utils import ModelEma

    torch.manual_seed(3)
    source = torch.nn.Linear(2, 2)
    source_ema = ModelEma(source)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=3)
    source_optimizer.step()
    source_scheduler.step()
    source_scaler = torch.amp.GradScaler("cpu")
    run = load_run(_args(tmp_path))
    batching_plan = _checkpoint_batching_plan()
    payload = _checkpoint(
        epoch=4,
        model=source,
        ema=source_ema,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        scaler=source_scaler,
        run=run,
        batching_plan=batching_plan,
        best_acc1=73.5,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    target = torch.nn.Linear(2, 2)
    target_ema = ModelEma(target)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=3)
    target_scaler = torch.amp.GradScaler("cpu")
    start_epoch, best_acc1 = _load_resume(
        path,
        model=target,
        ema=target_ema,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        scaler=target_scaler,
        run=run,
        batching_plan=batching_plan,
    )
    assert (start_epoch, best_acc1) == (5, 73.5)
    for source_parameter, target_parameter in zip(
        source.parameters(),
        target.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
    for source_parameter, target_parameter in zip(
        source_ema.ema.parameters(),
        target_ema.ema.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
    assert target_scheduler.state_dict() == source_scheduler.state_dict()
    assert target_scaler.state_dict() == source_scaler.state_dict()

    changed_physical_plan = BatchingPlan(
        world_size=1,
        physical_batch_size=512,
        effective_batch_size=2048,
        augmentation_group_size=256,
        grad_accum=4,
        samples_per_epoch=1_280_000,
        updates_per_epoch=625,
    )
    with pytest.raises(ValueError, match="does not match"):
        _load_resume(
            path,
            model=target,
            ema=target_ema,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            scaler=target_scaler,
            run=run,
            batching_plan=changed_physical_plan,
        )


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
