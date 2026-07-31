from __future__ import annotations

import copy
from pathlib import Path
import sys
from threading import Event, Thread
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lsso import CoreMode, LSSO, LSSOConfig
from lsso.ball import cuda
from lsso.ball.reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    qr_soft_frame,
    rank_rotary,
)


pytestmark = pytest.mark.cuda


def _require_native_cuda() -> None:
    try:
        cuda.load()
    except RuntimeError as error:
        pytest.skip(f"native CUDA extension is unavailable: {error}")


def _reference_fast_mix(
    projected: torch.Tensor,
    core_base_raw: torch.Tensor,
    core_drive_weight: torch.Tensor,
    eta_raw: torch.Tensor,
    centered_positions: torch.Tensor | None,
    valid_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose the FP64 reference oracle for the strict CUDA ABI."""

    calc_dtype = torch.float64
    projected = projected.to(dtype=calc_dtype)
    core_base_raw = core_base_raw.to(dtype=calc_dtype)
    core_drive_weight = core_drive_weight.to(dtype=calc_dtype)
    eta_raw = eta_raw.to(dtype=calc_dtype)
    batch, length, width = projected.shape
    heads = core_base_raw.shape[0]
    rank = core_base_raw.shape[-1]
    dim = width - heads * rank
    head_dim = dim // heads
    if valid_counts is None:
        counts = torch.full(
            (batch,),
            float(length),
            device=projected.device,
            dtype=calc_dtype,
        )
    else:
        counts = valid_counts.to(device=projected.device, dtype=calc_dtype)
    if centered_positions is None:
        positions = torch.arange(length, device=projected.device, dtype=calc_dtype)
        positions = positions - 0.5 * (length - 1)
    else:
        positions = centered_positions.to(dtype=calc_dtype)

    relation, content = projected.split((heads * rank, dim), dim=-1)
    relation = relation.view(batch, length, heads, rank).transpose(1, 2)
    content = content.view(batch, length, heads, head_dim).transpose(1, 2)
    relation = rank_rotary(
        relation,
        positions.expand(batch, -1),
    ) / counts.sqrt().view(batch, 1, 1, 1)
    frame = qr_soft_frame(relation)
    compact_state = frame.mT @ content
    coordinates = core_base_raw.unsqueeze(0) + (
        compact_state @ core_drive_weight / counts.sqrt().view(batch, 1, 1, 1)
    )
    return accretive_equilibrium_mix(
        frame,
        accretive_generator(coordinates),
        compact_state,
        content,
        bounded_complement(eta_raw),
    ).transpose(1, 2).contiguous().view(batch, length, dim)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.to(torch.float64) - expected.to(torch.float64)).norm()
    denominator = expected.to(torch.float64).norm().clamp_min(1e-12)
    return float((difference / denominator).detach())


def _assert_tc16_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    limit: float,
) -> None:
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    assert _relative_l2(actual, expected) <= limit


def test_cuda_boundary_is_explicit() -> None:
    if cuda.is_available():
        cuda.require_available()
    else:
        with pytest.raises(RuntimeError, match="extension is not loaded"):
            cuda.require_available()


def test_fast_mix_rejects_position_gradients_before_loading() -> None:
    positions = torch.arange(4, dtype=torch.float32, requires_grad=True)
    with pytest.raises(ValueError, match="does not support gradients"):
        cuda.fast_mix(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            positions,
        )


def test_fast_mix_selects_the_strict_train_and_inference_abis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def inference(*arguments: torch.Tensor | None) -> torch.Tensor:
        calls.append("forward_inference")
        projected = arguments[0]
        assert isinstance(projected, torch.Tensor)
        assert projected.dtype == torch.float32
        return projected.detach().clone()

    def train(*arguments: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append("forward_train")
        projected = arguments[0]
        assert isinstance(projected, torch.Tensor)
        assert projected.dtype == torch.float32
        return (
            projected.clone(),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.int32),
        )

    def backward(*arguments: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        calls.append("backward")
        assert len(arguments) == 9
        assert isinstance(arguments[0], torch.Tensor)
        assert arguments[0].dtype == torch.float32
        tape, pivots, positions, counts = arguments[5:]
        assert isinstance(tape, torch.Tensor) and tape.dtype == torch.float32
        assert isinstance(pivots, torch.Tensor) and pivots.dtype == torch.int32
        assert isinstance(positions, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        torch.testing.assert_close(positions, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(counts, torch.tensor([2.0]))
        return tuple(torch.ones_like(argument) for argument in arguments[1:5])

    namespace = SimpleNamespace(
        contract_version=lambda: cuda._NATIVE_CONTRACT_VERSION,
        forward_inference=inference,
        forward_train=train,
        backward=backward,
    )
    monkeypatch.setattr(
        cuda,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(lsso_equilibrium=namespace),
            is_grad_enabled=torch.is_grad_enabled,
        ),
    )
    monkeypatch.setattr(cuda, "require_available", lambda: None)

    projected = torch.randn(1, 2, 3, dtype=torch.float32, requires_grad=True)
    base = torch.randn(1, 16, 16, requires_grad=True)
    drive = torch.randn(1, 16, 16, requires_grad=True)
    eta = torch.randn(1, requires_grad=True)
    positions = torch.tensor([0.0, 1.0])
    counts = torch.tensor([2.0])
    output = cuda.fast_mix(projected, base, drive, eta, positions, counts)
    output.sum().backward()
    assert calls == ["forward_train", "backward"]

    calls.clear()
    with torch.inference_mode():
        inference_output = cuda.fast_mix(projected, base, drive, eta, positions, counts)
    assert not inference_output.requires_grad
    assert calls == ["forward_inference"]


def test_cuda_probe_uses_equilibrium_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = SimpleNamespace(
        contract_version=lambda: cuda._NATIVE_CONTRACT_VERSION,
        forward_inference=object(),
        forward_train=object(),
        backward=object(),
    )
    monkeypatch.setattr(
        cuda,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(lsso_equilibrium=namespace)),
    )
    assert cuda.is_available()


def test_cuda_probe_rejects_an_incomplete_strict_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = SimpleNamespace(forward_train=object(), backward=object())
    monkeypatch.setattr(
        cuda,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(lsso_equilibrium=namespace)),
    )
    assert not cuda.is_available()


def test_cuda_probe_rejects_a_stale_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = SimpleNamespace(
        contract_version=lambda: cuda._NATIVE_CONTRACT_VERSION - 1,
        forward_inference=object(),
        forward_train=object(),
        backward=object(),
    )
    monkeypatch.setattr(
        cuda,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(lsso_equilibrium=namespace)),
    )
    monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", None)
    assert not cuda.is_available()
    with pytest.raises(RuntimeError, match="native contract version"):
        cuda.require_available()


def test_cuda_loader_rejects_a_stale_same_name_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "lsso_equilibrium_sm80.so"
    library.touch()
    namespace = SimpleNamespace()

    def load_library(path: str) -> None:
        assert path == str(library)
        namespace.contract_version = lambda: cuda._NATIVE_CONTRACT_VERSION - 1
        namespace.forward_inference = object()
        namespace.forward_train = object()
        namespace.backward = object()

    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", None)
    monkeypatch.setattr(
        cuda,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                lsso_equilibrium=namespace,
                load_library=load_library,
            )
        ),
    )
    with pytest.raises(RuntimeError, match="native contract version"):
        cuda.load(path=library)


def test_cuda_require_available_caches_a_verified_explicit_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", 80)
    monkeypatch.setattr(
        cuda,
        "is_available",
        lambda: pytest.fail("the cached fast path must not query the dispatcher"),
    )
    cuda.require_available()


def test_default_cuda_library_path_is_arch_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LSSO_CUDA_LIBRARY", raising=False)
    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    assert cuda._default_library_path().name == "lsso_equilibrium_sm80.so"


def test_default_cuda_library_path_discovers_a_matching_runtime_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lsso import __version__

    library = tmp_path / "lsso_equilibrium_sm80.so"
    library.touch()
    runtime = ModuleType("lsso_cuda_runtime")
    runtime.LSSO_VERSION = __version__
    runtime.NATIVE_CONTRACT_VERSION = cuda._NATIVE_CONTRACT_VERSION
    runtime.TORCH_VERSION = torch.__version__
    runtime.CUDA_VERSION = torch.version.cuda or ""
    runtime.CXX11_ABI = int(torch.compiled_with_cxx11_abi())
    runtime.ARCHITECTURES = (80,)
    runtime.library_path = lambda architecture: library
    monkeypatch.setitem(sys.modules, "lsso_cuda_runtime", runtime)
    monkeypatch.delenv("LSSO_CUDA_LIBRARY", raising=False)
    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    monkeypatch.setattr(
        cuda,
        "_development_library_path",
        lambda architecture: tmp_path / "development" / f"sm{architecture}.so",
    )

    assert cuda._default_library_path() == library


def test_runtime_wheel_rejects_a_mismatched_torch_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType("lsso_cuda_runtime")
    runtime.LSSO_VERSION = "0.6.1"
    runtime.NATIVE_CONTRACT_VERSION = cuda._NATIVE_CONTRACT_VERSION
    runtime.TORCH_VERSION = "not-the-installed-torch"
    runtime.CUDA_VERSION = torch.version.cuda or ""
    runtime.CXX11_ABI = int(torch.compiled_with_cxx11_abi())
    runtime.ARCHITECTURES = (80,)
    runtime.library_path = lambda architecture: tmp_path / "lsso_equilibrium_sm80.so"
    monkeypatch.setitem(sys.modules, "lsso_cuda_runtime", runtime)
    monkeypatch.delenv("LSSO_CUDA_LIBRARY", raising=False)
    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    monkeypatch.setattr(
        cuda,
        "_development_library_path",
        lambda architecture: tmp_path / "development" / f"sm{architecture}.so",
    )

    with pytest.raises(RuntimeError, match="TORCH_VERSION"):
        cuda._default_library_path()


def test_cuda_architecture_normalizes_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "get_device_capability",
        lambda device: (12, 1),
    )
    assert cuda._device_architecture() == 120


def test_cuda_architecture_accepts_turing_sm75(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "get_device_capability",
        lambda device: (7, 5),
    )
    assert cuda._device_architecture() == 75


def test_cuda_loader_rejects_a_different_loaded_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", 89)
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="targets SM89, not requested SM80"):
        cuda.load()


def test_cuda_loader_serializes_a_mixed_architecture_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second load must observe the first architecture after taking the lock."""

    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 89)
    monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", None)
    monkeypatch.setattr(cuda, "is_available", lambda: True)

    started = Event()
    finished = Event()
    errors: list[RuntimeError] = []

    def request_second_architecture() -> None:
        started.set()
        try:
            cuda.load(device=89)
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    # This emulates the period in which another thread has registered SM80
    # but has not yet released the loader lock. The old outer availability
    # probe returned successfully here before observing that architecture.
    with cuda._LOAD_LOCK:
        thread = Thread(target=request_second_architecture)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.1)
        monkeypatch.setattr(cuda, "_LOADED_ARCHITECTURE", 80)

    thread.join(timeout=1.0)
    assert finished.is_set()
    assert len(errors) == 1
    assert "targets SM80, not requested SM89" in str(errors[0])


def test_cuda_loader_rejects_a_misnamed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "lsso_equilibrium.so"
    library.touch()
    monkeypatch.setattr(cuda, "_device_architecture", lambda device=None: 80)
    monkeypatch.setattr(cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires lsso_equilibrium_sm80.so"):
        cuda.load(path=library)


def test_reference_is_default_and_cuda_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16)).eval()
    x = torch.randn(2, 5, 32)

    def fail_fast_mix(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("reference dispatch must not call the CUDA fast path")

    monkeypatch.setattr(cuda, "fast_mix", fail_fast_mix)
    assert layer(x).shape == x.shape
    with pytest.raises(ValueError, match="'reference' or 'cuda'"):
        layer(x, implementation="automatic")


@pytest.mark.parametrize(
    "config",
    [
        LSSOConfig(32, 2, rank=16, core_mode=CoreMode.STATIC),
        LSSOConfig(32, 2, rank=16, rank_rotary=False),
        LSSOConfig(32, 2, rank=24),
    ],
)
def test_cuda_dispatch_rejects_unsupported_contracts(config: LSSOConfig) -> None:
    layer = LSSO(config)
    x = torch.randn(1, 4, config.dim)
    with pytest.raises(ValueError, match="implementation='cuda'"):
        layer(x, implementation="cuda")


@pytest.mark.parametrize(
    ("rank", "head_dim"),
    [(16, 1), (32, 17), (48, 256), (64, 384)],
)
def test_cuda_dispatch_accepts_supported_rank_and_generic_head_dimensions(
    rank: int,
    head_dim: int,
) -> None:
    config = LSSOConfig(dim=2 * head_dim, num_heads=2, rank=rank)
    layer = LSSO(config)
    x = torch.randn(1, 4, config.dim)

    # A CPU input reaches the device check, proving that the Python contract
    # accepts each compiled rank and has no head-dimension whitelist.
    with pytest.raises(ValueError, match="CUDA tensor"):
        layer(x, implementation="cuda")


def test_cuda_dispatch_accepts_mask_and_batch_specific_positions() -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16))
    x = torch.randn(2, 4, 32)
    with pytest.raises(ValueError, match="CUDA tensor"):
        layer(
            x,
            valid_mask=torch.ones(2, 4, dtype=torch.bool),
            position_ids=torch.arange(4).expand(2, -1),
            implementation="cuda",
        )


def test_cuda_dispatch_requires_a_cuda_input() -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16))
    with pytest.raises(ValueError, match="CUDA tensor"):
        layer(torch.randn(1, 4, 32), implementation="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.float32))
def test_cuda_dispatch_passes_the_strict_fast_path_abi(
    monkeypatch: pytest.MonkeyPatch,
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    layer = LSSO(
        LSSOConfig(dim=32, num_heads=2, rank=16, bias=True)
    ).cuda().eval()
    x = torch.randn(2, 5, 32, device="cuda", dtype=input_dtype)
    positions = 3 * torch.arange(5, device="cuda", dtype=torch.int64) + 11
    captured: dict[str, torch.Tensor | None] = {}

    def fake_fast_mix(
        projected: torch.Tensor,
        core_base_raw: torch.Tensor,
        core_drive_weight: torch.Tensor,
        eta_raw: torch.Tensor,
        centered_positions: torch.Tensor | None,
        valid_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        captured["projected"] = projected
        captured["core_base_raw"] = core_base_raw
        captured["core_drive_weight"] = core_drive_weight
        captured["eta_raw"] = eta_raw
        captured["centered_positions"] = centered_positions
        captured["valid_counts"] = valid_counts
        return torch.zeros(
            projected.shape[0],
            projected.shape[1],
            layer.config.dim,
            device=projected.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(cuda, "fast_mix", fake_fast_mix)
    actual = layer(x, position_ids=positions, implementation="cuda")

    projected = captured["projected"]
    assert isinstance(projected, torch.Tensor)
    assert projected.shape == (2, 5, 2 * 16 + 32)
    assert projected.dtype is torch.float32
    assert projected.is_contiguous()
    assert captured["core_base_raw"] is layer.core_base_raw
    assert captured["core_drive_weight"] is layer.core_drive_weight
    assert captured["eta_raw"] is layer.eta_raw
    centered = captured["centered_positions"]
    assert isinstance(centered, torch.Tensor)
    expected_positions = LSSO._center_positions(
        positions,
        torch.ones(2, 5, dtype=torch.bool, device="cuda"),
        dtype=torch.float32,
        all_valid=True,
    )[0]
    torch.testing.assert_close(centered, expected_positions)
    assert captured["valid_counts"] is None
    assert layer.w_o.bias is not None
    expected = layer.w_o.bias.to(dtype=input_dtype).view(1, 1, -1).expand_as(actual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.float32))
def test_cuda_dispatch_packs_masked_batch_specific_inputs(
    monkeypatch: pytest.MonkeyPatch,
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(5)
    layer = LSSO(
        LSSOConfig(dim=32, num_heads=2, rank=16, bias=True)
    ).cuda().eval()
    mask = torch.tensor(
        [[True, False, True, True, False, True], [False, True, True, False, True, False]],
        device="cuda",
    )
    clean = torch.randn(2, 6, 32, device="cuda", dtype=input_dtype)
    x = torch.where(mask[:, :, None], clean, torch.full_like(clean, float("nan")))
    positions = torch.stack(
        (
            3 * torch.arange(6, device="cuda", dtype=torch.int64) + 11,
            5 * torch.arange(6, device="cuda", dtype=torch.int64) + 101,
        )
    )
    captured: dict[str, torch.Tensor | None] = {}

    def fake_fast_mix(
        projected: torch.Tensor,
        core_base_raw: torch.Tensor,
        core_drive_weight: torch.Tensor,
        eta_raw: torch.Tensor,
        centered_positions: torch.Tensor | None,
        valid_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        del core_base_raw, core_drive_weight, eta_raw
        captured["projected"] = projected
        captured["centered_positions"] = centered_positions
        captured["valid_counts"] = valid_counts
        return torch.zeros(
            projected.shape[0],
            projected.shape[1],
            layer.config.dim,
            device=projected.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(cuda, "fast_mix", fake_fast_mix)
    actual = layer(
        x,
        valid_mask=mask,
        position_ids=positions,
        implementation="cuda",
    )

    projected = captured["projected"]
    assert isinstance(projected, torch.Tensor)
    assert torch.isfinite(projected).all()
    assert torch.count_nonzero(projected[~mask]) == 0
    centered = captured["centered_positions"]
    assert isinstance(centered, torch.Tensor)
    expected_positions = LSSO._center_positions(
        positions,
        mask,
        dtype=torch.float32,
        all_valid=False,
    )
    torch.testing.assert_close(centered, expected_positions)
    counts = captured["valid_counts"]
    assert isinstance(counts, torch.Tensor)
    torch.testing.assert_close(counts, mask.sum(dim=-1).to(dtype=torch.float32))

    assert layer.w_o.bias is not None
    bias = layer.w_o.bias.to(dtype=input_dtype).view(1, 1, -1).expand_as(actual)
    expected = torch.where(mask[:, :, None], bias, torch.zeros_like(bias))
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_dispatch_rejects_bfloat16_input() -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16)).cuda()
    x = torch.randn(1, 5, 32, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="does not support x with dtype torch.bfloat16"):
        layer(x, implementation="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "parameter_name",
    ("w_bc.weight", "w_bc.bias", "w_o.weight", "w_o.bias"),
)
def test_cuda_dispatch_requires_fp32_projection_parameters(
    parameter_name: str,
) -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16, bias=True)).cuda()
    parameter = dict(layer.named_parameters())[parameter_name]
    parameter.data = parameter.data.to(dtype=torch.float16)
    x = torch.randn(1, 5, 32, device="cuda", dtype=torch.float16)

    with pytest.raises(
        TypeError,
        match=rf"requires {parameter_name} to use torch.float32",
    ):
        layer(x, implementation="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_dispatch_rejects_position_gradients() -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16)).cuda()
    x = torch.randn(1, 5, 32, device="cuda")
    positions = torch.arange(
        5,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    with pytest.raises(ValueError, match="does not support gradients"):
        layer(x, position_ids=positions, implementation="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", (31, 32, 33, 65, 511, 512, 513))
def test_native_cuda_fast_mix_matches_reference_and_gradients(length: int) -> None:
    _require_native_cuda()

    torch.manual_seed(19)
    batch, heads, dim, rank = 1, 2, 32, 16
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    positions = positions - positions.mean()
    projected_seed = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=torch.float32,
    )
    base_seed = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    drive_seed = torch.randn(
        heads,
        dim // heads,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    eta_seed = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    upstream = torch.randn(batch, length, dim, device="cuda", dtype=torch.float32)

    fast_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )

    fast = cuda.fast_mix(*fast_inputs, positions)
    assert fast.dtype == torch.float32
    reference = _reference_fast_mix(*reference_inputs, positions)

    fast_gradients = torch.autograd.grad((fast * upstream).sum(), fast_inputs)
    reference_gradients = torch.autograd.grad(
        (reference * upstream).sum(),
        reference_inputs,
    )
    torch.cuda.synchronize()

    _assert_tc16_close(fast, reference, limit=5e-3)
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        _assert_tc16_close(fast_gradient, reference_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_cuda_eta_vjp_matches_the_fp64_oracle_for_a_narrow_head() -> None:
    """Keep the FP32 complement VJP independent of TC16 readout cancellation."""

    _require_native_cuda()

    torch.manual_seed(190)
    batch, heads, length, rank, head_dim = 1, 1, 1, 16, 1
    dim = heads * head_dim
    relation = 10.0 * torch.randn(
        batch,
        length,
        heads * rank,
        device="cuda",
        dtype=torch.float32,
    )
    content = torch.randn(
        batch,
        length,
        dim,
        device="cuda",
        dtype=torch.float32,
    )
    projected_seed = torch.cat((relation, content), dim=-1)
    base_seed = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    drive_seed = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    eta_seed = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    upstream = torch.randn(batch, length, dim, device="cuda", dtype=torch.float32)

    fast_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    oracle_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    fast = cuda.fast_mix(*fast_inputs)
    oracle = _reference_fast_mix(*oracle_inputs, None)
    fast_eta_gradient = torch.autograd.grad((fast * upstream).sum(), fast_inputs)[-1]
    oracle_eta_gradient = torch.autograd.grad(
        (oracle * upstream.double()).sum(),
        oracle_inputs,
    )[-1]
    torch.cuda.synchronize()

    _assert_tc16_close(fast_eta_gradient, oracle_eta_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_cuda_complement_tail_stays_inside_the_oracle_envelope() -> None:
    """FP32 tail coordinates must retain the reference complement VJP."""

    _require_native_cuda()

    torch.manual_seed(191)
    batch, heads, length, rank, head_dim = 1, 2, 7, 16, 8
    dim = heads * head_dim
    projected_seed = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=torch.float32,
    )
    # This keeps the compact generator at its initialized point, making the
    # complement contribution large enough to expose a saturated tail VJP.
    base_seed = torch.zeros(heads, rank, rank, device="cuda", dtype=torch.float32)
    drive_seed = torch.zeros(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    )
    eta_seed = torch.tensor([-9.5, 9.5], device="cuda", dtype=torch.float32)
    upstream = torch.randn(batch, length, dim, device="cuda", dtype=torch.float32)

    fast_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    oracle_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    fast = cuda.fast_mix(*fast_inputs)
    oracle = _reference_fast_mix(*oracle_inputs, None)
    fast_eta_gradient = torch.autograd.grad((fast * upstream).sum(), fast_inputs)[-1]
    oracle_eta_gradient = torch.autograd.grad(
        (oracle * upstream.double()).sum(),
        oracle_inputs,
    )[-1]
    torch.cuda.synchronize()

    assert torch.isfinite(fast_eta_gradient).all()
    assert torch.count_nonzero(fast_eta_gradient) == heads
    _assert_tc16_close(fast_eta_gradient, oracle_eta_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("batch", "rank", "heads", "head_dim", "length"),
    [
        (1, 16, 1, 1, 31),
        (1, 16, 1, 15, 31),
        (1, 16, 1, 17, 31),
        (1, 16, 1, 31, 33),
        (1, 16, 1, 32, 33),
        (1, 16, 1, 33, 33),
        (1, 16, 1, 257, 33),
        (1, 16, 1, 512, 33),
        (1, 32, 2, 48, 31),
        (2, 32, 3, 17, 35),
        (1, 48, 2, 96, 33),
        (1, 64, 1, 33, 33),
        (1, 64, 1, 257, 33),
        (1, 16, 1, 17, 2048),
        (1, 32, 1, 17, 2048),
        (1, 48, 1, 17, 2048),
        (1, 64, 1, 17, 2048),
        # Exercise a parallel-Gram schedule with a one-token tail.
        (1, 16, 1, 17, 2017),
        (1, 64, 1, 17, 2017),
        # Keep one larger r=64 gradient case beyond the first long schedule.
        (1, 64, 1, 17, 4096),
        # LRA defaults: N=4096, D=256, H=8, r=32.
        (1, 32, 8, 32, 4096),
        # GenomicBenchmarks Mouse reaches N=4776 with the paper DNA width.
        (1, 16, 4, 32, 4776),
    ],
)
def test_native_cuda_fast_mix_expanded_shapes_match_reference_and_gradients(
    batch: int,
    rank: int,
    heads: int,
    head_dim: int,
    length: int,
) -> None:
    _require_native_cuda()

    torch.manual_seed(71 + batch + rank + head_dim)
    dim = heads * head_dim
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    positions = positions - positions.mean()
    projected_seed = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=torch.float32,
    )
    base_seed = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.05
    drive_seed = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.05
    eta_seed = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.05
    upstream = torch.randn(batch, length, dim, device="cuda", dtype=torch.float32)

    fast_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    fast = cuda.fast_mix(*fast_inputs, positions)
    reference = _reference_fast_mix(*reference_inputs, positions)
    fast_gradients = torch.autograd.grad((fast * upstream).sum(), fast_inputs)
    reference_gradients = torch.autograd.grad(
        (reference * upstream).sum(), reference_inputs
    )
    torch.cuda.synchronize()

    _assert_tc16_close(fast, reference, limit=5e-3)
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        _assert_tc16_close(fast_gradient, reference_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("rank", "head_dim", "length"),
    [
        (16, 17, 35),
        (32, 64, 35),
        (48, 96, 35),
        (64, 257, 35),
        (16, 17, 2048),
        # Exercise the parallel Gram schedule with gapped and all-masked rows.
        (64, 17, 2017),
    ],
)
def test_native_cuda_fast_mix_masked_generic_shapes_match_reference_and_gradients(
    rank: int,
    head_dim: int,
    length: int,
) -> None:
    _require_native_cuda()

    torch.manual_seed(97 + rank + head_dim)
    batch, heads = 2, 2
    dim = heads * head_dim
    mask_pattern = torch.tensor(
        [True, False, True, True, False, True, True],
        device="cuda",
    )
    first_mask = mask_pattern.repeat(
        (length + mask_pattern.numel() - 1) // mask_pattern.numel()
    )[:length]
    mask = torch.stack(
        (
            first_mask,
            torch.zeros(length, device="cuda", dtype=torch.bool),
        )
    )
    position_ids = torch.stack(
        (
            3 * torch.arange(length, device="cuda", dtype=torch.int64) + 7,
            5 * torch.arange(length, device="cuda", dtype=torch.int64) + 101,
        )
    )
    centered_positions = LSSO._center_positions(
        position_ids,
        mask,
        dtype=torch.float32,
        all_valid=False,
    )
    valid_counts = mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)
    projected_seed = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=torch.float32,
    )
    projected_seed = torch.where(
        mask[:, :, None], projected_seed, torch.zeros_like(projected_seed)
    )
    base_seed = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.05
    drive_seed = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.05
    eta_seed = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.05
    upstream = torch.randn(batch, length, dim, device="cuda", dtype=torch.float32)
    upstream = torch.where(mask[:, :, None], upstream, torch.zeros_like(upstream))

    fast_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (projected_seed, base_seed, drive_seed, eta_seed)
    )
    fast = cuda.fast_mix(
        *fast_inputs,
        centered_positions,
        valid_counts,
    )
    reference = _reference_fast_mix(
        *reference_inputs,
        centered_positions,
        valid_counts,
    )
    fast_gradients = torch.autograd.grad((fast * upstream).sum(), fast_inputs)
    reference_gradients = torch.autograd.grad(
        (reference * upstream).sum(), reference_inputs
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(fast[~mask]) == 0
    _assert_tc16_close(fast, reference, limit=5e-3)
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        _assert_tc16_close(fast_gradient, reference_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_cuda_full_lsso_masked_batch_matches_fp64_oracle() -> None:
    _require_native_cuda()

    torch.manual_seed(109)
    config = LSSOConfig(dim=34, num_heads=2, rank=16, bias=True)
    fast_layer = LSSO(config).cuda().eval()
    reference_layer = copy.deepcopy(fast_layer).double().eval()
    mask = torch.tensor(
        [
            [True, False, True, True, False, True, True] * 5,
            [False] * 35,
        ],
        device="cuda",
    )
    position_ids = torch.stack(
        (
            3 * torch.arange(35, device="cuda", dtype=torch.int64) + 7,
            5 * torch.arange(35, device="cuda", dtype=torch.int64) + 101,
        )
    )
    fast_x = torch.randn(2, 35, config.dim, device="cuda", dtype=torch.float32)
    fast_x[~mask] = float("nan")
    fast_x.requires_grad_()
    reference_x = fast_x.detach().double().requires_grad_()
    upstream = torch.randn_like(fast_x)
    fast_parameters = tuple(fast_layer.named_parameters())
    reference_parameters = tuple(reference_layer.named_parameters())
    assert tuple(name for name, _ in fast_parameters) == tuple(
        name for name, _ in reference_parameters
    )

    fast_output = fast_layer(
        fast_x,
        valid_mask=mask,
        position_ids=position_ids,
        implementation="cuda",
    )
    reference_output = reference_layer(
        reference_x,
        valid_mask=mask,
        position_ids=position_ids,
        implementation="reference",
    )
    fast_gradients = torch.autograd.grad(
        (fast_output * upstream).sum(),
        (fast_x, *(parameter for _name, parameter in fast_parameters)),
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * upstream.double()).sum(),
        (reference_x, *(parameter for _name, parameter in reference_parameters)),
    )
    torch.cuda.synchronize()

    assert torch.isfinite(fast_output).all()
    assert torch.count_nonzero(fast_output[1]) == 0
    assert torch.count_nonzero(fast_gradients[0][~mask]) == 0
    _assert_tc16_close(fast_output, reference_output, limit=5e-3)
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        _assert_tc16_close(fast_gradient, reference_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_native_cuda_fast_mix_rejects_non_fp32_packed_input(
    dtype: torch.dtype,
) -> None:
    _require_native_cuda()

    batch, length, heads, rank, head_dim = 1, 5, 1, 16, 16
    dim = heads * head_dim
    projected = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=dtype,
    )
    base = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
    )
    drive = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    )
    eta = torch.randn(heads, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="strict TC16/FP32 CUDA contract"):
        cuda.fast_mix(projected, base, drive, eta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("rank", "head_dim", "length"),
    [
        (16, 16, 33),
        (32, 64, 33),
        (64, 128, 33),
        (16, 16, 2017),
        (64, 128, 2049),
    ],
)
def test_native_cuda_train_tape_and_inference_forward_match(
    rank: int,
    head_dim: int,
    length: int,
) -> None:
    _require_native_cuda()

    torch.manual_seed(47 + rank + head_dim)
    batch, heads = 1, 2
    dim = heads * head_dim
    projected = torch.randn(
        batch,
        length,
        heads * rank + dim,
        device="cuda",
        dtype=torch.float32,
    )
    base = torch.randn(heads, rank, rank, device="cuda", dtype=torch.float32) * 0.1
    drive = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    eta = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    positions = positions - positions.mean()

    with torch.inference_mode():
        inference_output = torch.ops.lsso_equilibrium.forward_inference(
            projected,
            base,
            drive,
            eta,
            positions,
        )
    train_output, tape, pivots = torch.ops.lsso_equilibrium.forward_train(
        projected,
        base,
        drive,
        eta,
        positions,
    )
    torch.cuda.synchronize()

    assert inference_output.dtype == torch.float32
    assert train_output.dtype == torch.float32
    assert train_output.shape == inference_output.shape
    assert tape.is_cuda and tape.is_contiguous() and tape.dtype == torch.float32
    assert pivots.is_cuda and pivots.is_contiguous() and pivots.dtype == torch.int32
    assert pivots.numel() == batch * heads * rank
    torch.testing.assert_close(train_output, inference_output, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_cuda_forward_train_rejects_direct_autograd() -> None:
    _require_native_cuda()

    batch, length, heads, rank, head_dim = 1, 5, 1, 16, 16
    projected = torch.randn(
        batch,
        length,
        heads * (rank + head_dim),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    base = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    drive = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    eta = torch.randn(
        heads,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    with pytest.raises(RuntimeError, match="private tape-producing entry point"):
        torch.ops.lsso_equilibrium.forward_train(projected, base, drive, eta, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_cuda_forward_inference_rejects_direct_autograd() -> None:
    _require_native_cuda()

    batch, length, heads, rank, head_dim = 1, 5, 1, 16, 16
    projected = torch.randn(
        batch,
        length,
        heads * (rank + head_dim),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    base = torch.randn(
        heads,
        rank,
        rank,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    drive = torch.randn(
        heads,
        head_dim,
        rank,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    eta = torch.randn(
        heads,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    with pytest.raises(RuntimeError, match="inference-only entry point"):
        torch.ops.lsso_equilibrium.forward_inference(projected, base, drive, eta, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("dim", "heads", "rank"), [(32, 2, 16), (128, 1, 64)])
def test_native_cuda_full_lsso_matches_fp64_oracle(
    dim: int,
    heads: int,
    rank: int,
) -> None:
    _require_native_cuda()

    torch.manual_seed(59 + rank)
    config = LSSOConfig(dim=dim, num_heads=heads, rank=rank, bias=True)
    fast_layer = LSSO(config).cuda().eval()
    reference_layer = copy.deepcopy(fast_layer).double().eval()
    length = 33
    position_ids = 3 * torch.arange(length, device="cuda", dtype=torch.int64) + 7
    x_seed = torch.randn(1, length, config.dim, device="cuda", dtype=torch.float32)
    upstream = torch.randn_like(x_seed)
    fast_x = x_seed.detach().clone().requires_grad_()
    reference_x = x_seed.detach().double().requires_grad_()
    fast_named_parameters = tuple(fast_layer.named_parameters())
    reference_named_parameters = tuple(reference_layer.named_parameters())
    assert tuple(name for name, _parameter in fast_named_parameters) == tuple(
        name for name, _parameter in reference_named_parameters
    )

    fast_output = fast_layer(
        fast_x,
        position_ids=position_ids,
        implementation="cuda",
    )
    reference_output = reference_layer(
        reference_x,
        position_ids=position_ids,
        implementation="reference",
    )
    fast_gradients = torch.autograd.grad(
        (fast_output * upstream).sum(),
        (fast_x, *(parameter for _name, parameter in fast_named_parameters)),
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * upstream.double()).sum(),
        (
            reference_x,
            *(parameter for _name, parameter in reference_named_parameters),
        ),
    )
    torch.cuda.synchronize()

    assert fast_output.dtype == torch.float32
    _assert_tc16_close(fast_output, reference_output, limit=5e-3)
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        _assert_tc16_close(fast_gradient, reference_gradient, limit=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_float64_positions_preserve_relative_coordinates() -> None:
    _require_native_cuda()

    torch.manual_seed(113)
    config = LSSOConfig(dim=32, num_heads=2, rank=16)
    fast_layer = LSSO(config).cuda().eval()
    reference_layer = copy.deepcopy(fast_layer).double().eval()
    x = torch.randn(1, 9, config.dim, device="cuda")
    positions = torch.arange(9, device="cuda", dtype=torch.float64)
    shifted_positions = positions + 1e12

    fast_output = fast_layer(
        x,
        position_ids=shifted_positions,
        implementation="cuda",
    )
    reference_output = reference_layer(
        x.double(),
        position_ids=shifted_positions,
        implementation="reference",
    )
    unshifted_output = fast_layer(
        x,
        position_ids=positions,
        implementation="cuda",
    )

    _assert_tc16_close(fast_output, reference_output, limit=5e-3)
    torch.testing.assert_close(fast_output, unshifted_output, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("dim", "heads", "rank"), [(32, 2, 16), (128, 1, 64)])
def test_native_cuda_full_lsso_low_precision_backward_is_finite(
    dim: int,
    heads: int,
    rank: int,
) -> None:
    _require_native_cuda()

    torch.manual_seed(61 + rank)
    layer = LSSO(LSSOConfig(dim=dim, num_heads=heads, rank=rank, bias=True)).cuda()
    x = torch.randn(
        1,
        33,
        dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    output = layer(x, implementation="cuda")
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    assert output.dtype == torch.float16
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
