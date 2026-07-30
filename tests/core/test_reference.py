from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from lsso.ball.reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    qr_soft_frame,
    tensor_core_linear,
    tensor_core_matmul,
    tf32_fp32_linear,
)


pytestmark = pytest.mark.core


def test_qr_soft_frame_preserves_gram_identity() -> None:
    torch.manual_seed(0)
    relation = torch.randn(3, 9, 5, dtype=torch.float64)
    frame = qr_soft_frame(relation)

    rank_eye = torch.eye(5, dtype=torch.float64)
    expected = relation @ torch.linalg.solve(
        rank_eye + relation.mT @ relation,
        relation.mT,
    )
    torch.testing.assert_close(frame @ frame.mT, expected)
    assert torch.all(torch.linalg.matrix_norm(frame, ord=2) <= 1.0 + 2e-12)


def test_qr_soft_frame_has_normal_range_gradients() -> None:
    torch.manual_seed(1)
    relation = torch.randn(2, 11, 4, dtype=torch.float32, requires_grad=True)
    frame = qr_soft_frame(relation)
    frame.square().mean().backward()

    assert torch.isfinite(frame).all()
    assert relation.grad is not None and torch.isfinite(relation.grad).all()


def test_tensor_core_matmul_keeps_fp64_as_the_autograd_oracle() -> None:
    torch.manual_seed(9)
    left = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, 5, dtype=torch.float64)

    output = tensor_core_matmul(left, right)
    expected = left @ right
    gradients = torch.autograd.grad((output * upstream).sum(), (left, right))
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(),
        (left, right),
    )

    assert output.dtype is torch.float64
    torch.testing.assert_close(output, expected)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensor_core_matmul_cuda_has_fp32_output_and_vjp() -> None:
    torch.manual_seed(10)
    left = torch.randn(2, 3, 4, 5, device="cuda", requires_grad=True)
    right = torch.randn(3, 5, 6, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 4, 6, device="cuda")

    output = tensor_core_matmul(left, right)
    gradients = torch.autograd.grad((output * upstream).sum(), (left, right))

    left_batches = left.detach().to(dtype=torch.float16).reshape(-1, 4, 5)
    right_batches = right.detach().to(dtype=torch.float16).expand(
        2,
        -1,
        -1,
        -1,
    ).reshape(-1, 5, 6)
    expected = torch.bmm(
        left_batches,
        right_batches,
        out_dtype=torch.float32,
    ).reshape(2, 3, 4, 6)
    upstream_batches = upstream.to(dtype=torch.float16).reshape(-1, 4, 6)
    expected_left_gradient = torch.bmm(
        upstream_batches,
        right_batches.mT,
        out_dtype=torch.float32,
    ).reshape_as(left)
    expected_right_gradient = torch.bmm(
        left_batches.mT,
        upstream_batches,
        out_dtype=torch.float32,
    ).reshape(2, 3, 5, 6).sum(dim=0)

    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[0], expected_left_gradient, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[1], expected_right_gradient, rtol=0.0, atol=0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.float32))
def test_tensor_core_linear_cuda_uses_flattened_tc16_vjp(
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(12)
    value = torch.randn(
        2,
        3,
        5,
        device="cuda",
        dtype=input_dtype,
        requires_grad=True,
    )
    weight = torch.randn(7, 5, device="cuda", requires_grad=True)
    bias = torch.randn(7, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 7, device="cuda")

    actual = tensor_core_linear(value, weight, bias)
    gradients = torch.autograd.grad((actual * upstream).sum(), (value, weight, bias))

    value_tc16 = value.detach().to(dtype=torch.float16).reshape(-1, 5)
    weight_tc16 = weight.detach().to(dtype=torch.float16)
    gradient_tc16 = upstream.to(dtype=torch.float16).reshape(-1, 7)
    expected = torch.mm(
        value_tc16,
        weight_tc16.mT,
        out_dtype=torch.float32,
    ).reshape_as(actual) + bias.detach()
    expected_value_gradient = torch.mm(
        gradient_tc16,
        weight_tc16,
        out_dtype=torch.float32,
    ).reshape_as(value).to(dtype=input_dtype)
    expected_weight_gradient = torch.mm(
        gradient_tc16.mT,
        value_tc16,
        out_dtype=torch.float32,
    )
    expected_bias_gradient = upstream.sum(dim=(0, 1))

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gradients[0], expected_value_gradient, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gradients[1], expected_weight_gradient, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gradients[2], expected_bias_gradient, rtol=0.0, atol=0.0
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.float32))
def test_tf32_fp32_linear_matches_tf32_oracle_and_restores_policy(
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(101)
    value = (
        40.0 * torch.randn(2, 3, 13, device="cuda", dtype=input_dtype)
    ).requires_grad_()
    weight = (40.0 * torch.randn(11, 13, device="cuda")).requires_grad_()
    bias = (40.0 * torch.randn(11, device="cuda")).requires_grad_()
    upstream = torch.randn(2, 3, 11, device="cuda")
    oracle_value = value.detach().float().requires_grad_()
    oracle_weight = weight.detach().clone().requires_grad_()
    oracle_bias = bias.detach().clone().requires_grad_()

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    matmul.fp32_precision = "ieee"
    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=input_dtype is torch.float16,
        ):
            actual = tf32_fp32_linear(value, weight, bias)
        actual_gradients = torch.autograd.grad(
            (actual * upstream).sum(),
            (value, weight, bias),
        )
        assert matmul.fp32_precision == "ieee"
        matmul.fp32_precision = "tf32"

        with torch.autocast(device_type="cuda", enabled=False):
            expected = functional.linear(oracle_value, oracle_weight, oracle_bias)
        expected_gradients = torch.autograd.grad(
            (expected * upstream).sum(),
            (oracle_value, oracle_weight, oracle_bias),
        )
    finally:
        matmul.fp32_precision = previous

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    expected_input_gradient = expected_gradients[0].to(dtype=input_dtype)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        (expected_input_gradient, *expected_gradients[1:]),
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-6,
            atol=2e-6,
        )


def test_accretive_generator_matches_its_parameterization() -> None:
    torch.manual_seed(2)
    raw = torch.randn(2, 4, 4, dtype=torch.float64)
    generator = accretive_generator(raw)

    offset = math.log(math.expm1(1.0))
    diagonal = torch.diagonal(raw, dim1=-2, dim2=-1)
    factor = torch.tril(raw, diagonal=-1) + torch.diag_embed(
        torch.nn.functional.softplus(diagonal + offset)
    )
    upper = torch.triu(raw, diagonal=1)
    expected = factor @ factor.mT + upper - upper.mT

    torch.testing.assert_close(generator, expected)
    symmetric = 0.5 * (generator + generator.mT)
    assert torch.all(torch.linalg.eigvalsh(symmetric) > 0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_accretive_generator_cuda_forces_ieee_factor_gram_vjp() -> None:
    """The compact factor is full FP32 even when the ambient policy is TF32."""

    torch.manual_seed(22)
    raw_seed = 4.0 * torch.randn(2, 3, 16, 16, device="cuda")
    upstream = torch.randn_like(raw_seed)
    actual_raw = raw_seed.detach().clone().requires_grad_()
    expected_raw = raw_seed.detach().clone().requires_grad_()

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"
        actual = accretive_generator(actual_raw)
        actual_gradient = torch.autograd.grad((actual * upstream).sum(), actual_raw)[0]
        assert matmul.fp32_precision == "tf32"

        matmul.fp32_precision = "ieee"
        diagonal = torch.diagonal(expected_raw, dim1=-2, dim2=-1)
        factor = torch.tril(expected_raw, diagonal=-1) + torch.diag_embed(
            functional.softplus(diagonal + math.log(math.expm1(1.0)))
        )
        upper = torch.triu(expected_raw, diagonal=1)
        expected = factor @ factor.mT + upper - upper.mT
        expected_gradient = torch.autograd.grad(
            (expected * upstream).sum(), expected_raw
        )[0]
    finally:
        matmul.fp32_precision = previous

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=5e-6,
        atol=5e-6,
    )


def test_zero_raw_generator_is_identity() -> None:
    raw = torch.zeros(3, 4, 4, dtype=torch.float64)
    expected = torch.eye(4, dtype=torch.float64).expand_as(raw)
    torch.testing.assert_close(accretive_generator(raw), expected)


def test_shared_compact_state_preserves_dynamic_drive_algebra() -> None:
    torch.manual_seed(3)
    batch, heads, length, rank, head_dim = 2, 3, 7, 4, 6
    relation = torch.randn(batch, heads, length, rank, dtype=torch.float64)
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    drive_weight = torch.randn(heads, head_dim, rank, dtype=torch.float64)
    valid_count = torch.tensor([7.0, 5.0], dtype=torch.float64)
    frame = qr_soft_frame(
        relation / valid_count.sqrt().view(batch, 1, 1, 1)
    )

    compact_state = frame.mT @ content
    legacy_drive = torch.einsum("bhnd,hdr->bhnr", content, drive_weight)
    legacy = frame.mT @ legacy_drive
    shared = torch.einsum("bhrd,hdk->bhrk", compact_state, drive_weight)
    torch.testing.assert_close(shared, legacy)

    normalized_legacy = legacy / valid_count.sqrt().view(batch, 1, 1, 1)
    normalized_shared = shared / valid_count.sqrt().view(batch, 1, 1, 1)
    torch.testing.assert_close(normalized_shared, normalized_legacy)


def test_equilibrium_mix_satisfies_direct_solve_equation() -> None:
    torch.manual_seed(4)
    batch, heads, length, rank, head_dim = 2, 3, 8, 4, 5
    frame = qr_soft_frame(
        torch.randn(batch, heads, length, rank, dtype=torch.float64)
    )
    raw = torch.randn(batch, heads, rank, rank, dtype=torch.float64)
    generator = accretive_generator(raw)
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = bounded_complement(torch.tensor([-0.7, 0.2, 0.9], dtype=torch.float64))

    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        eta,
    )

    rank_eye = torch.eye(rank, dtype=torch.float64)
    equilibrium = torch.linalg.solve(generator + rank_eye, compact_state)
    torch.testing.assert_close(
        (generator + rank_eye) @ equilibrium,
        compact_state,
    )
    eta_batch = eta.view(1, heads, 1, 1)
    expected = eta_batch * content + frame @ (
        2.0 * equilibrium - (1.0 + eta_batch) * compact_state
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dynamic", [False, True])
def test_equilibrium_mix_accepts_static_and_dynamic_generators(
    dynamic: bool,
) -> None:
    torch.manual_seed(5)
    batch, heads, length, rank, head_dim = 2, 3, 7, 4, 6
    frame = qr_soft_frame(torch.randn(batch, heads, length, rank))
    static_generator = accretive_generator(torch.randn(heads, rank, rank))
    generator = (
        static_generator.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        if dynamic
        else static_generator
    )
    content = torch.randn(batch, heads, length, head_dim)
    compact_state = frame.mT @ content
    eta = bounded_complement(torch.linspace(-0.6, 0.9, heads))

    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        eta,
    )
    assert output.shape == content.shape
    assert torch.isfinite(output).all()


def test_zero_compact_core_uses_the_unscaled_complement_formula() -> None:
    torch.manual_seed(6)
    batch, heads, length, rank, head_dim = 2, 1, 7, 4, 3
    frame = qr_soft_frame(
        torch.randn(batch, heads, length, rank, dtype=torch.float64)
    )
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = torch.tensor([0.9], dtype=torch.float64)

    output = accretive_equilibrium_mix(
        frame,
        None,
        compact_state,
        content,
        eta,
    )
    expected = eta.view(1, heads, 1, 1) * (
        content - frame @ compact_state
    )
    torch.testing.assert_close(output, expected)


def test_direct_equilibrium_has_normal_range_gradients() -> None:
    torch.manual_seed(7)
    relation = torch.randn(2, 2, 8, 4, dtype=torch.float64, requires_grad=True)
    raw = torch.randn(2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    content = torch.randn(2, 2, 8, 5, dtype=torch.float64, requires_grad=True)
    eta_raw = torch.tensor([-0.4, 0.6], dtype=torch.float64, requires_grad=True)

    frame = qr_soft_frame(relation)
    generator = accretive_generator(raw)
    compact_state = frame.mT @ content
    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        bounded_complement(eta_raw),
    )
    gradients = torch.autograd.grad(
        output.square().mean(),
        (relation, raw, content, eta_raw),
    )
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert torch.count_nonzero(gradients[1]) > 0
def test_direct_equilibrium_passes_first_and_second_order_gradcheck() -> None:
    torch.manual_seed(8)
    frame = qr_soft_frame(torch.randn(1, 1, 5, 3, dtype=torch.float64))
    content = torch.randn(1, 1, 5, 2, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = torch.tensor([0.35], dtype=torch.float64)
    raw = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)

    def mix(coordinates: torch.Tensor) -> torch.Tensor:
        return accretive_equilibrium_mix(
            frame,
            accretive_generator(coordinates),
            compact_state,
            content,
            eta,
        )

    assert torch.autograd.gradcheck(
        mix,
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-3,
    )
    assert torch.autograd.gradgradcheck(
        mix,
        (raw,),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-3,
    )


def test_bounded_complement_is_a_strict_interior_tanh() -> None:
    raw = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float32)
    actual = bounded_complement(raw)
    expected = (1.0 - torch.finfo(torch.float32).eps) * raw.tanh()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert torch.all(actual.abs() < 1.0)


def test_bounded_complement_retains_the_fp32_tanh_tail_gradient() -> None:
    raw = torch.tensor([-9.5, 9.5], dtype=torch.float32, requires_grad=True)
    complement = bounded_complement(raw)
    complement.sum().backward()

    assert torch.all(complement.abs() < 1.0)
    assert raw.grad is not None
    assert torch.all(torch.isfinite(raw.grad))
    assert torch.all(raw.grad.abs() > 0.0)


def test_reference_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="matrix dimensions"):
        qr_soft_frame(torch.randn(4))
    with pytest.raises(TypeError, match="floating-point"):
        qr_soft_frame(torch.ones(4, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="square"):
        accretive_generator(torch.randn(4, 3))

    frame = torch.randn(2, 3, 7, 4)
    content = torch.randn(2, 3, 7, 5)
    compact_state = frame.mT @ content
    eta = torch.ones(3)
    with pytest.raises(ValueError, match="dynamic generator"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(1, 3, 4, 4),
            compact_state,
            content,
            eta,
        )
    with pytest.raises(ValueError, match="compact_state"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(3, 4, 4),
            compact_state[..., :-1],
            content,
            eta,
        )
    with pytest.raises(ValueError, match="eta"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(3, 4, 4),
            compact_state,
            content,
            torch.tensor(0.9),
        )
