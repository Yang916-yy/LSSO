#include <cfloat>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublasdx.hpp>
#include <cusolverdx.hpp>

// MathDx 26.x exposes launch guards used by its multi-architecture examples;
// the CUDA-12-compatible 25.12 package predates those convenience macros.
// This extension is compiled into architecture-specific kernels, so the
// older package does not need an additional runtime guard.
#ifndef CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM
#define CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(Operation)
#endif
#ifndef CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM
#define CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Operation)
#endif

namespace {

#if CUDART_VERSION >= 13000
constexpr int kThorMathDxArch = 1100;
#else
// CUDA 13 renamed Thor from SM101 to SM110. MathDx descriptors follow the
// toolkit-specific name even though runtime compute capability reports 11.0.
constexpr int kThorMathDxArch = 1010;
#endif

template <typename scalar_t, bool Inverse>
__global__ void rank_rotary_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ output,
    int64_t pair_count,
    int64_t batches,
    int64_t heads,
    int64_t sequence,
    int64_t half_rank,
    int64_t factor_batches,
    int64_t stride_b,
    int64_t stride_h,
    int64_t stride_n,
    int64_t stride_r) {
    const int64_t pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pair_count) {
        return;
    }
    int64_t index = pair;
    const int64_t rank_pair = index % half_rank;
    index /= half_rank;
    const int64_t token = index % sequence;
    index /= sequence;
    const int64_t head = index % heads;
    const int64_t batch = index / heads;
    const int64_t input_base =
        batch * stride_b + head * stride_h + token * stride_n +
        2 * rank_pair * stride_r;
    const int64_t factor =
        (factor_batches == 1 ? 0 : batch * sequence * half_rank) +
        token * half_rank + rank_pair;
    const float even = static_cast<float>(input[input_base]);
    const float odd = static_cast<float>(input[input_base + stride_r]);
    const float c = static_cast<float>(cos[factor]);
    const float s = static_cast<float>(sin[factor]);
    if constexpr (Inverse) {
        output[2 * pair] = static_cast<scalar_t>(even * c + odd * s);
        output[2 * pair + 1] = static_cast<scalar_t>(-even * s + odd * c);
    } else {
        output[2 * pair] = static_cast<scalar_t>(even * c - odd * s);
        output[2 * pair + 1] = static_cast<scalar_t>(even * s + odd * c);
    }
}

template <int Rank, int RhsTile, int Arch, int BatchesPerBlock = 1>
struct SolverTraits {
    using Base = decltype(
        cusolverdx::Size<Rank, Rank, RhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::FillMode<cusolverdx::lower>() +
        cusolverdx::Arrangement<cusolverdx::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<256 * BatchesPerBlock>() +
        cusolverdx::BatchesPerBlock<BatchesPerBlock>() +
        cusolverdx::SM<Arch>());

    using Potrf = decltype(Base() + cusolverdx::Function<cusolverdx::potrf>());
    using Potrs = decltype(Base() + cusolverdx::Function<cusolverdx::potrs>());
};

template <typename scalar_t>
struct GemmInputPrecision {
    using Type = float;
};

template <>
struct GemmInputPrecision<c10::Half> {
    using Type = half;
};

template <>
struct GemmInputPrecision<c10::BFloat16> {
    using Type = __nv_bfloat16;
};

template <typename scalar_t, int Rank, int Columns, int KTile, int Arch>
struct GemmTraits {
    using Input = typename GemmInputPrecision<scalar_t>::Type;
    using Type = decltype(
        cublasdx::Size<Rank, Columns, KTile>() +
        cublasdx::Precision<Input, Input, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::col_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<256>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<Arch>());
};

template <typename scalar_t>
__device__ __forceinline__ float load_relation_value(
    const scalar_t* __restrict__ u,
    int64_t sample,
    int64_t head,
    int64_t token,
    int rank,
    int64_t stride_b,
    int64_t stride_h,
    int64_t stride_n,
    int64_t stride_r) {
    const int64_t base =
        sample * stride_b + head * stride_h + token * stride_n;
    return static_cast<float>(u[base + rank * stride_r]);
}

template <typename scalar_t, int Rank, int RhsTile, int KTile, int Arch>
__global__ void masked_trace_solve_readout_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ c,
    const bool* __restrict__ valid_mask,
    const float* __restrict__ length_scale,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    scalar_t* __restrict__ output,
    int* __restrict__ info,
    bool trace_normalize,
    float normalization_eps,
    float length_reference,
    bool length_normalize,
    float* __restrict__ effective_alpha_output,
    float* __restrict__ denominator_output,
    float* __restrict__ scale_squared_output,
    int64_t u_stride_b,
    int64_t u_stride_h,
    int64_t u_stride_n,
    int64_t u_stride_r,
    int64_t c_stride_b,
    int64_t c_stride_h,
    int64_t c_stride_n,
    int64_t c_stride_w,
    int64_t output_stride_b,
    int64_t output_stride_h,
    int64_t output_stride_n,
    int64_t output_stride_w,
    int64_t batches,
    int64_t heads,
    int64_t sequence,
    int64_t rhs_width) {
    using Input = typename GemmInputPrecision<scalar_t>::Type;
    using Gram = typename GemmTraits<scalar_t, Rank, Rank, KTile, Arch>::Type;
    using Cross = typename GemmTraits<scalar_t, Rank, RhsTile, KTile, Arch>::Type;
    using Readout = typename GemmTraits<scalar_t, KTile, RhsTile, Rank, Arch>::Type;
    using Solver = SolverTraits<Rank, RhsTile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;

    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(Gram);
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);

    // Each CTA owns one compact system. The token dimension is intentionally
    // reduced in KTile chunks below, so long sequences do not require an
    // intermediate [B, H, chunks, r, r] statistics tensor or a second
    // reduction kernel. A grid over all B*H systems naturally supplies the
    // large-batch schedule and keeps the Cholesky state CTA-local.
    const int64_t batch = static_cast<int64_t>(blockIdx.x);
    if (batch >= batches) return;

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int u_tile_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr int cross_b_elements = cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr int readout_b_elements = cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr int c_tile_elements =
        cross_b_elements > readout_b_elements ? cross_b_elements : readout_b_elements;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Input* u_tile = reinterpret_cast<Input*>(smem_raw);
    Input* c_tile = u_tile + u_tile_elements;
    uintptr_t float_address = reinterpret_cast<uintptr_t>(c_tile + c_tile_elements);
    float_address = (float_address + 15u) & ~uintptr_t(15u);
    float* a = reinterpret_cast<float*>(float_address);
    float* b = a + Rank * lda;
    float* gemm_output = b + Rank * ldb;

    auto gram_a = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_b());
    auto cross_a = cublasdx::make_tensor(u_tile, Cross::get_layout_smem_a());
    auto cross_b = cublasdx::make_tensor(c_tile, Cross::get_layout_smem_b());
    auto gram_output_tensor = cublasdx::make_tensor(gemm_output, Gram::get_layout_smem_c());
    auto cross_output_tensor = cublasdx::make_tensor(gemm_output, Cross::get_layout_smem_c());
    auto readout_a = cublasdx::make_tensor(u_tile, Readout::get_layout_smem_a());
    auto readout_b = cublasdx::make_tensor(c_tile, Readout::get_layout_smem_b());
    auto readout_output_tensor = cublasdx::make_tensor(
        gemm_output, Readout::get_layout_smem_c());

    const int64_t sample = batch / heads;
    const int64_t head = batch - sample * heads;
    const bool* mask_batch = valid_mask == nullptr
        ? nullptr : valid_mask + sample * sequence;
    const float scale = trace_normalize || length_scale == nullptr
        ? 1.0f : length_scale[sample];
    static_assert(KTile == 32, "masked tile ballot assumes one warp per token tile");
    __shared__ unsigned int tile_valid_bits;
    __shared__ int valid_count;
    __shared__ float effective_alpha_shared;
    if (threadIdx.x == 0) valid_count = 0;
    __syncthreads();

    for (int linear = threadIdx.x; linear < Rank * Rank; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
        if (threadIdx.x < KTile) {
            const int64_t token = sequence_start + threadIdx.x;
            const bool valid = token < sequence &&
                (mask_batch == nullptr || mask_batch[token]);
            const unsigned int bits = __ballot_sync(0xffffffffu, valid);
            if (threadIdx.x == 0) tile_valid_bits = bits;
        }
        __syncthreads();
        const unsigned int valid_bits = tile_valid_bits;
        if (threadIdx.x == 0) valid_count += __popc(valid_bits);
        if (valid_bits == 0) continue;
        const bool full_tile = valid_bits == 0xffffffffu;
        for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
            const int row = linear / KTile;
            const int k = linear - row * KTile;
            const int64_t token = sequence_start + k;
            const bool valid = full_tile || ((valid_bits >> k) & 1u);
            gram_a(row, k) = valid
                ? static_cast<Input>(load_relation_value(
                    u, sample, head, token, row,
                    u_stride_b, u_stride_h, u_stride_n, u_stride_r) * scale)
                : Input(0.0f);
        }
        __syncthreads();
        Gram().execute(1.0f, gram_a, gram_b, 1.0f, gram_output_tensor);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float scale_squared = 1.0f;
        float denominator = 1.0f;
        if (trace_normalize) {
            float energy = 0.0f;
            for (int diagonal = 0; diagonal < Rank; ++diagonal) {
                energy += gemm_output[diagonal * Rank + diagonal];
            }
            const int active_tokens = valid_count > 0 ? valid_count : 1;
            const float element_count =
                static_cast<float>(active_tokens * Rank);
            denominator = energy + normalization_eps * element_count;
            const float target = length_normalize
                ? static_cast<float>(Rank) * length_reference
                : element_count;
            scale_squared = target / fmaxf(denominator, FLT_MIN);
        }
        effective_alpha_shared = alpha[batch] * scale_squared;
        if (effective_alpha_output != nullptr) {
            effective_alpha_output[batch] = effective_alpha_shared;
            denominator_output[batch] = denominator;
            scale_squared_output[batch] = scale_squared;
        }
    }
    __syncthreads();
    const float alpha_batch = effective_alpha_shared;
    for (int linear = threadIdx.x; linear < Rank * Rank; linear += blockDim.x) {
        const int row = linear / Rank;
        const int col = linear - row * Rank;
        a[row * lda + col] = alpha_batch * gemm_output[linear] + (row == col ? 1.0f : 0.0f);
    }
    __syncthreads();
    Potrf().execute(a, lda, info + batch);
    __syncthreads();

    for (int64_t rhs_start = 0; rhs_start < rhs_width; rhs_start += RhsTile) {
        for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
            gemm_output[linear] = 0.0f;
        }
        __syncthreads();
        for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
            if (threadIdx.x < KTile) {
                const int64_t token = sequence_start + threadIdx.x;
                const bool valid = token < sequence &&
                    (mask_batch == nullptr || mask_batch[token]);
                const unsigned int bits = __ballot_sync(0xffffffffu, valid);
                if (threadIdx.x == 0) tile_valid_bits = bits;
            }
            __syncthreads();
            const unsigned int valid_bits = tile_valid_bits;
            if (valid_bits == 0) continue;
            const bool full_tile = valid_bits == 0xffffffffu;
            for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
                const int row = linear / KTile;
                const int k = linear - row * KTile;
                const int64_t token = sequence_start + k;
                const bool valid = full_tile || ((valid_bits >> k) & 1u);
                cross_a(row, k) = valid
                    ? static_cast<Input>(load_relation_value(
                        u, sample, head, token, row,
                        u_stride_b, u_stride_h, u_stride_n, u_stride_r) * scale)
                    : Input(0.0f);
            }
            for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                const int k = linear / RhsTile;
                const int col = linear - k * RhsTile;
                const int64_t token = sequence_start + k;
                const int64_t global_col = rhs_start + col;
                const bool valid = full_tile || ((valid_bits >> k) & 1u);
                cross_b(k, col) = valid && global_col < rhs_width
                    ? static_cast<Input>(c[
                        sample * c_stride_b + head * c_stride_h +
                        token * c_stride_n + global_col * c_stride_w])
                    : Input(0.0f);
            }
            __syncthreads();
            Cross().execute(1.0f, cross_a, cross_b, 1.0f, cross_output_tensor);
            __syncthreads();
        }
        for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
            const int row = linear / RhsTile;
            const int col = linear - row * RhsTile;
            b[row * ldb + col] = gemm_output[linear];
        }
        __syncthreads();
        Potrs().execute(a, lda, b, ldb);
        __syncthreads();
        for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
            const int row = linear / RhsTile;
            const int col = linear - row * RhsTile;
            // Match the fallback contract exactly: the FP32 compact solve is
            // rounded once to the activation dtype before Tensor-Core readout.
            readout_b(row, col) = static_cast<Input>(b[row * ldb + col]);
        }
        __syncthreads();
        for (int64_t token_start = 0; token_start < sequence; token_start += KTile) {
                for (int linear = threadIdx.x; linear < KTile * Rank; linear += blockDim.x) {
                    const int row = linear / Rank;
                    const int col = linear - row * Rank;
                    const int64_t token = token_start + row;
                    const bool valid = token < sequence &&
                        (mask_batch == nullptr || mask_batch[token]);
                    // Mask is checked before the global load: padding values
                    // cannot enter either statistics or the readout.
                    readout_a(row, col) = valid
                        ? static_cast<Input>(load_relation_value(
                            u, sample, head, token, col, u_stride_b, u_stride_h,
                            u_stride_n, u_stride_r) * scale)
                        : Input(0.0f);
                }
                __syncthreads();
                for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                    gemm_output[linear] = 0.0f;
                }
                __syncthreads();
                Readout().execute(1.0f, readout_a, readout_b, 0.0f, readout_output_tensor);
                __syncthreads();
                for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                    const int row = linear / RhsTile;
                    const int col = linear - row * RhsTile;
                    const int64_t token = token_start + row;
                    const int64_t global_col = rhs_start + col;
                    if (token < sequence && global_col < rhs_width) {
                        if (mask_batch == nullptr || mask_batch[token]) {
                            const float local = static_cast<float>(c[
                                sample * c_stride_b + head * c_stride_h +
                                token * c_stride_n + global_col * c_stride_w]);
                            const int64_t output_index =
                                sample * output_stride_b + head * output_stride_h +
                                token * output_stride_n + global_col * output_stride_w;
                            output[output_index] = static_cast<scalar_t>(
                                (local - alpha_batch * gemm_output[linear]) *
                                gain[batch]);
                        } else {
                            const int64_t output_index =
                                sample * output_stride_b + head * output_stride_h +
                                token * output_stride_n + global_col * output_stride_w;
                            output[output_index] = scalar_t(0);
                        }
                    }
                }
                __syncthreads();
        }
    }
}

template <typename scalar_t, int Rank, int Arch>
void launch_masked_trace_solve_typed(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    const at::Tensor& gain,
    at::Tensor& output,
    at::Tensor& info,
    bool trace_normalize,
    float normalization_eps,
    float length_reference,
    bool length_normalize,
    at::Tensor& effective_alpha_output,
    at::Tensor& denominator_output,
    at::Tensor& scale_squared_output,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    constexpr int k_tile = 32;
    using Input = typename GemmInputPrecision<scalar_t>::Type;
    using Gram = typename GemmTraits<scalar_t, Rank, Rank, k_tile, Arch>::Type;
    using Cross = typename GemmTraits<scalar_t, Rank, rhs_tile, k_tile, Arch>::Type;
    using Readout = typename GemmTraits<scalar_t, k_tile, rhs_tile, Rank, Arch>::Type;
    using Solver = SolverTraits<Rank, rhs_tile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;
    constexpr int c_tile_elements =
        cublasdx::cosize(Cross::get_layout_smem_b()) >
                cublasdx::cosize(Readout::get_layout_smem_b())
            ? cublasdx::cosize(Cross::get_layout_smem_b())
            : cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr size_t smem_bytes = sizeof(Input) * (
        cublasdx::cosize(Gram::get_layout_smem_a()) + c_tile_elements) +
        15 + sizeof(float) * (
        Rank * Potrf::lda + Rank * Potrs::ldb +
        (Rank * Rank > k_tile * rhs_tile
            ? Rank * Rank : k_tile * rhs_tile));
    static_assert(cublasdx::cosize(Gram::get_layout_smem_a()) >=
                  cublasdx::cosize(Readout::get_layout_smem_a()));

    auto kernel = masked_trace_solve_readout_kernel<
        scalar_t, Rank, rhs_tile, k_tile, Arch>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    const int64_t systems = u.size(0) * u.size(1);
    TORCH_CHECK(
        systems <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "B * H exceeds the CUDA launch-grid limit");
    kernel<<<static_cast<unsigned int>(systems), 256, smem_bytes, stream>>>(
        u.const_data_ptr<scalar_t>(), c.const_data_ptr<scalar_t>(),
        valid_mask.numel() == 0 ? nullptr : valid_mask.const_data_ptr<bool>(),
        length_scale.numel() == 0 ? nullptr : length_scale.const_data_ptr<float>(),
        alpha.const_data_ptr<float>(), gain.const_data_ptr<float>(),
        output.mutable_data_ptr<scalar_t>(), info.mutable_data_ptr<int>(),
        trace_normalize, normalization_eps, length_reference, length_normalize,
        effective_alpha_output.numel() == 0
            ? nullptr : effective_alpha_output.mutable_data_ptr<float>(),
        denominator_output.numel() == 0
            ? nullptr : denominator_output.mutable_data_ptr<float>(),
        scale_squared_output.numel() == 0
            ? nullptr : scale_squared_output.mutable_data_ptr<float>(),
        u.stride(0), u.stride(1), u.stride(2), u.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        systems, u.size(1), u.size(2), c.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch>
void dispatch_masked_trace_solve_rank(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    const at::Tensor& gain,
    at::Tensor& output,
    at::Tensor& info,
    bool trace_normalize,
    float normalization_eps,
    float length_reference,
    bool length_normalize,
    at::Tensor& effective_alpha_output,
    at::Tensor& denominator_output,
    at::Tensor& scale_squared_output,
    cudaStream_t stream) {
    AT_DISPATCH_SWITCH(
        u.scalar_type(), "masked_trace_solve_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            auto launch = [&](auto rank_tag) {
                constexpr int rank = decltype(rank_tag)::value;
                launch_masked_trace_solve_typed<scalar_t, rank, Arch>(
                    u, c, valid_mask, length_scale, alpha, gain, output, info,
                    trace_normalize, normalization_eps, length_reference,
                    length_normalize, effective_alpha_output,
                    denominator_output, scale_squared_output, stream);
            };
            if (u.size(3) == 16) launch(std::integral_constant<int, 16>{});
            else if (u.size(3) == 32) launch(std::integral_constant<int, 32>{});
            else if (u.size(3) == 48) launch(std::integral_constant<int, 48>{});
            else if (u.size(3) == 64) launch(std::integral_constant<int, 64>{});
            else TORCH_CHECK(false, "MathDx fused backend supports rank 16/32/48/64, got ", u.size(3));
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            auto launch = [&](auto rank_tag) {
                constexpr int rank = decltype(rank_tag)::value;
                launch_masked_trace_solve_typed<scalar_t, rank, Arch>(
                    u, c, valid_mask, length_scale, alpha, gain, output, info,
                    trace_normalize, normalization_eps, length_reference,
                    length_normalize, effective_alpha_output,
                    denominator_output, scale_squared_output, stream);
            };
            if (u.size(3) == 16) launch(std::integral_constant<int, 16>{});
            else if (u.size(3) == 32) launch(std::integral_constant<int, 32>{});
            else if (u.size(3) == 48) launch(std::integral_constant<int, 48>{});
            else if (u.size(3) == 64) launch(std::integral_constant<int, 64>{});
            else TORCH_CHECK(false, "MathDx fused backend supports rank 16/32/48/64, got ", u.size(3));
        }));
}
template <typename mma_t>
__global__ void dual_backward_statistics_tensorcore_kernel(
    const mma_t* __restrict__ u,
    const mma_t* __restrict__ y,
    const mma_t* __restrict__ p,
    float* __restrict__ ytu,
    float* __restrict__ ptu,
    float* __restrict__ grad_mu,
    const bool* __restrict__ valid_mask,
    int64_t systems,
    int64_t sequence,
    int64_t width,
    int64_t rank_dim,
    int64_t heads,
    int64_t u_stride_b,
    int64_t u_stride_h,
    int64_t u_stride_n,
    int64_t u_stride_r,
    int64_t y_stride_b,
    int64_t y_stride_h,
    int64_t y_stride_n,
    int64_t y_stride_w,
    int64_t p_stride_b,
    int64_t p_stride_h,
    int64_t p_stride_n,
    int64_t p_stride_w) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    constexpr int tile = 16;
    const int64_t system = static_cast<int64_t>(blockIdx.x);
    const int64_t sample = system / heads;
    const int64_t head = system - sample * heads;
    const bool* mask_batch = valid_mask == nullptr
        ? nullptr : valid_mask + sample * sequence;
    const int width_start = static_cast<int>(blockIdx.y) * tile;
    const int rank_start = static_cast<int>(blockIdx.z) * tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (system >= systems || warp >= 2) return;

    extern __shared__ __align__(32) unsigned char smem_raw[];
    mma_t* y_tile = reinterpret_cast<mma_t*>(smem_raw);
    mma_t* p_tile = y_tile + tile * tile;
    mma_t* u_tile = p_tile + tile * tile;
    float* output_tiles = reinterpret_cast<float*>(u_tile + tile * tile);
    float* reduction = output_tiles + 2 * tile * tile;

    wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    float mu_partial = 0.0f;
    for (int64_t token_start = 0; token_start < sequence; token_start += tile) {
        for (int linear = threadIdx.x; linear < tile * tile;
             linear += blockDim.x) {
            const int row = linear / tile;
            const int column = linear - row * tile;
            const int64_t token = token_start + column;
            const int channel = width_start + row;
            const bool source_active = token < sequence && channel < width &&
                (mask_batch == nullptr || mask_batch[token]);
            const mma_t y_value = source_active
                ? y[sample * y_stride_b + head * y_stride_h +
                    token * y_stride_n + channel * y_stride_w]
                : mma_t(0.0f);
            const mma_t p_value = source_active
                ? p[sample * p_stride_b + head * p_stride_h +
                    token * p_stride_n + channel * p_stride_w]
                : mma_t(0.0f);
            y_tile[linear] = y_value;
            p_tile[linear] = p_value;
            if (rank_start == 0 && source_active) {
                mu_partial -= static_cast<float>(y_value) *
                    static_cast<float>(p_value);
            }
            const int64_t u_token = token_start + row;
            const int rank = rank_start + column;
            u_tile[linear] = u_token < sequence && rank < rank_dim &&
                    (mask_batch == nullptr || mask_batch[u_token])
                ? u[sample * u_stride_b + head * u_stride_h +
                    u_token * u_stride_n + rank * u_stride_r]
                : mma_t(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, tile, tile, tile, mma_t, wmma::row_major>
            source_fragment;
        wmma::fragment<wmma::matrix_b, tile, tile, tile, mma_t, wmma::row_major>
            u_fragment;
        wmma::load_matrix_sync(
            source_fragment, warp == 0 ? y_tile : p_tile, tile);
        wmma::load_matrix_sync(u_fragment, u_tile, tile);
        wmma::mma_sync(accumulator, source_fragment, u_fragment, accumulator);
        __syncthreads();
    }

    float* warp_output = output_tiles + warp * tile * tile;
    wmma::store_matrix_sync(
        warp_output, accumulator, tile, wmma::mem_row_major);
    __syncwarp();
    float* destination = warp == 0 ? ytu : ptu;
    const int64_t output_base = system * width * rank_dim;
    for (int linear = lane; linear < tile * tile; linear += 32) {
        const int row = linear / tile;
        const int column = linear - row * tile;
        const int channel = width_start + row;
        const int rank = rank_start + column;
        if (channel < width && rank < rank_dim) {
            destination[output_base + channel * rank_dim + rank] =
                warp_output[linear];
        }
    }

    if (rank_start == 0) {
        reduction[threadIdx.x] = mu_partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicAdd(grad_mu + system, reduction[0]);
    }
#endif
}

template <typename mma_t>
void launch_dual_backward_statistics_tensorcore(
    const at::Tensor& u,
    const at::Tensor& y,
    const at::Tensor& p,
    at::Tensor& ytu,
    at::Tensor& ptu,
    at::Tensor& grad_mu,
    const at::Tensor& valid_mask,
    int64_t heads,
    cudaStream_t stream) {
    constexpr int tile = 16;
    const bool four_dimensional = u.dim() == 4;
    const int64_t systems = four_dimensional ? u.size(0) * u.size(1) : u.size(0);
    const int64_t sequence = four_dimensional ? u.size(2) : u.size(1);
    const int64_t rank_dim = four_dimensional ? u.size(3) : u.size(2);
    const int64_t width = four_dimensional ? y.size(3) : y.size(2);
    const int64_t u_stride_b = four_dimensional ? u.stride(0) : heads * u.stride(0);
    const int64_t u_stride_h = four_dimensional ? u.stride(1) : u.stride(0);
    const int64_t u_stride_n = four_dimensional ? u.stride(2) : u.stride(1);
    const int64_t u_stride_r = four_dimensional ? u.stride(3) : u.stride(2);
    const int64_t y_stride_b = four_dimensional ? y.stride(0) : heads * y.stride(0);
    const int64_t y_stride_h = four_dimensional ? y.stride(1) : y.stride(0);
    const int64_t y_stride_n = four_dimensional ? y.stride(2) : y.stride(1);
    const int64_t y_stride_w = four_dimensional ? y.stride(3) : y.stride(2);
    const int64_t p_stride_b = four_dimensional ? p.stride(0) : heads * p.stride(0);
    const int64_t p_stride_h = four_dimensional ? p.stride(1) : p.stride(0);
    const int64_t p_stride_n = four_dimensional ? p.stride(2) : p.stride(1);
    const int64_t p_stride_w = four_dimensional ? p.stride(3) : p.stride(2);
    const dim3 grid(
        static_cast<unsigned int>(systems),
        static_cast<unsigned int>((width + tile - 1) / tile),
        static_cast<unsigned int>((rank_dim + tile - 1) / tile));
    constexpr int threads = 64;
    constexpr size_t smem_bytes =
        sizeof(mma_t) * 3 * tile * tile +
        sizeof(float) * (2 * tile * tile + threads);
    dual_backward_statistics_tensorcore_kernel<mma_t>
        <<<grid, threads, smem_bytes, stream>>>(
            reinterpret_cast<const mma_t*>(u.const_data_ptr()),
            reinterpret_cast<const mma_t*>(y.const_data_ptr()),
            reinterpret_cast<const mma_t*>(p.const_data_ptr()),
            ytu.mutable_data_ptr<float>(),
            ptu.mutable_data_ptr<float>(),
            grad_mu.mutable_data_ptr<float>(),
            valid_mask.numel() == 0 ? nullptr : valid_mask.const_data_ptr<bool>(),
            systems, sequence, width, rank_dim, heads,
            u_stride_b, u_stride_h, u_stride_n, u_stride_r,
            y_stride_b, y_stride_h, y_stride_n, y_stride_w,
            p_stride_b, p_stride_h, p_stride_n, p_stride_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_backward_statistics_tensorcore_cuda(
    const at::Tensor& u,
    const at::Tensor& y,
    const at::Tensor& p,
    const at::Tensor& valid_mask,
    int64_t heads) {
    TORCH_CHECK(
        u.is_cuda() && y.is_cuda() && p.is_cuda(),
        "dual backward-statistics inputs must be CUDA tensors");
    TORCH_CHECK(
        u.scalar_type() == y.scalar_type() && y.scalar_type() == p.scalar_type() &&
        (u.scalar_type() == at::kHalf || u.scalar_type() == at::kBFloat16),
        "dual backward statistics requires matching FP16/BF16 inputs");
    TORCH_CHECK(heads > 0, "heads must be positive");
    const bool four_dimensional = u.dim() == 4;
    TORCH_CHECK(
        (u.dim() == 3 || four_dimensional) && y.dim() == u.dim() &&
        p.dim() == u.dim() && y.sizes() == p.sizes(),
        "expected 3-D system-major or 4-D [B,H,N,width] inputs");
    TORCH_CHECK(
        four_dimensional
            ? (u.sizes().slice(0, 3) == y.sizes().slice(0, 3) &&
               u.size(1) == heads)
            : (u.size(0) == y.size(0) && u.size(1) == y.size(1) &&
               heads > 0 && u.size(0) % heads == 0),
        "U and Y/P logical dimensions differ");
    const int64_t systems = four_dimensional ? u.size(0) * u.size(1) : u.size(0);
    const int64_t sequence = four_dimensional ? u.size(2) : u.size(1);
    const int64_t rank_dim = four_dimensional ? u.size(3) : u.size(2);
    const int64_t width = four_dimensional ? y.size(3) : y.size(2);
    const int64_t batches = systems / heads;
    TORCH_CHECK(
        rank_dim == 16 || rank_dim == 32 || rank_dim == 48 || rank_dim == 64,
        "dual backward statistics requires rank 16/32/48/64");
    TORCH_CHECK(heads > 0 && systems % heads == 0,
                "heads must divide the system count");
    TORCH_CHECK(
        valid_mask.numel() == 0 ||
            (valid_mask.is_cuda() && valid_mask.scalar_type() == at::kBool &&
             valid_mask.is_contiguous() && valid_mask.dim() == 2 &&
              valid_mask.size(0) == batches &&
              valid_mask.size(1) == sequence),
        "valid_mask must be empty or contiguous bool [B,N]");
    const c10::cuda::CUDAGuard device_guard(u.device());
    auto options = u.options().dtype(at::kFloat);
    auto ytu = at::empty({systems, width, rank_dim}, options);
    auto ptu = at::empty_like(ytu);
    auto grad_mu = at::zeros({systems}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();
    if (u.scalar_type() == at::kBFloat16) {
        launch_dual_backward_statistics_tensorcore<__nv_bfloat16>(
            u, y, p, ytu, ptu, grad_mu, valid_mask, heads, stream);
    } else {
        launch_dual_backward_statistics_tensorcore<half>(
            u, y, p, ytu, ptu, grad_mu, valid_mask, heads, stream);
    }
    return {ytu, ptu, grad_mu};
}

template <typename mma_t>
__global__ void dual_grad_u_tensorcore_kernel(
    const mma_t* __restrict__ p,
    const mma_t* __restrict__ y,
    const mma_t* __restrict__ ytu,
    const mma_t* __restrict__ ptu,
    const float* __restrict__ coefficient,
    const mma_t* __restrict__ radial_u,
    const float* __restrict__ radial_coefficient,
    const bool* __restrict__ valid_mask,
    mma_t* __restrict__ grad_u,
    int64_t systems,
    int64_t sequence,
    int64_t width,
    int64_t rank_dim,
    int64_t heads,
    int64_t p_stride_b,
    int64_t p_stride_h,
    int64_t p_stride_n,
    int64_t p_stride_w,
    int64_t y_stride_b,
    int64_t y_stride_h,
    int64_t y_stride_n,
    int64_t y_stride_w,
    int64_t radial_stride_b,
    int64_t radial_stride_h,
    int64_t radial_stride_n,
    int64_t radial_stride_r) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    constexpr int tile = 16;
    const int rank_tiles = static_cast<int>((rank_dim + tile - 1) / tile);
    const int64_t system = static_cast<int64_t>(blockIdx.x);
    const int64_t sample = system / heads;
    const int64_t head = system - sample * heads;
    const bool* mask_batch = valid_mask == nullptr
        ? nullptr : valid_mask + sample * sequence;
    const int64_t token_start = static_cast<int64_t>(blockIdx.y) * tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (system >= systems || warp >= rank_tiles) return;

    extern __shared__ __align__(32) unsigned char smem_raw[];
    mma_t* p_tile = reinterpret_cast<mma_t*>(smem_raw);
    mma_t* y_tile = p_tile + tile * tile;
    float* output_tiles = reinterpret_cast<float*>(y_tile + tile * tile);

    wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const int rank_start = warp * tile;
    const int64_t compact_base = system * width * rank_dim;

    for (int64_t k_start = 0; k_start < width; k_start += tile) {
        for (int linear = threadIdx.x; linear < tile * tile;
             linear += blockDim.x) {
            const int row = linear / tile;
            const int column = linear - row * tile;
            const int64_t token = token_start + row;
            const int64_t channel = k_start + column;
            const bool active = token < sequence && channel < width &&
                (mask_batch == nullptr || mask_batch[token]);
            p_tile[linear] = active
                ? p[sample * p_stride_b + head * p_stride_h +
                    token * p_stride_n + channel * p_stride_w]
                : mma_t(0.0f);
            y_tile[linear] = active
                ? y[sample * y_stride_b + head * y_stride_h +
                    token * y_stride_n + channel * y_stride_w]
                : mma_t(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, tile, tile, tile, mma_t, wmma::row_major> p_fragment;
        wmma::fragment<wmma::matrix_a, tile, tile, tile, mma_t, wmma::row_major> y_fragment;
        wmma::fragment<wmma::matrix_b, tile, tile, tile, mma_t, wmma::row_major> ytu_fragment;
        wmma::fragment<wmma::matrix_b, tile, tile, tile, mma_t, wmma::row_major> ptu_fragment;
        wmma::load_matrix_sync(p_fragment, p_tile, tile);
        wmma::load_matrix_sync(y_fragment, y_tile, tile);
        wmma::load_matrix_sync(
            ytu_fragment,
            ytu + compact_base + k_start * rank_dim + rank_start,
            rank_dim);
        wmma::load_matrix_sync(
            ptu_fragment,
            ptu + compact_base + k_start * rank_dim + rank_start,
            rank_dim);
        wmma::mma_sync(accumulator, p_fragment, ytu_fragment, accumulator);
        wmma::mma_sync(accumulator, y_fragment, ptu_fragment, accumulator);
        __syncthreads();
    }

    const float scale = coefficient[system];
    for (int index = 0; index < accumulator.num_elements; ++index) {
        accumulator.x[index] *= scale;
    }
    float* warp_output = output_tiles + warp * tile * tile;
    wmma::store_matrix_sync(
        warp_output, accumulator, tile, wmma::mem_row_major);
    __syncwarp();
    const int64_t output_base = system * sequence * rank_dim;
    for (int linear = lane; linear < tile * tile; linear += 32) {
        const int row = linear / tile;
        const int column = linear - row * tile;
        const int64_t token = token_start + row;
        if (token < sequence) {
            const int64_t output_index =
                output_base + token * rank_dim + rank_start + column;
            if (mask_batch != nullptr && !mask_batch[token]) {
                grad_u[output_index] = mma_t(0.0f);
                continue;
            }
            float value = warp_output[linear];
            if (radial_u != nullptr) {
                value += radial_coefficient[system] *
                    static_cast<float>(radial_u[
                        sample * radial_stride_b + head * radial_stride_h +
                        token * radial_stride_n +
                        (rank_start + column) * radial_stride_r]);
            }
            grad_u[output_index] = mma_t(value);
        }
    }
#endif
}

template <typename mma_t>
void launch_dual_grad_u_tensorcore(
    const at::Tensor& p,
    const at::Tensor& y,
    const at::Tensor& ytu,
    const at::Tensor& ptu,
    const at::Tensor& coefficient,
    const at::Tensor& radial_u,
    const at::Tensor& radial_coefficient,
    const at::Tensor& valid_mask,
    int64_t heads,
    at::Tensor& grad_u,
    cudaStream_t stream) {
    constexpr int tile = 16;
    const bool four_dimensional = p.dim() == 4;
    const int64_t systems = four_dimensional ? p.size(0) * p.size(1) : p.size(0);
    const int64_t sequence = four_dimensional ? p.size(2) : p.size(1);
    const int64_t width = four_dimensional ? p.size(3) : p.size(2);
    const int64_t p_stride_b = four_dimensional ? p.stride(0) : heads * p.stride(0);
    const int64_t p_stride_h = four_dimensional ? p.stride(1) : p.stride(0);
    const int64_t p_stride_n = four_dimensional ? p.stride(2) : p.stride(1);
    const int64_t p_stride_w = four_dimensional ? p.stride(3) : p.stride(2);
    const int64_t y_stride_b = four_dimensional ? y.stride(0) : heads * y.stride(0);
    const int64_t y_stride_h = four_dimensional ? y.stride(1) : y.stride(0);
    const int64_t y_stride_n = four_dimensional ? y.stride(2) : y.stride(1);
    const int64_t y_stride_w = four_dimensional ? y.stride(3) : y.stride(2);
    const bool radial_four_dimensional = radial_u.dim() == 4;
    const int64_t radial_stride_b = radial_u.numel() == 0 ? 0 : (
        radial_four_dimensional ? radial_u.stride(0) : heads * radial_u.stride(0));
    const int64_t radial_stride_h = radial_u.numel() == 0 ? 0 : (
        radial_four_dimensional ? radial_u.stride(1) : radial_u.stride(0));
    const int64_t radial_stride_n = radial_u.numel() == 0 ? 0 : (
        radial_four_dimensional ? radial_u.stride(2) : radial_u.stride(1));
    const int64_t radial_stride_r = radial_u.numel() == 0 ? 0 : (
        radial_four_dimensional ? radial_u.stride(3) : radial_u.stride(2));
    const int rank_tiles = static_cast<int>((ytu.size(2) + tile - 1) / tile);
    const dim3 grid(
        static_cast<unsigned int>(systems),
        static_cast<unsigned int>((sequence + tile - 1) / tile));
    const int threads = 32 * rank_tiles;
    const size_t smem_bytes =
        sizeof(mma_t) * 2 * tile * tile +
        sizeof(float) * rank_tiles * tile * tile;
    dual_grad_u_tensorcore_kernel<mma_t>
        <<<grid, threads, smem_bytes, stream>>>(
            reinterpret_cast<const mma_t*>(p.const_data_ptr()),
            reinterpret_cast<const mma_t*>(y.const_data_ptr()),
            reinterpret_cast<const mma_t*>(ytu.const_data_ptr()),
            reinterpret_cast<const mma_t*>(ptu.const_data_ptr()),
            coefficient.const_data_ptr<float>(),
            radial_u.numel() == 0
                ? nullptr
                : reinterpret_cast<const mma_t*>(radial_u.const_data_ptr()),
            radial_coefficient.numel() == 0
                ? nullptr
                : radial_coefficient.const_data_ptr<float>(),
            valid_mask.numel() == 0 ? nullptr : valid_mask.const_data_ptr<bool>(),
            reinterpret_cast<mma_t*>(grad_u.mutable_data_ptr()),
            systems, sequence, width, ytu.size(2), heads,
            p_stride_b, p_stride_h, p_stride_n, p_stride_w,
            y_stride_b, y_stride_h, y_stride_n, y_stride_w,
            radial_stride_b, radial_stride_h, radial_stride_n, radial_stride_r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor dual_grad_u_tensorcore_cuda(
    const at::Tensor& p,
    const at::Tensor& y,
    const at::Tensor& ytu,
    const at::Tensor& ptu,
    const at::Tensor& coefficient,
    const at::Tensor& radial_u,
    const at::Tensor& radial_coefficient,
    const at::Tensor& valid_mask,
    int64_t heads) {
    TORCH_CHECK(
        p.is_cuda() && y.is_cuda() && ytu.is_cuda() && ptu.is_cuda() && coefficient.is_cuda(),
        "dual grad-U tensors must be CUDA tensors");
    TORCH_CHECK(
        p.scalar_type() == y.scalar_type() && p.scalar_type() == ytu.scalar_type() &&
        p.scalar_type() == ptu.scalar_type() &&
        (p.scalar_type() == at::kHalf || p.scalar_type() == at::kBFloat16),
        "dual grad-U requires matching FP16/BF16 inputs");
    TORCH_CHECK(
        coefficient.scalar_type() == at::kFloat && coefficient.dim() == 1,
        "dual grad-U coefficient must be float32 [systems]");
    TORCH_CHECK(heads > 0, "heads must be positive");
    const bool four_dimensional = p.dim() == 4;
    TORCH_CHECK(
        (p.dim() == 3 || four_dimensional) && p.sizes() == y.sizes() &&
        ytu.dim() == 3 && ytu.sizes() == ptu.sizes(),
        "dual grad-U expects 3-D system-major or 4-D [B,H,N,width] P/Y");
    TORCH_CHECK(
        four_dimensional ? p.size(1) == heads : p.size(0) % heads == 0,
        "heads must match or divide the system dimensions");
    const int64_t systems = four_dimensional ? p.size(0) * p.size(1) : p.size(0);
    const int64_t sequence = four_dimensional ? p.size(2) : p.size(1);
    const int64_t width = four_dimensional ? p.size(3) : p.size(2);
    const int64_t batches = systems / heads;
    TORCH_CHECK(
        systems == ytu.size(0) && width == ytu.size(1) &&
        coefficient.size(0) == systems,
        "dual grad-U dimensions differ");
    const bool add_radial = radial_u.numel() != 0;
    TORCH_CHECK(
        add_radial == (radial_coefficient.numel() != 0),
        "radial_u and radial_coefficient must both be empty or both be present");
    if (add_radial) {
        TORCH_CHECK(
            radial_u.is_cuda() && radial_u.scalar_type() == p.scalar_type() &&
            radial_u.dim() == p.dim() &&
            (four_dimensional
                ? (radial_u.size(0) == p.size(0) &&
                   radial_u.size(1) == p.size(1) &&
                   radial_u.size(2) == sequence && radial_u.size(3) == ytu.size(2))
                : (radial_u.size(0) == systems &&
                   radial_u.size(1) == sequence && radial_u.size(2) == ytu.size(2))),
            "radial_u logical dimensions differ");
        TORCH_CHECK(
            radial_coefficient.is_cuda() &&
            radial_coefficient.scalar_type() == at::kFloat &&
            radial_coefficient.dim() == 1 &&
            radial_coefficient.size(0) == systems &&
            radial_coefficient.is_contiguous(),
            "radial_coefficient must be contiguous float32 [systems]");
    }
    TORCH_CHECK(
        width % 16 == 0 &&
        (ytu.size(2) == 16 || ytu.size(2) == 32 ||
         ytu.size(2) == 48 || ytu.size(2) == 64),
        "dual grad-U requires width multiple of 16 and rank 16/32/48/64");
    TORCH_CHECK(
        valid_mask.numel() == 0 ||
            (valid_mask.is_cuda() && valid_mask.scalar_type() == at::kBool &&
             valid_mask.is_contiguous() && valid_mask.dim() == 2 &&
              valid_mask.size(0) == batches &&
              valid_mask.size(1) == sequence),
        "valid_mask must be empty or contiguous bool [B,N]");
    TORCH_CHECK(
        ytu.is_contiguous() && ptu.is_contiguous() && coefficient.is_contiguous(),
        "dual grad-U compact inputs and coefficients must be contiguous");
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto grad_u = four_dimensional
        ? at::empty({p.size(0), p.size(1), sequence, ytu.size(2)}, p.options())
        : at::empty({systems, sequence, ytu.size(2)}, p.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(p.get_device()).stream();
    if (p.scalar_type() == at::kBFloat16) {
        launch_dual_grad_u_tensorcore<__nv_bfloat16>(
            p, y, ytu, ptu, coefficient, radial_u, radial_coefficient,
            valid_mask, heads, grad_u, stream);
    } else {
        launch_dual_grad_u_tensorcore<half>(
            p, y, ytu, ptu, coefficient, radial_u, radial_coefficient,
            valid_mask, heads, grad_u, stream);
    }
    return grad_u;
}

template <int Rank, int RhsTile, int Arch, int BatchesPerBlock>
__global__ void solve_spd_kernel(
    const float* __restrict__ gram,
    const float* __restrict__ rhs,
    float* __restrict__ solution,
    int* __restrict__ info,
    int64_t batches,
    int64_t input_rank,
    int64_t rhs_width) {
    using Traits = SolverTraits<Rank, RhsTile, Arch, BatchesPerBlock>;
    using Potrf = typename Traits::Potrf;
    using Potrs = typename Traits::Potrs;

    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);

    const int64_t batch_base =
        static_cast<int64_t>(blockIdx.x) * BatchesPerBlock;
    if (batch_base >= batches) {
        return;
    }

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int a_elements = Rank * lda;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* a = reinterpret_cast<float*>(smem_raw);
    float* b = a + BatchesPerBlock * a_elements;

    for (int linear = threadIdx.x;
         linear < BatchesPerBlock * Rank * Rank;
         linear += blockDim.x) {
        const int local_batch = linear / (Rank * Rank);
        const int matrix_linear = linear - local_batch * Rank * Rank;
        const int row = matrix_linear / Rank;
        const int col = matrix_linear - row * Rank;
        const int64_t batch = batch_base + local_batch;
        a[local_batch * a_elements + row * lda + col] =
            row < input_rank && col < input_rank
            ? gram[batch * input_rank * input_rank + row * input_rank + col]
            : (row == col ? 1.0f : 0.0f);
    }
    __syncthreads();

    Potrf().execute(a, lda, info + batch_base);
    __syncthreads();

    for (int64_t rhs_start = 0; rhs_start < rhs_width; rhs_start += RhsTile) {
        for (int linear = threadIdx.x;
             linear < BatchesPerBlock * Rank * RhsTile;
             linear += blockDim.x) {
            const int local_batch = linear / (Rank * RhsTile);
            const int tile_linear = linear - local_batch * Rank * RhsTile;
            const int row = tile_linear / RhsTile;
            const int col = tile_linear - row * RhsTile;
            const int64_t global_col = rhs_start + col;
            const int64_t batch = batch_base + local_batch;
            b[local_batch * Rank * ldb + row * ldb + col] =
                row < input_rank && global_col < rhs_width
                ? rhs[batch * input_rank * rhs_width + row * rhs_width + global_col]
                : 0.0f;
        }
        __syncthreads();

        Potrs().execute(a, lda, b, ldb);
        __syncthreads();

        for (int linear = threadIdx.x;
             linear < BatchesPerBlock * Rank * RhsTile;
             linear += blockDim.x) {
            const int local_batch = linear / (Rank * RhsTile);
            const int tile_linear = linear - local_batch * Rank * RhsTile;
            const int row = tile_linear / RhsTile;
            const int col = tile_linear - row * RhsTile;
            const int64_t global_col = rhs_start + col;
            if (row < input_rank && global_col < rhs_width) {
                const int64_t batch = batch_base + local_batch;
                solution[batch * input_rank * rhs_width + row * rhs_width + global_col] =
                    b[local_batch * Rank * ldb + row * ldb + col];
            }
        }
        __syncthreads();
    }
}

template <int Rank, int Arch>
void launch_solve_typed(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    using Traits = SolverTraits<Rank, rhs_tile, Arch, 1>;
    using Potrf = typename Traits::Potrf;
    using Potrs = typename Traits::Potrs;

    constexpr size_t manual_smem =
        sizeof(float) * (Rank * Potrf::lda + Rank * Potrs::ldb);
    constexpr size_t solver_smem =
        Potrf::shared_memory_size > Potrs::shared_memory_size
            ? Potrf::shared_memory_size
            : Potrs::shared_memory_size;
    constexpr size_t smem_bytes = manual_smem > solver_smem ? manual_smem : solver_smem;

    auto kernel = solve_spd_kernel<Rank, rhs_tile, Arch, 1>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    const auto batches = gram.size(0);
    kernel<<<static_cast<unsigned int>(batches), 256,
             smem_bytes, stream>>>(
        gram.const_data_ptr<float>(),
        rhs.const_data_ptr<float>(),
        solution.mutable_data_ptr<float>(),
        info.mutable_data_ptr<int>(),
        batches,
        gram.size(1),
        rhs.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Rank, int Arch>
void launch_solve(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    launch_solve_typed<Rank, Arch>(gram, rhs, solution, info, stream);
}

template <int Arch>
void dispatch_rank(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    if (gram.size(1) <= 16) {
        launch_solve<16, Arch>(gram, rhs, solution, info, stream);
    } else if (gram.size(1) <= 32) {
        launch_solve<32, Arch>(gram, rhs, solution, info, stream);
    } else if (gram.size(1) <= 48) {
        launch_solve<48, Arch>(gram, rhs, solution, info, stream);
    } else if (gram.size(1) <= 64) {
        launch_solve<64, Arch>(gram, rhs, solution, info, stream);
    } else {
        TORCH_CHECK(false, "MathDx backend supports bucketed ranks 1 through 64, got ",
                    gram.size(1));
    }
}

std::tuple<at::Tensor, at::Tensor> solve_spd_impl(
    const at::Tensor& gram,
    const at::Tensor& rhs) {
    TORCH_CHECK(gram.is_cuda() && rhs.is_cuda(), "gram and rhs must be CUDA tensors");
    TORCH_CHECK(gram.scalar_type() == at::kFloat, "gram must be float32");
    TORCH_CHECK(rhs.scalar_type() == at::kFloat, "rhs must be float32");
    TORCH_CHECK(gram.is_contiguous() && rhs.is_contiguous(), "gram and rhs must be contiguous");
    TORCH_CHECK(gram.dim() == 3, "gram must have shape [batch, rank, rank]");
    TORCH_CHECK(rhs.dim() == 3, "rhs must have shape [batch, rank, rhs_width]");
    TORCH_CHECK(gram.size(0) == rhs.size(0), "gram and rhs batch dimensions differ");
    TORCH_CHECK(gram.size(1) == gram.size(2), "gram must be square");
    TORCH_CHECK(gram.size(1) == rhs.size(1), "gram rank and rhs rank differ");
    TORCH_CHECK(
        gram.size(0) > 0 && gram.size(1) > 0 && rhs.size(2) > 0,
        "empty batches/ranks/RHS are unsupported");
    TORCH_CHECK(gram.get_device() == rhs.get_device(), "gram and rhs must be on the same GPU");
    const c10::cuda::CUDAGuard device_guard(gram.device());
    auto solution = at::empty_like(rhs);
    auto info = at::zeros(
        {gram.size(0)},
        gram.options().dtype(at::kInt));
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const int cc = props->major * 10 + props->minor;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(gram.get_device()).stream();

    // MathDx descriptors are architecture-specific. Minor revisions without a
    // dedicated descriptor use the closest compatible family descriptor.
#ifdef LSSO_MATHDX_NATIVE_ARCH
    dispatch_rank<LSSO_MATHDX_NATIVE_ARCH>(
        gram, rhs, solution, info, stream);
#else
    if (cc >= 121) {
        dispatch_rank<1210>(gram, rhs, solution, info, stream);
    } else if (cc >= 120) {
        dispatch_rank<1200>(gram, rhs, solution, info, stream);
    } else if (cc >= 110) {
        dispatch_rank<kThorMathDxArch>(gram, rhs, solution, info, stream);
    } else if (cc >= 103) {
        dispatch_rank<1030>(gram, rhs, solution, info, stream);
    } else if (cc >= 100) {
        dispatch_rank<1000>(gram, rhs, solution, info, stream);
    } else if (cc >= 90) {
        dispatch_rank<900>(gram, rhs, solution, info, stream);
    } else if (cc >= 89) {
        dispatch_rank<890>(gram, rhs, solution, info, stream);
    } else if (cc >= 87) {
        dispatch_rank<870>(gram, rhs, solution, info, stream);
    } else if (cc >= 86) {
        dispatch_rank<860>(gram, rhs, solution, info, stream);
    } else if (cc >= 80) {
        dispatch_rank<800>(gram, rhs, solution, info, stream);
    } else {
        TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU, got compute capability ",
                    props->major, ".", props->minor);
    }
#endif
    return {solution, info};
}

std::tuple<at::Tensor, at::Tensor> solve_spd_cuda(
    const at::Tensor& gram, const at::Tensor& rhs) {
    return solve_spd_impl(gram, rhs);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_readout_impl(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    const at::Tensor& gain,
    bool trace_normalize,
    double normalization_eps,
    double length_reference,
    bool length_normalize) {
    const bool unmasked_direct =
        valid_mask.numel() == 0 && length_scale.numel() == 0;
    TORCH_CHECK(
        u.is_cuda() && c.is_cuda() && valid_mask.is_cuda() &&
        length_scale.is_cuda() && alpha.is_cuda() && gain.is_cuda(),
        "all readout inputs must be CUDA tensors");
    TORCH_CHECK(
        u.scalar_type() == c.scalar_type() &&
        (u.scalar_type() == at::kHalf || u.scalar_type() == at::kBFloat16),
        "u and c must have the same float16 or bfloat16 dtype");
    TORCH_CHECK(valid_mask.scalar_type() == at::kBool, "valid_mask must be bool");
    TORCH_CHECK(length_scale.scalar_type() == at::kFloat, "length_scale must be float32");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
    TORCH_CHECK(
        gain.scalar_type() == at::kFloat && gain.is_contiguous() &&
        gain.dim() == 1 && gain.size(0) == u.size(0) * u.size(1),
        "gain must be contiguous float32 with shape [B * H]");
    TORCH_CHECK(
        u.is_contiguous() && valid_mask.is_contiguous() &&
        length_scale.is_contiguous() && alpha.is_contiguous(),
        "U, masks, scales, and coefficients must be contiguous");
    TORCH_CHECK(u.dim() == 4, "u must have shape [B, H, N, r]");
    TORCH_CHECK(c.dim() == 4, "c must have shape [B, H, N, rhs_width]");
    TORCH_CHECK(
        unmasked_direct || valid_mask.dim() == 2,
        "valid_mask must be empty or have shape [B, N]");
    TORCH_CHECK(
        unmasked_direct || length_scale.dim() == 1,
        "length_scale must be empty or have shape [B]");
    TORCH_CHECK(alpha.dim() == 1, "alpha must have shape [B * H]");
    TORCH_CHECK(u.sizes().slice(0, 3) == c.sizes().slice(0, 3),
                "u/c leading dimensions differ");
    TORCH_CHECK(
        unmasked_direct ||
            (valid_mask.size(0) == u.size(0) && valid_mask.size(1) == u.size(2)),
        "valid_mask shape does not match u");
    TORCH_CHECK(
        unmasked_direct || length_scale.size(0) == u.size(0),
        "length_scale batch differs");
    TORCH_CHECK(
        alpha.size(0) == u.size(0) * u.size(1),
        "alpha must contain B * H values");
    TORCH_CHECK(
        u.size(0) > 0 && u.size(1) > 0 && u.size(2) > 0 && c.size(3) > 0,
        "empty dimensions are unsupported");
    TORCH_CHECK(
        u.get_device() == c.get_device() &&
        u.get_device() == valid_mask.get_device() &&
        u.get_device() == length_scale.get_device() &&
        u.get_device() == alpha.get_device() &&
        u.get_device() == gain.get_device(),
        "all readout inputs must be on the same GPU");

    const c10::cuda::CUDAGuard device_guard(u.device());
    const int64_t systems = u.size(0) * u.size(1);
    // Preserve the logical [B,H,N,W] API while storing the activation in
    // token-major [B,N,H,W] order. The following module output projection can
    // consume output.transpose(1,2) without materializing a layout copy.
    auto output = at::empty(
        {u.size(0), u.size(2), u.size(1), c.size(3)}, c.options()
    ).transpose(1, 2);
    auto info = at::zeros({systems}, u.options().dtype(at::kInt));
    auto trace_options = u.options().dtype(at::kFloat);
    auto effective_alpha_output = trace_normalize
        ? at::empty({systems}, trace_options) : at::empty({0}, trace_options);
    auto denominator_output = trace_normalize
        ? at::empty({systems}, trace_options) : at::empty({0}, trace_options);
    auto scale_squared_output = trace_normalize
        ? at::empty({systems}, trace_options) : at::empty({0}, trace_options);
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const int cc = props->major * 10 + props->minor;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();

#define DISPATCH_MASKED_TRACE(ARCH) \
    dispatch_masked_trace_solve_rank<ARCH>( \
        u, c, valid_mask, length_scale, alpha, gain, output, info, \
        trace_normalize, static_cast<float>(normalization_eps), \
        static_cast<float>(length_reference), length_normalize, \
        effective_alpha_output, denominator_output, scale_squared_output, stream)
#ifdef LSSO_MATHDX_NATIVE_ARCH
    DISPATCH_MASKED_TRACE(LSSO_MATHDX_NATIVE_ARCH);
#else
    if (cc >= 121) DISPATCH_MASKED_TRACE(1210);
    else if (cc >= 120) DISPATCH_MASKED_TRACE(1200);
    else if (cc >= 110) DISPATCH_MASKED_TRACE(kThorMathDxArch);
    else if (cc >= 103) DISPATCH_MASKED_TRACE(1030);
    else if (cc >= 100) DISPATCH_MASKED_TRACE(1000);
    else if (cc >= 90) DISPATCH_MASKED_TRACE(900);
    else if (cc >= 89) DISPATCH_MASKED_TRACE(890);
    else if (cc >= 87) DISPATCH_MASKED_TRACE(870);
    else if (cc >= 86) DISPATCH_MASKED_TRACE(860);
    else if (cc >= 80) DISPATCH_MASKED_TRACE(800);
    else TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU");
#endif
#undef DISPATCH_MASKED_TRACE
    return {
        output, info, effective_alpha_output,
        denominator_output, scale_squared_output,
    };
}
std::tuple<at::Tensor, at::Tensor> masked_stats_solve_readout_cuda(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    const at::Tensor& gain) {
    auto result = masked_readout_impl(
        u, c, valid_mask, length_scale, alpha, gain,
        false, 0.0, 1.0, false);
    return {std::get<0>(result), std::get<1>(result)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_trace_stats_solve_readout_cuda(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& alpha,
    const at::Tensor& gain,
    double normalization_eps,
    double length_reference,
    bool length_normalize) {
    // An empty scale is the explicit unmasked contract: the fused CTA kernel
    // derives N directly and receives null mask/scale pointers.  Masked calls
    // retain a per-sample unit scale because their valid lengths are read from
    // the mask inside the kernel.
    auto unit_scale = valid_mask.numel() == 0
        ? at::empty({0}, u.options().dtype(at::kFloat))
        : at::ones({u.size(0)}, u.options().dtype(at::kFloat));
    return masked_readout_impl(
        u, c, valid_mask, unit_scale, alpha, gain,
        true, normalization_eps, length_reference, length_normalize);
}

at::Tensor rank_rotary_cuda(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin,
    bool inverse) {
    TORCH_CHECK(input.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "input, cos, and sin must be CUDA tensors");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                "cos and sin must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must have shape [B, H, N, r]");
    TORCH_CHECK(input.size(3) % 2 == 0, "rank must be even");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin shapes must match");
    const int64_t factors_per_sample = input.size(2) * input.size(3) / 2;
    TORCH_CHECK(
        cos.numel() == factors_per_sample ||
        cos.numel() == input.size(0) * factors_per_sample,
        "cos/sin must contain N*(r/2) shared factors or B*N*(r/2) factors");
    TORCH_CHECK(input.scalar_type() == cos.scalar_type() &&
                input.scalar_type() == sin.scalar_type(),
                "input, cos, and sin dtypes must match");
    TORCH_CHECK(input.get_device() == cos.get_device() &&
                input.get_device() == sin.get_device(),
                "input, cos, and sin must be on the same GPU");

    const c10::cuda::CUDAGuard device_guard(input.device());
    // The kernel writes logical [B,H,N,r] order regardless of input strides.
    auto output = at::empty(input.sizes(), input.options());
    const int64_t pair_count = input.numel() / 2;
    const int64_t factor_batches = cos.numel() / factors_per_sample;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pair_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "rank_rotary_cuda",
        [&] {
            if (inverse) {
                rank_rotary_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
                    input.const_data_ptr<scalar_t>(),
                    cos.const_data_ptr<scalar_t>(),
                    sin.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    pair_count,
                    input.size(0), input.size(1), input.size(2), input.size(3) / 2,
                    factor_batches,
                    input.stride(0), input.stride(1), input.stride(2), input.stride(3));
            } else {
                rank_rotary_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
                    input.const_data_ptr<scalar_t>(),
                    cos.const_data_ptr<scalar_t>(),
                    sin.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    pair_count,
                    input.size(0), input.size(1), input.size(2), input.size(3) / 2,
                    factor_batches,
                    input.stride(0), input.stride(1), input.stride(2), input.stride(3));
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

TORCH_LIBRARY(lsso_mathdx, m) {
    m.def("backend_abi() -> int", []() -> int64_t { return 1; });
    m.def("solve_spd(Tensor gram, Tensor rhs) -> (Tensor solution, Tensor info)");
    m.def("masked_stats_solve_readout(Tensor u, Tensor c, Tensor valid_mask, Tensor length_scale, Tensor alpha, Tensor gain) -> (Tensor output, Tensor info)");
    m.def("masked_trace_stats_solve_readout(Tensor u, Tensor c, Tensor valid_mask, Tensor alpha, Tensor gain, float normalization_eps, float length_reference, bool length_normalize) -> (Tensor output, Tensor info, Tensor effective_alpha, Tensor denominator, Tensor scale_squared)");
    m.def("dual_backward_statistics_tensorcore(Tensor u, Tensor y, Tensor p, Tensor valid_mask, int heads) -> (Tensor ytu, Tensor ptu, Tensor grad_mu)");
    m.def("dual_grad_u_tensorcore(Tensor p, Tensor y, Tensor ytu, Tensor ptu, Tensor coefficient, Tensor radial_u, Tensor radial_coefficient, Tensor valid_mask, int heads) -> Tensor");
    m.def("rank_rotary(Tensor input, Tensor cos, Tensor sin, bool inverse=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(lsso_mathdx, CUDA, m) {
    m.impl("solve_spd", &solve_spd_cuda);
    m.impl("masked_stats_solve_readout", &masked_stats_solve_readout_cuda);
    m.impl("masked_trace_stats_solve_readout", &masked_trace_stats_solve_readout_cuda);
    m.impl("dual_backward_statistics_tensorcore", &dual_backward_statistics_tensorcore_cuda);
    m.impl("dual_grad_u_tensorcore", &dual_grad_u_tensorcore_cuda);
    m.impl("rank_rotary", &rank_rotary_cuda);
}
