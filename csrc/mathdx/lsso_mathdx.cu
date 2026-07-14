#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
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

using block_barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cuda_exp = cuda::device::experimental;
#if CUDART_VERSION >= 13000
constexpr int kThorMathDxArch = 1100;
#else
// CUDA 13 renamed Thor from SM101 to SM110. MathDx descriptors follow the
// toolkit-specific name even though runtime compute capability reports 11.0.
constexpr int kThorMathDxArch = 1010;
#endif

template <int TileRows, int TileColumns>
__device__ __forceinline__ block_barrier::arrival_token issue_tma_tile(
    void* destination,
    const CUtensorMap* tensor_map,
    int column,
    int row,
    int batch,
    block_barrier& barrier) {
    if (threadIdx.x == 0) {
        cuda_exp::cp_async_bulk_tensor_3d_global_to_shared(
            destination, tensor_map, column, row, batch, barrier);
        return cuda::device::barrier_arrive_tx(
            barrier, 1, TileRows * TileColumns * 2);
    }
    return barrier.arrive();
}

template <int KTile, int Rank, int RhsTile>
__device__ __forceinline__ block_barrier::arrival_token issue_tma_cross_tiles(
    void* u_destination,
    void* c_destination,
    const CUtensorMap* u_map,
    const CUtensorMap* c_map,
    int sequence_start,
    int rhs_start,
    int batch,
    block_barrier& barrier) {
    if (threadIdx.x == 0) {
        cuda_exp::cp_async_bulk_tensor_3d_global_to_shared(
            u_destination, u_map, 0, sequence_start, batch, barrier);
        cuda_exp::cp_async_bulk_tensor_3d_global_to_shared(
            c_destination, c_map, rhs_start, sequence_start, batch, barrier);
        return cuda::device::barrier_arrive_tx(
            barrier, 1, KTile * (Rank + RhsTile) * 2);
    }
    return barrier.arrive();
}

template <typename scalar_t, bool Inverse>
__global__ void rank_rotary_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ output,
    int64_t pair_count,
    int64_t factors_per_head) {
    const int64_t pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t factor = pair % factors_per_head;
    const float even = static_cast<float>(input[2 * pair]);
    const float odd = static_cast<float>(input[2 * pair + 1]);
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

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t>
__device__ __forceinline__ void issue_async_contiguous_tile(
    scalar_t* dst, const scalar_t* src, int elements) {
    constexpr int copy_bytes = 16;
    const int bytes = elements * static_cast<int>(sizeof(scalar_t));
    for (int offset = threadIdx.x * copy_bytes; offset < bytes;
         offset += blockDim.x * copy_bytes) {
        __pipeline_memcpy_async(
            reinterpret_cast<char*>(dst) + offset,
            reinterpret_cast<const char*>(src) + offset,
            copy_bytes);
    }
}

template <typename scalar_t, int Rows, int Columns>
__device__ __forceinline__ void issue_async_rhs_tile(
    scalar_t* dst,
    const scalar_t* src,
    int64_t sequence_start,
    int64_t rhs_start,
    int64_t sequence,
    int64_t rhs_width) {
    constexpr int copy_bytes = 16;
    constexpr int values_per_copy = copy_bytes / sizeof(scalar_t);
    constexpr int copies_per_row = Columns / values_per_copy;
    for (int copy = threadIdx.x; copy < Rows * copies_per_row;
         copy += blockDim.x) {
        const int row = copy / copies_per_row;
        const int column = (copy - row * copies_per_row) * values_per_copy;
        const int64_t token = sequence_start + row;
        if (token < sequence) {
            __pipeline_memcpy_async(
                dst + row * Columns + column,
                src + token * rhs_width + rhs_start + column,
                copy_bytes);
        }
    }
}

template <typename scalar_t, bool Rotary>
__global__ void prepare_basis_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ output,
    float* __restrict__ inv_rms,
    int64_t tokens,
    int64_t sequence,
    int rank,
    float eps,
    float length_scale) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int lane = threadIdx.x;
    if (token >= tokens) return;
    const scalar_t* x = input + token * rank;
    scalar_t* y = output + token * rank;
    float square_sum = 0.0f;
    for (int j = lane; j < rank; j += 32) {
        const float value = static_cast<float>(x[j]);
        square_sum += value * value;
    }
    square_sum = warp_sum(square_sum);
    const float inv = rsqrtf(__shfl_sync(0xffffffff, square_sum, 0) / rank + eps);
    if (lane == 0) inv_rms[token] = inv;
    const float norm_scale = inv * length_scale;
    if constexpr (Rotary) {
        const int half = rank / 2;
        const int64_t factor_base = (token % sequence) * half;
        for (int pair = lane; pair < half; pair += 32) {
            const float even = static_cast<float>(x[2 * pair]) * norm_scale;
            const float odd = static_cast<float>(x[2 * pair + 1]) * norm_scale;
            const float c = static_cast<float>(cos[factor_base + pair]);
            const float s = static_cast<float>(sin[factor_base + pair]);
            y[2 * pair] = static_cast<scalar_t>(even * c - odd * s);
            y[2 * pair + 1] = static_cast<scalar_t>(even * s + odd * c);
        }
    } else {
        for (int j = lane; j < rank; j += 32) {
            y[j] = static_cast<scalar_t>(static_cast<float>(x[j]) * norm_scale);
        }
    }
}

template <typename scalar_t, bool Rotary>
__global__ void prepare_basis_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ inv_rms,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ grad_input,
    int64_t tokens,
    int64_t sequence,
    int rank,
    float length_scale) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int lane = threadIdx.x;
    if (token >= tokens) return;
    const scalar_t* grad = grad_output + token * rank;
    const scalar_t* x = input + token * rank;
    scalar_t* grad_x = grad_input + token * rank;
    const int half = rank / 2;
    const int64_t factor_base = (token % sequence) * half;
    float dot = 0.0f;
    for (int j = lane; j < rank; j += 32) {
        float g;
        if constexpr (Rotary) {
            const int pair = j / 2;
            const float ge = static_cast<float>(grad[2 * pair]);
            const float go = static_cast<float>(grad[2 * pair + 1]);
            const float c = static_cast<float>(cos[factor_base + pair]);
            const float s = static_cast<float>(sin[factor_base + pair]);
            g = (j % 2 == 0) ? ge * c + go * s : -ge * s + go * c;
        } else {
            g = static_cast<float>(grad[j]);
        }
        dot += g * static_cast<float>(x[j]);
    }
    dot = __shfl_sync(0xffffffff, warp_sum(dot), 0);
    const float inv = inv_rms[token];
    const float correction_scale = inv * inv * dot / rank;
    const float common = length_scale * inv;
    for (int j = lane; j < rank; j += 32) {
        float g;
        if constexpr (Rotary) {
            const int pair = j / 2;
            const float ge = static_cast<float>(grad[2 * pair]);
            const float go = static_cast<float>(grad[2 * pair + 1]);
            const float c = static_cast<float>(cos[factor_base + pair]);
            const float s = static_cast<float>(sin[factor_base + pair]);
            g = (j % 2 == 0) ? ge * c + go * s : -ge * s + go * c;
        } else {
            g = static_cast<float>(grad[j]);
        }
        grad_x[j] = static_cast<scalar_t>(
            common * (g - static_cast<float>(x[j]) * correction_scale));
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

template <int Rank, int Columns, int KTile, int Arch>
struct GemmTraits {
    using Type = decltype(
        cublasdx::Size<Rank, Columns, KTile>() +
        cublasdx::Precision<float>() +
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

template <typename scalar_t, int Rank, int RhsTile, int KTile, int Arch>
__global__ void masked_stats_solve_spd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ c,
    const bool* __restrict__ valid_mask,
    const float* __restrict__ length_scale,
    const float* __restrict__ alpha,
    float* __restrict__ solution,
    int* __restrict__ info,
    int64_t batches,
    int64_t heads,
    int64_t sequence,
    int64_t rhs_width) {
    using Gram = typename GemmTraits<Rank, Rank, KTile, Arch>::Type;
    using Cross = typename GemmTraits<Rank, RhsTile, KTile, Arch>::Type;
    using Solver = SolverTraits<Rank, RhsTile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;

    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(Gram);
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);

    const int64_t batch = static_cast<int64_t>(blockIdx.x);
    if (batch >= batches) return;

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int u_tile_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr int c_tile_elements = cublasdx::cosize(Cross::get_layout_smem_b());

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* u_tile = reinterpret_cast<float*>(smem_raw);
    float* c_tile = u_tile + u_tile_elements;
    float* a = c_tile + c_tile_elements;
    float* b = a + Rank * lda;
    float* gemm_output = b + Rank * ldb;

    auto gram_a = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_b());
    auto cross_a = cublasdx::make_tensor(u_tile, Cross::get_layout_smem_a());
    auto cross_b = cublasdx::make_tensor(c_tile, Cross::get_layout_smem_b());
    auto gram_output_tensor = cublasdx::make_tensor(gemm_output, Gram::get_layout_gmem_c());
    auto cross_output_tensor = cublasdx::make_tensor(gemm_output, Cross::get_layout_gmem_c());

    const int64_t sample = batch / heads;
    const scalar_t* u_batch = u + batch * sequence * Rank;
    const scalar_t* c_batch = c + batch * sequence * rhs_width;
    const bool* mask_batch = valid_mask + sample * sequence;
    const float scale = length_scale[sample];
    float* solution_batch = solution + batch * Rank * rhs_width;
    __shared__ int tile_has_valid;

    auto gram_accumulator = Gram().suggest_accumulator();
    gram_accumulator.clear();
    for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
        if (threadIdx.x == 0) {
            tile_has_valid = 0;
            const int64_t tile_end = min(sequence_start + KTile, sequence);
            for (int64_t token = sequence_start; token < tile_end; ++token) {
                if (mask_batch[token]) {
                    tile_has_valid = 1;
                    break;
                }
            }
        }
        __syncthreads();
        if (!tile_has_valid) continue;
        for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
            const int row = linear / KTile;
            const int k = linear - row * KTile;
            const int64_t token = sequence_start + k;
            const bool valid = token < sequence && mask_batch[token];
            gram_a(row, k) = valid
                ? static_cast<float>(u_batch[token * Rank + row]) * scale
                : 0.0f;
        }
        __syncthreads();
        Gram().execute(gram_a, gram_b, gram_accumulator);
        __syncthreads();
    }
    gram_accumulator.partition_and_store(gram_output_tensor);
    __syncthreads();

    const float alpha_batch = alpha[batch];
    for (int linear = threadIdx.x; linear < Rank * Rank; linear += blockDim.x) {
        const int row = linear / Rank;
        const int col = linear - row * Rank;
        a[row * lda + col] = alpha_batch * gemm_output[linear] + (row == col ? 1.0f : 0.0f);
    }
    __syncthreads();
    Potrf().execute(a, lda, info + batch);
    __syncthreads();

    for (int64_t rhs_start = 0; rhs_start < rhs_width; rhs_start += RhsTile) {
        auto cross_accumulator = Cross().suggest_accumulator();
        cross_accumulator.clear();
        for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
            if (threadIdx.x == 0) {
                tile_has_valid = 0;
                const int64_t tile_end = min(sequence_start + KTile, sequence);
                for (int64_t token = sequence_start; token < tile_end; ++token) {
                    if (mask_batch[token]) {
                        tile_has_valid = 1;
                        break;
                    }
                }
            }
            __syncthreads();
            if (!tile_has_valid) continue;
            for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
                const int row = linear / KTile;
                const int k = linear - row * KTile;
                const int64_t token = sequence_start + k;
                const bool valid = token < sequence && mask_batch[token];
                cross_a(row, k) = valid
                    ? static_cast<float>(u_batch[token * Rank + row]) * scale
                    : 0.0f;
            }
            for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                const int k = linear / RhsTile;
                const int col = linear - k * RhsTile;
                const int64_t token = sequence_start + k;
                const int64_t global_col = rhs_start + col;
                const bool valid = token < sequence && mask_batch[token];
                cross_b(k, col) = valid && global_col < rhs_width
                    ? static_cast<float>(c_batch[token * rhs_width + global_col])
                    : 0.0f;
            }
            __syncthreads();
            Cross().execute(cross_a, cross_b, cross_accumulator);
            __syncthreads();
        }
        cross_accumulator.partition_and_store(cross_output_tensor);
        __syncthreads();
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
            const int64_t global_col = rhs_start + col;
            if (global_col < rhs_width) {
                solution_batch[row * rhs_width + global_col] = b[row * ldb + col];
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, int Rank, int RhsTile, int KTile, int Arch,
          bool FuseReadout, bool UseTma>
__global__ void stats_solve_spd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ c,
    const float* __restrict__ alpha,
    const float* __restrict__ inv_mu,
    float* __restrict__ solution,
    scalar_t* __restrict__ output,
    const __grid_constant__ CUtensorMap u_tensor_map,
    const __grid_constant__ CUtensorMap c_tensor_map,
    int* __restrict__ info,
    int64_t batches,
    int64_t sequence,
    int64_t rhs_width) {
    using Gram = typename GemmTraits<Rank, Rank, KTile, Arch>::Type;
    using Cross = typename GemmTraits<Rank, RhsTile, KTile, Arch>::Type;
    using Readout = typename GemmTraits<KTile, RhsTile, Rank, Arch>::Type;
    using Solver = SolverTraits<Rank, RhsTile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;

    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(Gram);
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);

    const int64_t batch = static_cast<int64_t>(blockIdx.x);
    if (batch >= batches) {
        return;
    }

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int u_tile_elements =
        cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr int c_tile_elements =
        cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr bool use_tma = UseTma && Arch >= 1000 && sizeof(scalar_t) == 2;
    constexpr bool use_async_staging =
        !use_tma && Arch >= 1000 && sizeof(scalar_t) == 2;
    constexpr int output_elements =
        (Rank > KTile ? Rank : KTile) * RhsTile;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ block_barrier tma_barriers[2];
    float* u_tile = reinterpret_cast<float*>(smem_raw);
    float* c_tile = u_tile + u_tile_elements;
    float* a = c_tile + c_tile_elements;
    float* b = a + Rank * lda;
    float* gemm_output = b + Rank * ldb;
    uintptr_t staging_address = reinterpret_cast<uintptr_t>(gemm_output + output_elements);
    staging_address = (staging_address + 127u) & ~uintptr_t(127u);
    scalar_t* raw_u = reinterpret_cast<scalar_t*>(staging_address);
    scalar_t* raw_c = raw_u + ((use_async_staging || use_tma) ? 2 * KTile * Rank : 0);

    auto gram_a = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(u_tile, Gram::get_layout_smem_b());
    auto cross_a = cublasdx::make_tensor(u_tile, Cross::get_layout_smem_a());
    auto cross_b = cublasdx::make_tensor(c_tile, Cross::get_layout_smem_b());
    auto gram_output_tensor = cublasdx::make_tensor(
        gemm_output,
        Gram::get_layout_gmem_c());
    auto cross_output_tensor = cublasdx::make_tensor(
        gemm_output,
        Cross::get_layout_gmem_c());
    auto readout_a = cublasdx::make_tensor(u_tile, Readout::get_layout_smem_a());
    auto readout_b = cublasdx::make_tensor(c_tile, Readout::get_layout_smem_b());
    auto readout_output_tensor = cublasdx::make_tensor(
        gemm_output, Readout::get_layout_gmem_c());

    const scalar_t* u_batch = u + batch * sequence * Rank;
    const scalar_t* c_batch = c + batch * sequence * rhs_width;
    float* solution_batch = FuseReadout ? nullptr : solution + batch * Rank * rhs_width;
    scalar_t* output_batch = FuseReadout ? output + batch * sequence * rhs_width : nullptr;

    auto gram_accumulator = Gram().suggest_accumulator();
    gram_accumulator.clear();
    if constexpr (use_tma) {
        if (threadIdx.x == 0) {
            init(&tma_barriers[0], blockDim.x);
            init(&tma_barriers[1], blockDim.x);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        auto first_token = issue_tma_tile<KTile, Rank>(
            raw_u, &u_tensor_map, 0, 0, batch, tma_barriers[0]);
        tma_barriers[0].wait(cuda::std::move(first_token));
    }
    if constexpr (use_async_staging) {
        const int first_tokens = static_cast<int>(sequence < KTile ? sequence : KTile);
        issue_async_contiguous_tile(
            raw_u, u_batch, first_tokens * Rank);
        __pipeline_commit();
    }
    int gram_tile_index = 0;
    for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
        block_barrier::arrival_token next_tma_token{};
        const int64_t next_start = sequence_start + KTile;
        const bool has_next_tile = next_start < sequence;
        if constexpr (use_tma) {
            if (has_next_tile) {
                const int next_buffer = (gram_tile_index + 1) & 1;
                next_tma_token = issue_tma_tile<KTile, Rank>(
                    raw_u + next_buffer * KTile * Rank,
                    &u_tensor_map, 0, static_cast<int>(next_start), batch,
                    tma_barriers[next_buffer]);
            }
        }
        if constexpr (use_async_staging) {
            __pipeline_wait_prior(0);
            __syncthreads();
            const int64_t next_start = sequence_start + KTile;
            if (next_start < sequence) {
                const int next_tokens = static_cast<int>(
                    sequence - next_start < KTile ? sequence - next_start : KTile);
                issue_async_contiguous_tile(
                    raw_u + ((gram_tile_index + 1) & 1) * KTile * Rank,
                    u_batch + next_start * Rank,
                    next_tokens * Rank);
                __pipeline_commit();
            }
        }
        for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
            const int row = linear / KTile;
            const int k = linear - row * KTile;
            const int64_t token = sequence_start + k;
            if constexpr (use_async_staging || use_tma) {
                const scalar_t* staged_u =
                    raw_u + (gram_tile_index & 1) * KTile * Rank;
                gram_a(row, k) = token < sequence
                    ? static_cast<float>(staged_u[k * Rank + row])
                    : 0.0f;
            } else {
                gram_a(row, k) = token < sequence
                    ? static_cast<float>(u_batch[token * Rank + row])
                    : 0.0f;
            }
        }
        __syncthreads();
        Gram().execute(gram_a, gram_b, gram_accumulator);
        __syncthreads();
        if constexpr (use_tma) {
            if (has_next_tile) {
                tma_barriers[(gram_tile_index + 1) & 1].wait(
                    cuda::std::move(next_tma_token));
            }
        }
        ++gram_tile_index;
    }
    gram_accumulator.partition_and_store(gram_output_tensor);
    __syncthreads();

    const float alpha_batch = alpha[batch];
    for (int linear = threadIdx.x; linear < Rank * Rank; linear += blockDim.x) {
        const int row = linear / Rank;
        const int col = linear - row * Rank;
        a[row * lda + col] = alpha_batch * gemm_output[linear] + (row == col ? 1.0f : 0.0f);
    }
    __syncthreads();
    Potrf().execute(a, lda, info + batch);
    __syncthreads();

    for (int64_t rhs_start = 0; rhs_start < rhs_width; rhs_start += RhsTile) {
        auto cross_accumulator = Cross().suggest_accumulator();
        cross_accumulator.clear();
        const bool async_rhs = use_async_staging &&
            rhs_start + RhsTile <= rhs_width;
        if constexpr (use_tma) {
            if (threadIdx.x == 0) {
                init(&tma_barriers[0], blockDim.x);
                init(&tma_barriers[1], blockDim.x);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            __syncthreads();
            auto first_token = issue_tma_cross_tiles<KTile, Rank, RhsTile>(
                raw_u, raw_c, &u_tensor_map, &c_tensor_map,
                0, static_cast<int>(rhs_start), batch, tma_barriers[0]);
            tma_barriers[0].wait(cuda::std::move(first_token));
        }
        if (async_rhs) {
            const int first_tokens = static_cast<int>(sequence < KTile ? sequence : KTile);
            issue_async_contiguous_tile(raw_u, u_batch, first_tokens * Rank);
            issue_async_rhs_tile<scalar_t, KTile, RhsTile>(
                raw_c, c_batch, 0, rhs_start, sequence, rhs_width);
            __pipeline_commit();
        }
        int cross_tile_index = 0;
        for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
            block_barrier::arrival_token next_tma_token{};
            const int64_t tma_next_start = sequence_start + KTile;
            const bool has_next_tile = tma_next_start < sequence;
            if constexpr (use_tma) {
                if (has_next_tile) {
                    const int next_buffer = (cross_tile_index + 1) & 1;
                    next_tma_token = issue_tma_cross_tiles<KTile, Rank, RhsTile>(
                        raw_u + next_buffer * KTile * Rank,
                        raw_c + next_buffer * KTile * RhsTile,
                        &u_tensor_map, &c_tensor_map,
                        static_cast<int>(tma_next_start), static_cast<int>(rhs_start),
                        batch, tma_barriers[next_buffer]);
                }
            }
            if (async_rhs) {
                __pipeline_wait_prior(0);
                __syncthreads();
                const int64_t next_start = sequence_start + KTile;
                if (next_start < sequence) {
                    const int next_tokens = static_cast<int>(
                        sequence - next_start < KTile ? sequence - next_start : KTile);
                    const int next_buffer = (cross_tile_index + 1) & 1;
                    issue_async_contiguous_tile(
                        raw_u + next_buffer * KTile * Rank,
                        u_batch + next_start * Rank,
                        next_tokens * Rank);
                    issue_async_rhs_tile<scalar_t, KTile, RhsTile>(
                        raw_c + next_buffer * KTile * RhsTile,
                        c_batch, next_start, rhs_start, sequence, rhs_width);
                    __pipeline_commit();
                }
            }
            for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
                const int row = linear / KTile;
                const int k = linear - row * KTile;
                const int64_t token = sequence_start + k;
                if (async_rhs || use_tma) {
                    const scalar_t* staged_u =
                        raw_u + (cross_tile_index & 1) * KTile * Rank;
                    cross_a(row, k) = token < sequence
                        ? static_cast<float>(staged_u[k * Rank + row])
                        : 0.0f;
                } else {
                    cross_a(row, k) = token < sequence
                        ? static_cast<float>(u_batch[token * Rank + row])
                        : 0.0f;
                }
            }
            for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                const int k = linear / RhsTile;
                const int col = linear - k * RhsTile;
                const int64_t token = sequence_start + k;
                const int64_t global_col = rhs_start + col;
                if (async_rhs || use_tma) {
                    const scalar_t* staged_c =
                        raw_c + (cross_tile_index & 1) * KTile * RhsTile;
                    cross_b(k, col) = token < sequence
                        ? static_cast<float>(staged_c[k * RhsTile + col])
                        : 0.0f;
                } else {
                    cross_b(k, col) = token < sequence && global_col < rhs_width
                        ? static_cast<float>(c_batch[token * rhs_width + global_col])
                        : 0.0f;
                }
            }
            __syncthreads();
            Cross().execute(cross_a, cross_b, cross_accumulator);
            __syncthreads();
            if constexpr (use_tma) {
                if (has_next_tile) {
                    tma_barriers[(cross_tile_index + 1) & 1].wait(
                        cuda::std::move(next_tma_token));
                }
            }
            ++cross_tile_index;
        }
        cross_accumulator.partition_and_store(cross_output_tensor);
        __syncthreads();
        for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
            const int row = linear / RhsTile;
            const int col = linear - row * RhsTile;
            b[row * ldb + col] = gemm_output[linear];
        }
        __syncthreads();
        Potrs().execute(a, lda, b, ldb);
        __syncthreads();

        if constexpr (!FuseReadout) {
            for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
                const int row = linear / RhsTile;
                const int col = linear - row * RhsTile;
                const int64_t global_col = rhs_start + col;
                if (global_col < rhs_width) {
                    solution_batch[row * rhs_width + global_col] = b[row * ldb + col];
                }
            }
            __syncthreads();
        } else {
            // The statistics buffers are dead after POTRS, so reuse them for
            // U@K and apply the Woodbury epilogue before the sole global write.
            for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
                const int row = linear / RhsTile;
                const int col = linear - row * RhsTile;
                readout_b(row, col) = b[row * ldb + col];
            }
            __syncthreads();
            if constexpr (use_tma) {
                if (threadIdx.x == 0) {
                    init(&tma_barriers[0], blockDim.x);
                    init(&tma_barriers[1], blockDim.x);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncthreads();
                auto first_token = issue_tma_tile<KTile, Rank>(
                    raw_u, &u_tensor_map, 0, 0, batch, tma_barriers[0]);
                tma_barriers[0].wait(cuda::std::move(first_token));
            }
            int readout_tile_index = 0;
            for (int64_t token_start = 0; token_start < sequence; token_start += KTile) {
                block_barrier::arrival_token next_tma_token{};
                const int64_t next_start = token_start + KTile;
                const bool has_next_tile = next_start < sequence;
                if constexpr (use_tma) {
                    if (has_next_tile) {
                        const int next_buffer = (readout_tile_index + 1) & 1;
                        next_tma_token = issue_tma_tile<KTile, Rank>(
                            raw_u + next_buffer * KTile * Rank,
                            &u_tensor_map, 0, static_cast<int>(next_start), batch,
                            tma_barriers[next_buffer]);
                    }
                }
                for (int linear = threadIdx.x; linear < KTile * Rank; linear += blockDim.x) {
                    const int row = linear / Rank;
                    const int col = linear - row * Rank;
                    const int64_t token = token_start + row;
                    if constexpr (use_tma) {
                        const scalar_t* staged_u =
                            raw_u + (readout_tile_index & 1) * KTile * Rank;
                        readout_a(row, col) = token < sequence
                            ? static_cast<float>(staged_u[row * Rank + col])
                            : 0.0f;
                    } else {
                        readout_a(row, col) = token < sequence
                            ? static_cast<float>(u_batch[token * Rank + col])
                            : 0.0f;
                    }
                }
                __syncthreads();
                auto readout_accumulator = Readout().suggest_accumulator();
                readout_accumulator.clear();
                Readout().execute(readout_a, readout_b, readout_accumulator);
                readout_accumulator.partition_and_store(readout_output_tensor);
                __syncthreads();
                for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                    const int row = linear / RhsTile;
                    const int col = linear - row * RhsTile;
                    const int64_t token = token_start + row;
                    const int64_t global_col = rhs_start + col;
                    if (token < sequence && global_col < rhs_width) {
                        const float local = static_cast<float>(
                            c_batch[token * rhs_width + global_col]);
                        output_batch[token * rhs_width + global_col] = static_cast<scalar_t>(
                            (local - alpha_batch * gemm_output[linear]) * inv_mu[batch]);
                    }
                }
                __syncthreads();
                if constexpr (use_tma) {
                    if (has_next_tile) {
                        tma_barriers[(readout_tile_index + 1) & 1].wait(
                            cuda::std::move(next_tma_token));
                    }
                }
                ++readout_tile_index;
            }
        }
    }
}

template <typename scalar_t, int Rank, int Arch>
void launch_masked_stats_solve_typed(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    constexpr int k_tile = 32;
    using Gram = typename GemmTraits<Rank, Rank, k_tile, Arch>::Type;
    using Cross = typename GemmTraits<Rank, rhs_tile, k_tile, Arch>::Type;
    using Solver = SolverTraits<Rank, rhs_tile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;
    constexpr size_t smem_bytes = sizeof(float) * (
        cublasdx::cosize(Gram::get_layout_smem_a()) +
        cublasdx::cosize(Cross::get_layout_smem_b()) +
        Rank * Potrf::lda + Rank * Potrs::ldb + Rank * rhs_tile);

    auto kernel = masked_stats_solve_spd_kernel<scalar_t, Rank, rhs_tile, k_tile, Arch>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    const int64_t systems = u.size(0) * u.size(1);
    kernel<<<static_cast<unsigned int>(systems), 256, smem_bytes, stream>>>(
        u.const_data_ptr<scalar_t>(), c.const_data_ptr<scalar_t>(),
        valid_mask.const_data_ptr<bool>(), length_scale.const_data_ptr<float>(),
        alpha.const_data_ptr<float>(), solution.mutable_data_ptr<float>(),
        info.mutable_data_ptr<int>(), systems, u.size(1), u.size(2), c.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch>
void dispatch_masked_stats_solve_rank(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, u.scalar_type(),
        "masked_stats_solve_spd_cuda", [&] {
            if (u.size(3) == 16) {
                launch_masked_stats_solve_typed<scalar_t, 16, Arch>(
                    u, c, valid_mask, length_scale, alpha, solution, info, stream);
            } else if (u.size(3) == 32) {
                launch_masked_stats_solve_typed<scalar_t, 32, Arch>(
                    u, c, valid_mask, length_scale, alpha, solution, info, stream);
            } else {
                TORCH_CHECK(false, "MathDx backend supports rank 16 or 32, got ", u.size(3));
            }
        });
}

template <int Rank, int RhsTile, int Arch, int BatchesPerBlock>
__global__ void solve_spd_kernel(
    const float* __restrict__ gram,
    const float* __restrict__ rhs,
    float* __restrict__ solution,
    int* __restrict__ info,
    int64_t batches,
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
            gram[batch * Rank * Rank + matrix_linear];
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
            b[local_batch * Rank * ldb + row * ldb + col] = global_col < rhs_width
                ? rhs[batch * Rank * rhs_width + row * rhs_width + global_col]
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
            if (global_col < rhs_width) {
                const int64_t batch = batch_base + local_batch;
                solution[batch * Rank * rhs_width + row * rhs_width + global_col] =
                    b[local_batch * Rank * ldb + row * ldb + col];
            }
        }
        __syncthreads();
    }
}

template <int Rank, int Arch, int BatchesPerBlock>
void launch_solve_typed(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    using Traits = SolverTraits<Rank, rhs_tile, Arch, BatchesPerBlock>;
    using Potrf = typename Traits::Potrf;
    using Potrs = typename Traits::Potrs;

    constexpr size_t manual_smem =
        sizeof(float) * BatchesPerBlock *
        (Rank * Potrf::lda + Rank * Potrs::ldb);
    constexpr size_t solver_smem =
        Potrf::shared_memory_size > Potrs::shared_memory_size
            ? Potrf::shared_memory_size
            : Potrs::shared_memory_size;
    constexpr size_t smem_bytes = manual_smem > solver_smem ? manual_smem : solver_smem;

    auto kernel = solve_spd_kernel<Rank, rhs_tile, Arch, BatchesPerBlock>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    const auto batches = gram.size(0);
    const int64_t blocks = batches / BatchesPerBlock;
    kernel<<<static_cast<unsigned int>(blocks), 256 * BatchesPerBlock,
             smem_bytes, stream>>>(
        gram.const_data_ptr<float>(),
        rhs.const_data_ptr<float>(),
        solution.mutable_data_ptr<float>(),
        info.mutable_data_ptr<int>(),
        batches,
        rhs.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Rank, int Arch>
void launch_solve(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream,
    int requested_batches_per_block = 0) {
    const int64_t batches = gram.size(0);
    // Forced 2/4-system modes are retained for architecture benchmarking.
    // SM120 measurements show that one CTA per system remains faster even at
    // 768+ systems, so auto deliberately selects BPB=1.
    if (requested_batches_per_block == 4 && batches % 4 == 0) {
        launch_solve_typed<Rank, Arch, 4>(
            gram, rhs, solution, info, stream);
    } else if (requested_batches_per_block == 2 && batches % 2 == 0) {
        launch_solve_typed<Rank, Arch, 2>(
            gram, rhs, solution, info, stream);
    } else {
        launch_solve_typed<Rank, Arch, 1>(
            gram, rhs, solution, info, stream);
    }
}

template <int Arch>
void dispatch_rank(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream,
    int requested_batches_per_block = 0) {
    if (gram.size(1) == 16) {
        launch_solve<16, Arch>(gram, rhs, solution, info, stream,
                               requested_batches_per_block);
    } else if (gram.size(1) == 32) {
        launch_solve<32, Arch>(gram, rhs, solution, info, stream,
                               requested_batches_per_block);
    } else {
        TORCH_CHECK(false, "MathDx backend supports rank 16 or 32, got ", gram.size(1));
    }
}

CUtensorMap encode_tma_3d_map(const at::Tensor& tensor, uint32_t tile_columns) {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
        cudaDriverEntryPointQueryResult status;
        void* address = nullptr;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &address, 12000,
            cudaEnableDefault, &status));
        TORCH_CHECK(status == cudaDriverEntryPointSuccess && address != nullptr,
                    "cuTensorMapEncodeTiled is unavailable in the CUDA driver");
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
    }();

    CUtensorMap map{};
    const uint64_t dimensions[3] = {
        static_cast<uint64_t>(tensor.size(2)),
        static_cast<uint64_t>(tensor.size(1)),
        static_cast<uint64_t>(tensor.size(0))};
    const uint64_t strides[2] = {
        static_cast<uint64_t>(tensor.size(2) * tensor.element_size()),
        static_cast<uint64_t>(tensor.size(1) * tensor.size(2) * tensor.element_size())};
    const uint32_t box[3] = {tile_columns, 32u, 1u};
    const uint32_t element_strides[3] = {1u, 1u, 1u};
    const auto data_type = tensor.scalar_type() == at::kBFloat16
        ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    const CUresult result = encode(
        &map, data_type, 3,
        const_cast<void*>(tensor.const_data_ptr()),
        dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with code ", result);
    return map;
}

template <typename scalar_t, int Rank, int Arch, bool FuseReadout, bool UseTma>
void launch_stats_solve_typed(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& alpha,
    const at::Tensor& inv_mu,
    at::Tensor& solution,
    at::Tensor& output,
    at::Tensor& info,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    constexpr int k_tile = 32;
    using Gram = typename GemmTraits<Rank, Rank, k_tile, Arch>::Type;
    using Cross = typename GemmTraits<Rank, rhs_tile, k_tile, Arch>::Type;
    using Readout = typename GemmTraits<k_tile, rhs_tile, Rank, Arch>::Type;
    using Solver = SolverTraits<Rank, rhs_tile, Arch>;
    using Potrf = typename Solver::Potrf;
    using Potrs = typename Solver::Potrs;
    static_assert(cublasdx::cosize(Gram::get_layout_smem_a()) >=
                  cublasdx::cosize(Readout::get_layout_smem_a()));
    static_assert(cublasdx::cosize(Cross::get_layout_smem_b()) >=
                  cublasdx::cosize(Readout::get_layout_smem_b()));
    constexpr size_t async_staging_bytes =
        (Arch >= 1000 && sizeof(scalar_t) == 2)
        ? sizeof(scalar_t) * 2 * k_tile * (Rank + rhs_tile)
        : 0;
    constexpr size_t smem_bytes = sizeof(float) * (
        cublasdx::cosize(Gram::get_layout_smem_a()) +
        cublasdx::cosize(Cross::get_layout_smem_b()) +
        Rank * Potrf::lda +
        Rank * Potrs::ldb +
        (Rank > k_tile ? Rank : k_tile) * rhs_tile) + async_staging_bytes + 127;

    auto kernel = stats_solve_spd_kernel<
        scalar_t, Rank, rhs_tile, k_tile, Arch, FuseReadout, UseTma>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    CUtensorMap u_map{};
    CUtensorMap c_map{};
    if constexpr (UseTma) {
        u_map = encode_tma_3d_map(u, Rank);
        c_map = encode_tma_3d_map(c, rhs_tile);
    }
    kernel<<<static_cast<unsigned int>(u.size(0)), 256, smem_bytes, stream>>>(
        u.const_data_ptr<scalar_t>(),
        c.const_data_ptr<scalar_t>(),
        alpha.const_data_ptr<float>(),
        FuseReadout ? inv_mu.const_data_ptr<float>() : nullptr,
        FuseReadout ? nullptr : solution.mutable_data_ptr<float>(),
        FuseReadout ? output.mutable_data_ptr<scalar_t>() : nullptr,
        u_map,
        c_map,
        info.mutable_data_ptr<int>(),
        u.size(0),
        u.size(1),
        c.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch, bool FuseReadout>
void dispatch_stats_solve_rank(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& alpha,
    const at::Tensor& inv_mu,
    at::Tensor& solution,
    at::Tensor& output,
    at::Tensor& info,
    cudaStream_t stream) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, u.scalar_type(),
        "stats_solve_spd_cuda", [&] {
            // SM100 benefits from TMA's descriptor-driven bulk transfers.
            // On the development SM120, the smaller cp.async staging path is
            // 2--3% faster for RRLSSO tiles, so auto keeps it. The environment
            // override supports architecture-specific A/B without rebuilding.
            bool tma_enabled = Arch >= 1000 && Arch < 1200;
            if (const char* mode = std::getenv("LSSO_MATHDX_TMA")) {
                if (mode[0] == '1' || mode[0] == 'y' || mode[0] == 'Y') {
                    tma_enabled = true;
                } else if (mode[0] == '0' || mode[0] == 'n' || mode[0] == 'N') {
                    tma_enabled = false;
                }
            }
            const bool use_tma = tma_enabled && Arch >= 1000 &&
                sizeof(scalar_t) == 2 &&
                c.size(2) >= 32 && c.size(2) % 32 == 0;
            if (u.size(2) == 16) {
                if (use_tma) {
                    launch_stats_solve_typed<scalar_t, 16, Arch, FuseReadout, true>(
                        u, c, alpha, inv_mu, solution, output, info, stream);
                } else {
                    launch_stats_solve_typed<scalar_t, 16, Arch, FuseReadout, false>(
                        u, c, alpha, inv_mu, solution, output, info, stream);
                }
            } else if (u.size(2) == 32) {
                if (use_tma) {
                    launch_stats_solve_typed<scalar_t, 32, Arch, FuseReadout, true>(
                        u, c, alpha, inv_mu, solution, output, info, stream);
                } else {
                    launch_stats_solve_typed<scalar_t, 32, Arch, FuseReadout, false>(
                        u, c, alpha, inv_mu, solution, output, info, stream);
                }
            } else {
                TORCH_CHECK(false, "MathDx backend supports rank 16 or 32, got ", u.size(2));
            }
        });
}

std::tuple<at::Tensor, at::Tensor> solve_spd_impl(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    int requested_batches_per_block) {
    TORCH_CHECK(gram.is_cuda() && rhs.is_cuda(), "gram and rhs must be CUDA tensors");
    TORCH_CHECK(gram.scalar_type() == at::kFloat, "gram must be float32");
    TORCH_CHECK(rhs.scalar_type() == at::kFloat, "rhs must be float32");
    TORCH_CHECK(gram.is_contiguous() && rhs.is_contiguous(), "gram and rhs must be contiguous");
    TORCH_CHECK(gram.dim() == 3, "gram must have shape [batch, rank, rank]");
    TORCH_CHECK(rhs.dim() == 3, "rhs must have shape [batch, rank, rhs_width]");
    TORCH_CHECK(gram.size(0) == rhs.size(0), "gram and rhs batch dimensions differ");
    TORCH_CHECK(gram.size(1) == gram.size(2), "gram must be square");
    TORCH_CHECK(gram.size(1) == rhs.size(1), "gram rank and rhs rank differ");
    TORCH_CHECK(gram.size(0) > 0 && rhs.size(2) > 0, "empty batches/RHS are unsupported");
    TORCH_CHECK(gram.get_device() == rhs.get_device(), "gram and rhs must be on the same GPU");
    TORCH_CHECK(
        requested_batches_per_block == 0 || requested_batches_per_block == 1 ||
        requested_batches_per_block == 2 || requested_batches_per_block == 4,
        "batches_per_block must be one of 0(auto), 1, 2, or 4");

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
    if (cc >= 121) {
        dispatch_rank<1210>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 120) {
        dispatch_rank<1200>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 110) {
        dispatch_rank<kThorMathDxArch>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 103) {
        dispatch_rank<1030>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 100) {
        dispatch_rank<1000>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 90) {
        dispatch_rank<900>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 89) {
        dispatch_rank<890>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 87) {
        dispatch_rank<870>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 86) {
        dispatch_rank<860>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else if (cc >= 80) {
        dispatch_rank<800>(gram, rhs, solution, info, stream, requested_batches_per_block);
    } else {
        TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU, got compute capability ",
                    props->major, ".", props->minor);
    }
    return {solution, info};
}

std::tuple<at::Tensor, at::Tensor> solve_spd_cuda(
    const at::Tensor& gram, const at::Tensor& rhs) {
    return solve_spd_impl(gram, rhs, 0);
}

std::tuple<at::Tensor, at::Tensor> solve_spd_bpb_cuda(
    const at::Tensor& gram, const at::Tensor& rhs,
    int64_t batches_per_block) {
    return solve_spd_impl(gram, rhs, static_cast<int>(batches_per_block));
}

template <bool FuseReadout>
std::tuple<at::Tensor, at::Tensor> stats_solve_spd_impl(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& alpha,
    const at::Tensor& inv_mu) {
    TORCH_CHECK(u.is_cuda() && c.is_cuda() && alpha.is_cuda(), "u, c, and alpha must be CUDA tensors");
    TORCH_CHECK(
        u.scalar_type() == c.scalar_type() &&
        (u.scalar_type() == at::kFloat || u.scalar_type() == at::kHalf ||
         u.scalar_type() == at::kBFloat16),
        "u and c must have the same float32, float16, or bfloat16 dtype");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
    TORCH_CHECK(u.is_contiguous() && c.is_contiguous() && alpha.is_contiguous(),
                "u, c, and alpha must be contiguous");
    TORCH_CHECK(u.dim() == 3, "u must have shape [batch, sequence, rank]");
    TORCH_CHECK(c.dim() == 3, "c must have shape [batch, sequence, rhs_width]");
    TORCH_CHECK(alpha.dim() == 1, "alpha must have shape [batch]");
    TORCH_CHECK(u.size(0) == c.size(0) && u.size(0) == alpha.size(0), "batch dimensions differ");
    TORCH_CHECK(u.size(1) == c.size(1), "sequence dimensions differ");
    TORCH_CHECK(u.size(0) > 0 && u.size(1) > 0 && c.size(2) > 0, "empty dimensions are unsupported");
    TORCH_CHECK(u.get_device() == c.get_device() && u.get_device() == alpha.get_device(),
                "u, c, and alpha must be on the same GPU");
    if constexpr (FuseReadout) {
        TORCH_CHECK(inv_mu.is_cuda() && inv_mu.scalar_type() == at::kFloat &&
                    inv_mu.is_contiguous() && inv_mu.dim() == 1 &&
                    inv_mu.size(0) == u.size(0),
                    "inv_mu must be contiguous float32 with shape [batch]");
        TORCH_CHECK(inv_mu.get_device() == u.get_device(), "inv_mu must be on the same GPU");
    }

    const c10::cuda::CUDAGuard device_guard(u.device());
    auto solution = FuseReadout
        ? at::empty({0}, u.options().dtype(at::kFloat))
        : at::empty({u.size(0), u.size(2), c.size(2)}, u.options().dtype(at::kFloat));
    auto output = FuseReadout ? at::empty_like(c) : at::empty({0}, u.options());
    auto info = at::zeros({u.size(0)}, u.options().dtype(at::kInt));
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const int cc = props->major * 10 + props->minor;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();

#define DISPATCH_STATS(ARCH) \
    dispatch_stats_solve_rank<ARCH, FuseReadout>( \
        u, c, alpha, inv_mu, solution, output, info, stream)
    if (cc >= 121) {
        DISPATCH_STATS(1210);
    } else if (cc >= 120) {
        DISPATCH_STATS(1200);
    } else if (cc >= 110) {
        DISPATCH_STATS(kThorMathDxArch);
    } else if (cc >= 103) {
        DISPATCH_STATS(1030);
    } else if (cc >= 100) {
        DISPATCH_STATS(1000);
    } else if (cc >= 90) {
        DISPATCH_STATS(900);
    } else if (cc >= 89) {
        DISPATCH_STATS(890);
    } else if (cc >= 87) {
        DISPATCH_STATS(870);
    } else if (cc >= 86) {
        DISPATCH_STATS(860);
    } else if (cc >= 80) {
        DISPATCH_STATS(800);
    } else {
        TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU, got compute capability ",
                    props->major, ".", props->minor);
    }
#undef DISPATCH_STATS
    return {FuseReadout ? output : solution, info};
}

std::tuple<at::Tensor, at::Tensor> stats_solve_spd_cuda(
    const at::Tensor& u, const at::Tensor& c, const at::Tensor& alpha) {
    auto empty = at::empty({0}, alpha.options());
    return stats_solve_spd_impl<false>(u, c, alpha, empty);
}

std::tuple<at::Tensor, at::Tensor> stats_solve_readout_cuda(
    const at::Tensor& u, const at::Tensor& c, const at::Tensor& alpha,
    const at::Tensor& inv_mu) {
    return stats_solve_spd_impl<true>(u, c, alpha, inv_mu);
}

std::tuple<at::Tensor, at::Tensor> masked_stats_solve_spd_cuda(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& valid_mask,
    const at::Tensor& length_scale,
    const at::Tensor& alpha) {
    TORCH_CHECK(
        u.is_cuda() && c.is_cuda() && valid_mask.is_cuda() &&
        length_scale.is_cuda() && alpha.is_cuda(),
        "all masked stats inputs must be CUDA tensors");
    TORCH_CHECK(
        u.scalar_type() == c.scalar_type() &&
        (u.scalar_type() == at::kFloat || u.scalar_type() == at::kHalf ||
         u.scalar_type() == at::kBFloat16 || u.scalar_type() == at::kDouble),
        "u and c must have the same floating dtype");
    TORCH_CHECK(valid_mask.scalar_type() == at::kBool, "valid_mask must be bool");
    TORCH_CHECK(length_scale.scalar_type() == at::kFloat, "length_scale must be float32");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
    TORCH_CHECK(
        u.is_contiguous() && c.is_contiguous() && valid_mask.is_contiguous() &&
        length_scale.is_contiguous() && alpha.is_contiguous(),
        "all masked stats inputs must be contiguous");
    TORCH_CHECK(u.dim() == 4, "u must have shape [B, H, N, r]");
    TORCH_CHECK(c.dim() == 4, "c must have shape [B, H, N, rhs_width]");
    TORCH_CHECK(valid_mask.dim() == 2, "valid_mask must have shape [B, N]");
    TORCH_CHECK(length_scale.dim() == 1, "length_scale must have shape [B]");
    TORCH_CHECK(alpha.dim() == 1, "alpha must have shape [B * H]");
    TORCH_CHECK(u.sizes().slice(0, 3) == c.sizes().slice(0, 3), "u/c leading dimensions differ");
    TORCH_CHECK(valid_mask.size(0) == u.size(0) && valid_mask.size(1) == u.size(2),
                "valid_mask shape does not match u");
    TORCH_CHECK(length_scale.size(0) == u.size(0), "length_scale batch differs");
    TORCH_CHECK(alpha.size(0) == u.size(0) * u.size(1), "alpha must contain B * H values");
    TORCH_CHECK(u.size(0) > 0 && u.size(1) > 0 && u.size(2) > 0 && c.size(3) > 0,
                "empty dimensions are unsupported");
    TORCH_CHECK(
        u.get_device() == c.get_device() && u.get_device() == valid_mask.get_device() &&
        u.get_device() == length_scale.get_device() && u.get_device() == alpha.get_device(),
        "all masked stats inputs must be on the same GPU");

    const c10::cuda::CUDAGuard device_guard(u.device());
    const int64_t systems = u.size(0) * u.size(1);
    auto solution = at::empty({systems, u.size(3), c.size(3)}, u.options().dtype(at::kFloat));
    auto info = at::zeros({systems}, u.options().dtype(at::kInt));
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const int cc = props->major * 10 + props->minor;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();

#define DISPATCH_MASKED_STATS(ARCH) \
    dispatch_masked_stats_solve_rank<ARCH>( \
        u, c, valid_mask, length_scale, alpha, solution, info, stream)
    if (cc >= 121) {
        DISPATCH_MASKED_STATS(1210);
    } else if (cc >= 120) {
        DISPATCH_MASKED_STATS(1200);
    } else if (cc >= 110) {
        DISPATCH_MASKED_STATS(kThorMathDxArch);
    } else if (cc >= 103) {
        DISPATCH_MASKED_STATS(1030);
    } else if (cc >= 100) {
        DISPATCH_MASKED_STATS(1000);
    } else if (cc >= 90) {
        DISPATCH_MASKED_STATS(900);
    } else if (cc >= 89) {
        DISPATCH_MASKED_STATS(890);
    } else if (cc >= 87) {
        DISPATCH_MASKED_STATS(870);
    } else if (cc >= 86) {
        DISPATCH_MASKED_STATS(860);
    } else if (cc >= 80) {
        DISPATCH_MASKED_STATS(800);
    } else {
        TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU");
    }
#undef DISPATCH_MASKED_STATS
    return {solution, info};
}

at::Tensor rank_rotary_cuda(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin,
    bool inverse) {
    TORCH_CHECK(input.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "input, cos, and sin must be CUDA tensors");
    TORCH_CHECK(input.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
                "input, cos, and sin must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must have shape [B, H, N, r]");
    TORCH_CHECK(input.size(3) % 2 == 0, "rank must be even");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin shapes must match");
    TORCH_CHECK(cos.numel() == input.size(2) * input.size(3) / 2,
                "cos/sin must contain N * (r / 2) factors");
    TORCH_CHECK(input.scalar_type() == cos.scalar_type() &&
                input.scalar_type() == sin.scalar_type(),
                "input, cos, and sin dtypes must match");
    TORCH_CHECK(input.get_device() == cos.get_device() &&
                input.get_device() == sin.get_device(),
                "input, cos, and sin must be on the same GPU");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = at::empty_like(input);
    const int64_t pair_count = input.numel() / 2;
    const int64_t factors_per_head = cos.numel();
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
                    factors_per_head);
            } else {
                rank_rotary_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
                    input.const_data_ptr<scalar_t>(),
                    cos.const_data_ptr<scalar_t>(),
                    sin.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    pair_count,
                    factors_per_head);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool Rotary>
std::tuple<at::Tensor, at::Tensor> prepare_basis_cuda_impl(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin,
    double eps,
    double length_scale) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA");
    TORCH_CHECK(input.dim() == 4, "input must have shape [B, H, N, r]");
    TORCH_CHECK(input.size(3) > 0 && input.size(3) <= 1024, "unsupported rank");
    if constexpr (Rotary) {
        TORCH_CHECK(input.size(3) % 2 == 0, "rotary rank must be even");
        TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cos.is_contiguous() && sin.is_contiguous(),
                    "cos and sin must be contiguous CUDA tensors");
        TORCH_CHECK(input.scalar_type() == cos.scalar_type() && input.scalar_type() == sin.scalar_type(),
                    "input and rotary factor dtypes must match");
        TORCH_CHECK(cos.numel() == input.size(2) * input.size(3) / 2 && cos.sizes() == sin.sizes(),
                    "invalid rotary factor shape");
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = at::empty_like(input);
    const int64_t tokens = input.numel() / input.size(3);
    auto inv_rms = at::empty({tokens}, input.options().dtype(at::kFloat));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "prepare_basis_cuda", [&] {
            prepare_basis_kernel<scalar_t, Rotary><<<static_cast<unsigned int>(tokens), 32, 0, stream>>>(
                input.const_data_ptr<scalar_t>(),
                Rotary ? cos.const_data_ptr<scalar_t>() : nullptr,
                Rotary ? sin.const_data_ptr<scalar_t>() : nullptr,
                output.mutable_data_ptr<scalar_t>(), inv_rms.mutable_data_ptr<float>(),
                tokens, input.size(2), input.size(3), static_cast<float>(eps),
                static_cast<float>(length_scale));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, inv_rms};
}

template <bool Rotary>
at::Tensor prepare_basis_backward_cuda_impl(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& inv_rms,
    const at::Tensor& cos,
    const at::Tensor& sin,
    double length_scale) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda() && inv_rms.is_cuda(),
                "backward tensors must be CUDA");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous() && inv_rms.is_contiguous(),
                "backward tensors must be contiguous");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "gradient shape mismatch");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto grad_input = at::empty_like(input);
    const int64_t tokens = input.numel() / input.size(3);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "prepare_basis_backward_cuda", [&] {
            prepare_basis_backward_kernel<scalar_t, Rotary><<<static_cast<unsigned int>(tokens), 32, 0, stream>>>(
                grad_output.const_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
                inv_rms.const_data_ptr<float>(),
                Rotary ? cos.const_data_ptr<scalar_t>() : nullptr,
                Rotary ? sin.const_data_ptr<scalar_t>() : nullptr,
                grad_input.mutable_data_ptr<scalar_t>(), tokens, input.size(2), input.size(3),
                static_cast<float>(length_scale));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}

std::tuple<at::Tensor, at::Tensor> normalize_basis_cuda(
    const at::Tensor& input, double eps, double length_scale) {
    auto empty = at::empty({0}, input.options());
    return prepare_basis_cuda_impl<false>(input, empty, empty, eps, length_scale);
}

at::Tensor normalize_basis_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& inv_rms, double length_scale) {
    auto empty = at::empty({0}, input.options());
    return prepare_basis_backward_cuda_impl<false>(
        grad_output, input, inv_rms, empty, empty, length_scale);
}

std::tuple<at::Tensor, at::Tensor> normalize_rank_rotary_cuda(
    const at::Tensor& input, const at::Tensor& cos, const at::Tensor& sin,
    double eps, double length_scale) {
    return prepare_basis_cuda_impl<true>(input, cos, sin, eps, length_scale);
}

at::Tensor normalize_rank_rotary_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& inv_rms, const at::Tensor& cos, const at::Tensor& sin,
    double length_scale) {
    return prepare_basis_backward_cuda_impl<true>(
        grad_output, input, inv_rms, cos, sin, length_scale);
}

}  // namespace

TORCH_LIBRARY(lsso_mathdx, m) {
    m.def("solve_spd(Tensor gram, Tensor rhs) -> (Tensor solution, Tensor info)");
    m.def("solve_spd_bpb(Tensor gram, Tensor rhs, int batches_per_block) -> (Tensor solution, Tensor info)");
    m.def("stats_solve_spd(Tensor u, Tensor c, Tensor alpha) -> (Tensor solution, Tensor info)");
    m.def("stats_solve_readout(Tensor u, Tensor c, Tensor alpha, Tensor inv_mu) -> (Tensor output, Tensor info)");
    m.def("masked_stats_solve_spd(Tensor u, Tensor c, Tensor valid_mask, Tensor length_scale, Tensor alpha) -> (Tensor solution, Tensor info)");
    m.def("rank_rotary(Tensor input, Tensor cos, Tensor sin, bool inverse=False) -> Tensor");
    m.def("normalize_basis(Tensor input, float eps, float length_scale) -> (Tensor output, Tensor inv_rms)");
    m.def("normalize_basis_backward(Tensor grad_output, Tensor input, Tensor inv_rms, float length_scale) -> Tensor");
    m.def("normalize_rank_rotary(Tensor input, Tensor cos, Tensor sin, float eps, float length_scale) -> (Tensor output, Tensor inv_rms)");
    m.def("normalize_rank_rotary_backward(Tensor grad_output, Tensor input, Tensor inv_rms, Tensor cos, Tensor sin, float length_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(lsso_mathdx, CUDA, m) {
    m.impl("solve_spd", &solve_spd_cuda);
    m.impl("solve_spd_bpb", &solve_spd_bpb_cuda);
    m.impl("stats_solve_spd", &stats_solve_spd_cuda);
    m.impl("stats_solve_readout", &stats_solve_readout_cuda);
    m.impl("masked_stats_solve_spd", &masked_stats_solve_spd_cuda);
    m.impl("rank_rotary", &rank_rotary_cuda);
    m.impl("normalize_basis", &normalize_basis_cuda);
    m.impl("normalize_basis_backward", &normalize_basis_backward_cuda);
    m.impl("normalize_rank_rotary", &normalize_rank_rotary_cuda);
    m.impl("normalize_rank_rotary_backward", &normalize_rank_rotary_backward_cuda);
}
