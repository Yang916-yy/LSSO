#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx/detail/shared_memory.hpp>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace lsso_equilibrium {
namespace {

constexpr float kSoftplusOneOffset = 0.5413248546129181f;

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t* values, int64_t index) {
    return static_cast<float>(values[index]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_scalar(
    scalar_t* values,
    int64_t index,
    float value) {
    values[index] = static_cast<scalar_t>(value);
}

__device__ __forceinline__ float softplus_one(float raw) {
    const float shifted = raw + kSoftplusOneOffset;
    return shifted > 20.0f ? shifted : log1pf(expf(shifted));
}

struct ForwardResult {
    at::Tensor output;
    at::Tensor tape;
    at::Tensor pivots;
};

constexpr int kGenericTokenTile = 32;
constexpr int kCrossTokenChunk = 64;
constexpr int kDefaultPhaseCacheEntriesPerDevice = 2;
constexpr size_t kDefaultPhaseCacheBytes = 4 * 1024 * 1024;
// Below this point, an extra partial-Gram launch and global workspace cost more
// than keeping the complete reduction in one system block.
constexpr int kParallelGramMinimumTokenTiles = 32;

template <int rank>
constexpr int generic_frame_materialize_token_tile() {
    // At the larger compiled ranks, one right-hand-side panel spans two
    // adjacent token tiles and amortizes loading the Cholesky factor.
    return rank >= 48 ? 64 : kGenericTokenTile;
}

template <int rank>
__device__ __forceinline__ float generic_inverse_frequency(int pair) {
    constexpr float kLog2PhaseBase = 13.287712379549449f;
    return exp2f(
        -kLog2PhaseBase * static_cast<float>(pair) / static_cast<float>(rank / 2));
}

template <int rank>
__global__ __launch_bounds__(kThreads) void rank_rotary_phase_kernel(
    const float* __restrict__ centered_positions,
    float2* __restrict__ phases,
    int64_t batch_count,
    int64_t length,
    bool batch_specific) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t phase_count =
        (batch_specific ? batch_count : 1) * length * (rank / 2);
    if (linear >= phase_count) {
        return;
    }
    const int64_t phase_token = linear / (rank / 2);
    const int64_t batch = batch_specific ? phase_token / length : 0;
    const int64_t token = batch_specific ? phase_token - batch * length : phase_token;
    const int pair = static_cast<int>(linear - phase_token * (rank / 2));
    const float position = centered_positions == nullptr
        ? static_cast<float>(token) - 0.5f * static_cast<float>(length - 1)
        : centered_positions[batch * (batch_specific ? length : 0) + token];
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(position * generic_inverse_frequency<rank>(pair), &sine, &cosine);
    phases[linear] = make_float2(cosine, sine);
}

template <int rank>
at::Tensor launch_rank_rotary_phase_table(
    const at::Tensor& projected,
    const c10::optional<at::Tensor>& centered_positions,
    FastPathShape shape) {
    const bool batch_specific =
        centered_positions.has_value() && centered_positions->dim() == 2;
    c10::cuda::CUDAGuard guard(projected.device());
    const auto stream = at::cuda::getCurrentCUDAStream(projected.get_device()).stream();
    if (!centered_positions.has_value() &&
        shape.length <=
            static_cast<int64_t>(kDefaultPhaseCacheBytes / (rank * sizeof(float)))) {
        struct CachedPhaseTable {
            at::Tensor phases;
            cudaEvent_t ready;
            cudaStream_t producer_stream;
        };
        static std::mutex mutex;
        static std::map<std::pair<int, int64_t>, CachedPhaseTable> cache;
        static std::unordered_map<int, int> entries_per_device;

        const auto key = std::make_pair(projected.get_device(), shape.length);
        std::lock_guard<std::mutex> lock(mutex);
        const auto existing = cache.find(key);
        if (existing != cache.end()) {
            if (existing->second.producer_stream != stream) {
                C10_CUDA_CHECK(cudaStreamWaitEvent(stream, existing->second.ready, 0));
            }
            return existing->second.phases;
        }
        if (entries_per_device[projected.get_device()] <
            kDefaultPhaseCacheEntriesPerDevice) {
            auto phases = at::empty(
                {shape.length, rank / 2, 2},
                projected.options().dtype(at::kFloat));
            const int64_t phase_count = shape.length * (rank / 2);
            const int64_t blocks = (phase_count + kThreads - 1) / kThreads;
            rank_rotary_phase_kernel<rank><<<blocks, kThreads, 0, stream>>>(
                nullptr,
                reinterpret_cast<float2*>(phases.data_ptr<float>()),
                shape.batch,
                shape.length,
                false);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            cudaEvent_t ready;
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventRecord(ready, stream));
            cache.emplace(key, CachedPhaseTable{phases, ready, stream});
            ++entries_per_device[projected.get_device()];
            return phases;
        }
    }
    auto phases = batch_specific
        ? at::empty(
              {shape.batch, shape.length, rank / 2, 2},
              projected.options().dtype(at::kFloat))
        : at::empty(
              {shape.length, rank / 2, 2},
              projected.options().dtype(at::kFloat));
    const int64_t phase_count =
        (batch_specific ? shape.batch : 1) * shape.length * (rank / 2);
    const int64_t blocks = (phase_count + kThreads - 1) / kThreads;
    rank_rotary_phase_kernel<rank><<<blocks, kThreads, 0, stream>>>(
        centered_positions.has_value() ? centered_positions->data_ptr<float>() : nullptr,
        reinterpret_cast<float2*>(phases.data_ptr<float>()),
        shape.batch,
        shape.length,
        batch_specific);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return phases;
}

template <typename scalar_t, int rank>
__device__ __forceinline__ float2 generic_rotated_relation_from_phase(
    const scalar_t* projected,
    const float2* phases,
    int64_t batch,
    int64_t head,
    int64_t token,
    int pair,
    int64_t length,
    int64_t projected_width,
    int64_t phase_batch_stride,
    float inverse_length_sqrt) {
    const int even_rank = 2 * pair;
    const int64_t relation_base =
        (batch * length + token) * projected_width + head * rank;
    const float even = load_scalar(projected, relation_base + even_rank);
    const float odd = load_scalar(projected, relation_base + even_rank + 1);
    const float2 phase = phases[
        batch * phase_batch_stride + token * (rank / 2) + pair];
    return make_float2(
        (even * phase.x - odd * phase.y) * inverse_length_sqrt,
        (even * phase.y + odd * phase.x) * inverse_length_sqrt);
}

template <int rank>
struct GenericFrameMathDx {
    using Gram = decltype(
        cublasdx::Size<rank, rank, kGenericTokenTile>() +
        cublasdx::Precision<float, float, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::col_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using Cross = decltype(
        cublasdx::Size<rank, kRhsTile, kGenericTokenTile>() +
        cublasdx::Precision<__half, __half, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::col_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using Core = decltype(
        cublasdx::Size<rank, rank, kRhsTile>() +
        cublasdx::Precision<__half, __half, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using Readout = decltype(
        cublasdx::Size<kGenericTokenTile, kRhsTile, rank>() +
        cublasdx::Precision<__half, __half, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using FrameBase = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());

    using Potrf = decltype(
        FrameBase() + cusolverdx::Function<cusolverdx::function::potrf>());

    using FrameTrsm = decltype(
        cusolverdx::Size<generic_frame_materialize_token_tile<rank>(), rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::right>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::transposed>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank>
struct GenericCoreFactorMathDx;

template <int rank>
constexpr size_t generic_frame_shared_bytes() {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;
    using Potrf = typename Traits::Potrf;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    offset = align_shared_offset(offset, alignof(float));
    offset += Potrf::shared_memory_size;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_gram_partial_shared_bytes() {
    using Gram = typename GenericFrameMathDx<rank>::Gram;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_factor_partials_shared_bytes() {
    using Potrf = typename GenericFrameMathDx<rank>::Potrf;
    return Potrf::shared_memory_size;
}

template <int rank>
constexpr size_t generic_cross_shared_bytes() {
    using Cross = typename GenericFrameMathDx<rank>::Cross;
    constexpr size_t a_elements = cublasdx::cosize(Cross::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Cross::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_materialize_shared_bytes() {
    using FrameTrsm = typename GenericFrameMathDx<rank>::FrameTrsm;
    return FrameTrsm::shared_memory_size;
}

template <int rank>
constexpr size_t generic_core_factor_shared_bytes() {
    using Core = typename GenericFrameMathDx<rank>::Core;
    using Getrf = typename GenericCoreFactorMathDx<rank>::Getrf;
    constexpr size_t core_a_elements = cublasdx::cosize(Core::get_layout_smem_a());
    constexpr size_t core_b_elements = cublasdx::cosize(Core::get_layout_smem_b());
    constexpr size_t core_c_elements = cublasdx::cosize(Core::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * core_a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * core_b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * core_c_elements;
    offset = align_shared_offset(offset, alignof(float));
    offset += Getrf::shared_memory_size;
    return offset;
}

template <int rank>
constexpr size_t generic_output_shared_bytes() {
    using Readout = typename GenericFrameMathDx<rank>::Readout;
    constexpr size_t a_elements = cublasdx::cosize(Readout::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Readout::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__half) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_kernel(
    const scalar_t* __restrict__ projected,
    const float2* __restrict__ phases,
    const float* __restrict__ valid_counts,
    float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t phase_batch_stride) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t projected_width = heads * rank + dim;
    const float inverse_length_sqrt = valid_counts == nullptr
        ? rsqrtf(static_cast<float>(length))
        : rsqrtf(valid_counts[batch]);
    float* system_tape = tape + system_index * workspace_layout.stride;
    float* b = system_tape + workspace_layout.b_offset;

    __shared__ float reduction[kThreads];

    float local_scale = 1.0f;
    for (int64_t linear = threadIdx.x; linear < length * (rank / 2);
         linear += blockDim.x) {
        const int64_t token = linear / (rank / 2);
        const int pair = static_cast<int>(linear - token * (rank / 2));
        const float2 relation = generic_rotated_relation_from_phase<scalar_t, rank>(
            projected,
            phases,
            batch,
            head,
            token,
            pair,
            length,
            projected_width,
            phase_batch_stride,
            inverse_length_sqrt);
        b[token * rank + 2 * pair] = relation.x;
        b[token * rank + 2 * pair + 1] = relation.y;
        local_scale = fmaxf(local_scale, fmaxf(fabsf(relation.x), fabsf(relation.y)));
    }
    reduction[threadIdx.x] = local_scale;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = reduction[0];
    const float inverse_scale = 1.0f / scale;
    if (scale != 1.0f) {
        for (int64_t linear = threadIdx.x; linear < length * rank;
             linear += blockDim.x) {
            b[linear] *= inverse_scale;
        }
    }
    if (threadIdx.x == 0) {
        system_tape[workspace_layout.scale_offset] = scale;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_factor_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t system_count,
    int64_t length) {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;
    using Potrf = typename Traits::Potrf;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    float* lower_tape = const_cast<float*>(system_tape) + workspace_layout.l_offset;

    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    float* a_tile = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * a_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto* factor_block = static_cast<unsigned char*>(
        shared.take_bytes(Potrf::shared_memory_size, alignof(float)));
    float* factor = reinterpret_cast<float*>(factor_block);
    auto gram_a = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_b());
    auto gram_c = cublasdx::make_tensor(gemm_output, Gram::get_layout_smem_c());
    __shared__ typename Potrf::status_type info;

    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    const float scale = system_tape[workspace_layout.scale_offset];
    const float regularizer = 1.0f / (scale * scale);
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        gram_c(row, column) = row == column ? regularizer : 0.0f;
    }
    __syncthreads();
    for (int64_t token_start = 0; token_start < length; token_start += kGenericTokenTile) {
        for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
             linear += blockDim.x) {
            const int column = linear / rank;
            const int row = linear - column * rank;
            const int64_t token = token_start + column;
            gram_a(row, column) = token < length ? b[token * rank + row] : 0.0f;
        }
        __syncthreads();
        Gram().execute(1.0f, gram_a, gram_b, 1.0f, gram_c);
        __syncthreads();
    }

    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        factor[row * Potrf::lda + column] = gram_c(row, column);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Potrf().execute(factor, Potrf::lda, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tape[linear] = row >= column
            ? factor[row * Potrf::lda + column]
            : 0.0f;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_gram_partials_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    float* __restrict__ gram_partials,
    int64_t gram_partial_system_stride,
    int64_t system_count,
    int64_t length,
    int64_t token_tiles) {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;

    const int64_t work_index = static_cast<int64_t>(blockIdx.x);
    const int64_t work_count = system_count * token_tiles;
    if (work_index >= work_count) {
        return;
    }
    const int64_t system_index = work_index / token_tiles;
    const int64_t tile_index = work_index - system_index * token_tiles;

    constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    float* a_tile = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * a_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto gram_a = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_b());
    auto gram_c = cublasdx::make_tensor(gemm_output, Gram::get_layout_smem_c());

    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    const int64_t token_start = tile_index * kGenericTokenTile;
    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
         linear += blockDim.x) {
        const int column = linear / rank;
        const int row = linear - column * rank;
        const int64_t token = token_start + column;
        gram_a(row, column) = token < length ? b[token * rank + row] : 0.0f;
    }
    __syncthreads();
    Gram().execute(1.0f, gram_a, gram_b, 0.0f, gram_c);
    __syncthreads();

    float* partial = gram_partials +
        system_index * gram_partial_system_stride +
        tile_index * kPackedLowerElements;
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        if (row >= column) {
            partial[row * (row + 1) / 2 + column] = gram_c(row, column);
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_factor_partials_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    const float* __restrict__ gram_partials,
    int64_t gram_partial_system_stride,
    int64_t system_count,
    int64_t token_tiles) {
    using Traits = GenericFrameMathDx<rank>;
    using Potrf = typename Traits::Potrf;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }

    constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    float* lower_tape = const_cast<float*>(system_tape) + workspace_layout.l_offset;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    auto* factor_block = static_cast<unsigned char*>(
        shared.take_bytes(Potrf::shared_memory_size, alignof(float)));
    float* factor = reinterpret_cast<float*>(factor_block);
    __shared__ typename Potrf::status_type info;

    const float scale = system_tape[workspace_layout.scale_offset];
    const float regularizer = 1.0f / (scale * scale);
    const float* system_partials = gram_partials +
        system_index * gram_partial_system_stride;
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        float value = row == column ? regularizer : 0.0f;
        if (row >= column) {
            const int packed_index = row * (row + 1) / 2 + column;
            for (int64_t tile = 0; tile < token_tiles; ++tile) {
                value += system_partials[tile * kPackedLowerElements + packed_index];
            }
        }
        factor[row * Potrf::lda + column] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Potrf().execute(factor, Potrf::lda, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tape[linear] = row >= column
            ? factor[row * Potrf::lda + column]
            : 0.0f;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_materialize_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
    int64_t system_count,
    int64_t length) {
    using FrameTrsm = typename GenericFrameMathDx<rank>::FrameTrsm;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    if (system_index >= system_count) {
        return;
    }
    constexpr int materialize_token_tile = generic_frame_materialize_token_tile<rank>();
    const int64_t token_start = tile_index * materialize_token_tile;
    const int token_count = static_cast<int>(
        length - token_start < materialize_token_tile
            ? length - token_start
            : materialize_token_tile);
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    const float* lower = system_tape + workspace_layout.l_offset;
    float* frame = const_cast<float*>(system_tape) + frame_offset;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lower_tile, rhs_tile] = cusolverdx::shared_memory::slice<float, float>(
        shared_raw,
        alignof(float),
        rank * FrameTrsm::lda,
        alignof(float));
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tile[row * FrameTrsm::lda + column] = lower[linear];
    }
    for (int linear = threadIdx.x; linear < materialize_token_tile * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const int64_t token = token_start + row;
        rhs_tile[row * FrameTrsm::ldb + column] = 0.0f;
        if (row < token_count) {
            rhs_tile[row * FrameTrsm::ldb + column] = b[token * rank + column];
        }
    }
    __syncthreads();
    FrameTrsm().execute(lower_tile, FrameTrsm::lda, rhs_tile, FrameTrsm::ldb);
    __syncthreads();
    for (int linear = threadIdx.x; linear < token_count * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        frame[(token_start + row) * rank + column] =
            rhs_tile[row * FrameTrsm::ldb + column];
    }
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_cross_state_partials_kernel(
    const scalar_t* __restrict__ projected,
    const float* __restrict__ tape,
    float* __restrict__ partial_cross,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim,
    int64_t rhs_tiles,
    int64_t partial_tile_stride,
    int64_t token_tile_start,
    int64_t token_tile_count) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    const int64_t local_token_tile = static_cast<int64_t>(blockIdx.z);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count || rhs_tile >= rhs_tiles ||
        local_token_tile >= token_tile_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t rhs_start = rhs_tile * kRhsTile;
    const int64_t token_start =
        (token_tile_start + local_token_tile) * kGenericTokenTile;
    const int64_t projected_width = heads * rank + dim;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* frame = system_tape + frame_offset;

    using Cross = typename GenericFrameMathDx<rank>::Cross;
    constexpr size_t a_elements = cublasdx::cosize(Cross::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Cross::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __half* a_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * a_elements, 16));
    __half* b_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto cross_a = cublasdx::make_tensor(a_tile, Cross::get_layout_smem_a());
    auto cross_b = cublasdx::make_tensor(b_tile, Cross::get_layout_smem_b());
    auto cross_c = cublasdx::make_tensor(gemm_output, Cross::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
         linear += blockDim.x) {
        const int column = linear / rank;
        const int row = linear - column * rank;
        const int64_t token = token_start + column;
        cross_a(row, column) = __float2half_rn(
            token < length ? frame[token * rank + row] : 0.0f);
    }
    for (int linear = threadIdx.x; linear < kGenericTokenTile * kRhsTile;
         linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        const int64_t token = token_start + row;
        const int64_t feature = rhs_start + column;
        cross_b(row, column) = __float2half_rn(
            token < length && feature < head_dim
                ? load_scalar(
                      projected,
                      (batch * length + token) * projected_width + heads * rank +
                          head * head_dim + feature)
                : 0.0f);
    }
    __syncthreads();
    Cross().execute(1.0f, cross_a, cross_b, 0.0f, cross_c);
    __syncthreads();
    float* output = partial_cross +
        ((system_index * rhs_tiles + rhs_tile) * partial_tile_stride + local_token_tile) *
            rank * kRhsTile;
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        output[linear] = cross_c(row, column);
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_cross_state_reduce_kernel(
    const float* __restrict__ partial_cross,
    float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t system_count,
    int64_t head_dim,
    int64_t rhs_tiles,
    int64_t partial_tile_stride,
    int64_t token_tile_count,
    bool accumulate) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    if (system_index >= system_count || rhs_tile >= rhs_tiles) {
        return;
    }
    const int64_t rhs_start = rhs_tile * kRhsTile;
    float* system_tape = tape + system_index * workspace_layout.stride;
    float* compact_state = system_tape + workspace_layout.z_offset;
    const float* partial = partial_cross +
        (system_index * rhs_tiles + rhs_tile) * partial_tile_stride * rank * kRhsTile;
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if (rhs_start + column < head_dim) {
            float value = accumulate
                ? compact_state[row * head_dim + rhs_start + column]
                : 0.0f;
            for (int64_t tile = 0; tile < token_tile_count; ++tile) {
                value += partial[tile * rank * kRhsTile + linear];
            }
            compact_state[row * head_dim + rhs_start + column] = value;
        }
    }
}

template <int rank>
__device__ __forceinline__ float generic_factor_value(
    float coordinate,
    int row,
    int column) {
    if (row > column) {
        return coordinate;
    }
    if (row == column) {
        return softplus_one(coordinate);
    }
    return 0.0f;
}

template <int rank>
struct GenericCoreFactorMathDx {
    using Getrf = decltype(
        cusolverdx::Size<rank, rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank>
struct GenericCoreSolveMathDx {
    using Getrs = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrs_partial_pivot>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kRhsTile>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank, bool record_tape>
__global__ __launch_bounds__(kThreads) void generic_core_factor_kernel(
    const float* __restrict__ tape,
    const float* __restrict__ core_base_raw,
    const float* __restrict__ core_drive_weight,
    const float* __restrict__ valid_counts,
    int* __restrict__ pivots,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t heads,
    int64_t head_dim,
    float inverse_length_sqrt) {
    using Core = typename GenericFrameMathDx<rank>::Core;
    using Getrf = typename GenericCoreFactorMathDx<rank>::Getrf;
    using MatrixScalar = typename Getrf::a_data_type;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const int64_t batch = system_index / heads;
    if (valid_counts != nullptr) {
        inverse_length_sqrt = rsqrtf(valid_counts[batch]);
    }
    float* system_tape = const_cast<float*>(tape) + system_index * workspace_layout.stride;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    float* lu = system_tape + workspace_layout.lu_offset;
    int* system_pivots = pivots + system_index * rank;
    float* coordinates = nullptr;
    if constexpr (record_tape) {
        coordinates = system_tape + workspace_layout.coordinates_offset;
    }

    __shared__ typename Getrf::status_type info;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    constexpr size_t core_a_elements = cublasdx::cosize(Core::get_layout_smem_a());
    constexpr size_t core_b_elements = cublasdx::cosize(Core::get_layout_smem_b());
    constexpr size_t core_c_elements = cublasdx::cosize(Core::get_layout_smem_c());
    static_assert(core_c_elements >= rank * rank);
    SharedCursor shared(shared_raw);
    __half* a_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * core_a_elements, 16));
    __half* b_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * core_b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * core_c_elements, 16));
    auto* solver_raw = static_cast<unsigned char*>(
        shared.take_bytes(Getrf::shared_memory_size, alignof(MatrixScalar)));
    auto [matrix, factor_pivots] = cusolverdx::shared_memory::slice<MatrixScalar, int>(
        solver_raw,
        alignof(MatrixScalar), rank * Getrf::lda,
        alignof(int));
    auto core_a = cublasdx::make_tensor(a_tile, Core::get_layout_smem_a());
    auto core_b = cublasdx::make_tensor(b_tile, Core::get_layout_smem_b());
    auto core_c = cublasdx::make_tensor(gemm_output, Core::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < core_c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    for (int64_t feature_start = 0; feature_start < head_dim;
         feature_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = feature_start + column;
            core_a(row, column) = __float2half_rn(
                feature < head_dim
                    ? compact_state[row * head_dim + feature]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            const int64_t feature = feature_start + row;
            core_b(row, column) = __float2half_rn(
                feature < head_dim
                    ? core_drive_weight[(head * head_dim + feature) * rank + column]
                    : 0.0f);
        }
        __syncthreads();
        Core().execute(1.0f, core_a, core_b, 1.0f, core_c);
        __syncthreads();
    }
    // Once the TC16 core product is consumed, reuse solver storage for F and
    // the upper Omega coordinates; gemm_output becomes the assembled K.
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const float coordinate =
            core_base_raw[(head * rank + row) * rank + column] +
            core_c(row, column) * inverse_length_sqrt;
        if constexpr (record_tape) {
            coordinates[linear] = coordinate;
        }
        matrix[row * Getrf::lda + column] = row >= column
            ? generic_factor_value<rank>(coordinate, row, column)
            : coordinate;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * (rank + 1) / 2;
         linear += blockDim.x) {
        int remaining = linear;
        int row = 0;
        while (remaining > row) {
            remaining -= ++row;
        }
        const int column = remaining;
        float gram = 0.0f;
#pragma unroll
        for (int inner = 0; inner < rank; ++inner) {
            if (inner <= column) {
                gram = fmaf(
                    matrix[row * Getrf::lda + inner],
                    matrix[column * Getrf::lda + inner],
                    gram);
            }
        }
        const float upper = row == column
            ? 0.0f
            : matrix[column * Getrf::lda + row];
        gemm_output[row * rank + column] = gram - upper;
        gemm_output[column * rank + row] = gram + upper;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        matrix[row * Getrf::lda + column] =
            gemm_output[linear] + (row == column ? 1.0f : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Getrf().execute(matrix, factor_pivots, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lu[linear] = matrix[row * Getrf::lda + column];
    }
    for (int linear = threadIdx.x; linear < rank; linear += blockDim.x) {
        // GETRF and GETRS use the same one-based pivot contract.
        system_pivots[linear] = factor_pivots[linear];
    }
}

template <int rank>
__global__ __launch_bounds__(kRhsTile) void generic_solve_kernel(
    const float* __restrict__ tape,
    const int* __restrict__ pivots,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t heads,
    int64_t head_dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t rhs_start = rhs_tile * kRhsTile;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    const float* lu = system_tape + workspace_layout.lu_offset;
    float* equilibrium = const_cast<float*>(system_tape) + workspace_layout.u_offset;
    const int* system_pivots = pivots + system_index * rank;

    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lu_tile, rhs, pivot_tile] = cusolverdx::shared_memory::slice<float, float, int>(
        shared_raw,
        16u, rank * Getrs::lda,
        16u, rank * Getrs::ldb,
        16u, rank);
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        lu_tile[linear] = lu[linear];
    }
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        rhs[linear] = rhs_start + column < head_dim
            ? compact_state[row * head_dim + rhs_start + column]
            : 0.0f;
    }
    for (int linear = threadIdx.x; linear < rank; linear += blockDim.x) {
        pivot_tile[linear] = system_pivots[linear];
    }
    __syncthreads();
    Getrs().execute(lu_tile, Getrs::lda, pivot_tile, rhs, Getrs::ldb);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if (rhs_start + column < head_dim) {
            equilibrium[row * head_dim + rhs_start + column] = rhs[linear];
        }
    }
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_output_kernel(
    const scalar_t* __restrict__ projected,
    const float* __restrict__ tape,
    const float* __restrict__ eta_raw,
    scalar_t* __restrict__ output,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t token_start = tile_index * kGenericTokenTile;
    const int token_count = static_cast<int>(
        length - token_start < kGenericTokenTile ? length - token_start : kGenericTokenTile);
    const int64_t projected_width = heads * rank + dim;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* frame = system_tape + frame_offset;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    const float* equilibrium = system_tape + workspace_layout.u_offset;
    const float eta = bounded_complement(eta_raw[head]);

    using Readout = typename GenericFrameMathDx<rank>::Readout;
    constexpr size_t a_elements = cublasdx::cosize(Readout::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Readout::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __half* a_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * a_elements, 16));
    __half* b_tile = reinterpret_cast<__half*>(
        shared.take_bytes(sizeof(__half) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto readout_a = cublasdx::make_tensor(a_tile, Readout::get_layout_smem_a());
    auto readout_b = cublasdx::make_tensor(b_tile, Readout::get_layout_smem_b());
    auto readout_c = cublasdx::make_tensor(gemm_output, Readout::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < kGenericTokenTile * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const int64_t token = token_start + row;
        readout_a(row, column) = __float2half_rn(
            row < token_count ? frame[token * rank + column] : 0.0f);
    }
    __syncthreads();
    for (int64_t rhs_start = 0; rhs_start < head_dim; rhs_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
            gemm_output[linear] = 0.0f;
        }
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = rhs_start + column;
            const float compact_mix = feature < head_dim
                ? 2.0f * equilibrium[row * head_dim + feature] -
                    (1.0f + eta) * compact_state[row * head_dim + feature]
                : 0.0f;
            readout_b(row, column) = __float2half_rn(compact_mix);
        }
        __syncthreads();
        Readout().execute(1.0f, readout_a, readout_b, 0.0f, readout_c);
        __syncthreads();
        for (int linear = threadIdx.x; linear < token_count * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = rhs_start + column;
            if (feature < head_dim) {
                const int64_t token = token_start + row;
                const float content = load_scalar(
                    projected,
                    (batch * length + token) * projected_width + heads * rank +
                        head * head_dim + feature);
                store_scalar(
                    output,
                    (batch * length + token) * dim + head * head_dim + feature,
                    eta * content + readout_c(row, column));
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, int rank, bool record_tape>
void configure_forward_kernel_attributes(int device) {
    static std::mutex mutex;
    static std::unordered_set<int> configured_devices;
    std::lock_guard<std::mutex> lock(mutex);
    if (configured_devices.find(device) != configured_devices.end()) {
        return;
    }

    constexpr size_t frame_shared_bytes = generic_frame_shared_bytes<rank>();
    constexpr size_t gram_partial_shared_bytes =
        generic_frame_gram_partial_shared_bytes<rank>();
    constexpr size_t factor_partials_shared_bytes =
        generic_frame_factor_partials_shared_bytes<rank>();
    constexpr size_t frame_materialize_shared_bytes =
        generic_frame_materialize_shared_bytes<rank>();
    constexpr size_t cross_shared_bytes = generic_cross_shared_bytes<rank>();
    constexpr size_t core_factor_shared_bytes = generic_core_factor_shared_bytes<rank>();
    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;
    constexpr size_t solve_shared_bytes = Getrs::shared_memory_size;
    constexpr size_t output_shared_bytes = generic_output_shared_bytes<rank>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_factor_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_gram_partials_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(gram_partial_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_factor_partials_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(factor_partials_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_materialize_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_materialize_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_cross_state_partials_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(cross_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_core_factor_kernel<rank, record_tape>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(core_factor_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_solve_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(solve_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_output_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(output_shared_bytes)));
    configured_devices.insert(device);
}

template <typename scalar_t, int rank, bool record_tape>
ForwardResult launch_generic_forward(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& centered_positions,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    auto output = at::empty({shape.batch, shape.length, shape.dim}, projected.options());
    const int64_t system_count = shape.batch * shape.heads;
    const auto tape_layout = training_tape_layout(shape);
    const auto workspace_layout = record_tape
        ? forward_workspace_layout(tape_layout)
        : inference_workspace_layout(shape);
    const auto workspace_options = projected.options().dtype(at::kFloat);
    auto workspace = at::empty({system_count, workspace_layout.stride}, workspace_options);
    auto pivot_workspace = at::empty({system_count, rank}, projected.options().dtype(at::kInt));
    at::Tensor tape;
    at::Tensor pivots;
    int64_t frame_offset = workspace_layout.b_offset;
    if constexpr (record_tape) {
        tape = workspace;
        pivots = pivot_workspace;
        frame_offset = tape_layout.p_offset;
    }

    c10::cuda::CUDAGuard guard(projected.device());
    configure_forward_kernel_attributes<scalar_t, rank, record_tape>(
        projected.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(projected.get_device()).stream();
    auto phases = rank_rotary_phase_table_cuda(
        projected, centered_positions, shape);
    const int64_t phase_batch_stride =
        centered_positions.has_value() && centered_positions->dim() == 2
        ? shape.length * (rank / 2)
        : 0;
    generic_frame_kernel<scalar_t, rank><<<system_count, kThreads, 0, stream>>>(
        projected.data_ptr<scalar_t>(),
        reinterpret_cast<const float2*>(phases.data_ptr<float>()),
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        workspace.data_ptr<float>(),
        workspace_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        phase_batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t token_tiles =
        (shape.length + kGenericTokenTile - 1) / kGenericTokenTile;
    if (token_tiles >= kParallelGramMinimumTokenTiles) {
        constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
        at::Tensor gram_partials;
        float* gram_partial_data = nullptr;
        int64_t gram_partial_system_stride =
            token_tiles * kPackedLowerElements;
        if constexpr (record_tape && rank < 64) {
            // At the long-sequence threshold, packed partials for r<64 fit in
            // the not-yet-materialized P[N, R] tape region of every system.
            gram_partial_data = workspace.data_ptr<float>() + tape_layout.p_offset;
            gram_partial_system_stride = workspace_layout.stride;
        } else {
            gram_partials = at::empty(
                {system_count, token_tiles, kPackedLowerElements}, workspace_options);
            gram_partial_data = gram_partials.data_ptr<float>();
        }
        const int64_t gram_blocks = system_count * token_tiles;
        constexpr size_t gram_partial_shared_bytes =
            generic_frame_gram_partial_shared_bytes<rank>();
        generic_frame_gram_partials_kernel<rank><<<
            static_cast<unsigned int>(gram_blocks),
            kThreads,
            gram_partial_shared_bytes,
            stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            gram_partial_data,
            gram_partial_system_stride,
            system_count,
            shape.length,
            token_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        constexpr size_t factor_partials_shared_bytes =
            generic_frame_factor_partials_shared_bytes<rank>();
        generic_frame_factor_partials_kernel<rank><<<
            system_count, kThreads, factor_partials_shared_bytes, stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            gram_partial_data,
            gram_partial_system_stride,
            system_count,
            token_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        constexpr size_t frame_shared_bytes = generic_frame_shared_bytes<rank>();
        generic_frame_factor_kernel<rank><<<
            system_count, kThreads, frame_shared_bytes, stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            system_count,
            shape.length);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    constexpr int materialize_token_tile = generic_frame_materialize_token_tile<rank>();
    const int64_t frame_token_tiles =
        (shape.length + materialize_token_tile - 1) / materialize_token_tile;
    const dim3 frame_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(frame_token_tiles));
    constexpr size_t frame_materialize_shared_bytes = generic_frame_materialize_shared_bytes<rank>();
    generic_frame_materialize_kernel<rank><<<
        frame_grid, kThreads, frame_materialize_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        workspace_layout,
        frame_offset,
        system_count,
        shape.length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t rhs_tiles = (shape.head_dim + kRhsTile - 1) / kRhsTile;
    const dim3 rhs_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(rhs_tiles));
    constexpr size_t cross_shared_bytes = generic_cross_shared_bytes<rank>();
    const int64_t cross_tile_capacity = token_tiles < kCrossTokenChunk
        ? token_tiles
        : kCrossTokenChunk;
    {
        auto partial_cross = at::empty(
            {system_count, rhs_tiles, cross_tile_capacity, rank, kRhsTile},
            workspace_options);
        for (int64_t token_tile_start = 0; token_tile_start < token_tiles;
             token_tile_start += kCrossTokenChunk) {
            const int64_t token_tile_count =
                token_tiles - token_tile_start < kCrossTokenChunk
                ? token_tiles - token_tile_start
                : kCrossTokenChunk;
            const dim3 cross_grid(
                static_cast<unsigned int>(system_count),
                static_cast<unsigned int>(rhs_tiles),
                static_cast<unsigned int>(token_tile_count));
            generic_cross_state_partials_kernel<scalar_t, rank><<<
                cross_grid, kThreads, cross_shared_bytes, stream>>>(
                projected.data_ptr<scalar_t>(),
                workspace.data_ptr<float>(),
                partial_cross.data_ptr<float>(),
                workspace_layout,
                frame_offset,
                shape.batch,
                shape.length,
                shape.heads,
                shape.dim,
                shape.head_dim,
                rhs_tiles,
                cross_tile_capacity,
                token_tile_start,
                token_tile_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            generic_cross_state_reduce_kernel<rank><<<rhs_grid, kThreads, 0, stream>>>(
                partial_cross.data_ptr<float>(),
                workspace.data_ptr<float>(),
                workspace_layout,
                system_count,
                shape.head_dim,
                rhs_tiles,
                cross_tile_capacity,
                token_tile_count,
                token_tile_start != 0);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    constexpr size_t core_factor_shared_bytes = generic_core_factor_shared_bytes<rank>();
    generic_core_factor_kernel<rank, record_tape><<<
        system_count, kThreads, core_factor_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        core_base_raw.data_ptr<float>(),
        core_drive_weight.data_ptr<float>(),
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        pivot_workspace.data_ptr<int>(),
        workspace_layout,
        shape.batch,
        shape.heads,
        shape.head_dim,
        1.0f / std::sqrt(static_cast<float>(shape.length)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;
    constexpr size_t solve_shared_bytes = Getrs::shared_memory_size;
    generic_solve_kernel<rank><<<rhs_grid, kRhsTile, solve_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        pivot_workspace.data_ptr<int>(),
        workspace_layout,
        shape.batch,
        shape.heads,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 output_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(token_tiles));
    constexpr size_t output_shared_bytes = generic_output_shared_bytes<rank>();
    generic_output_kernel<scalar_t, rank><<<
        output_grid, kThreads, output_shared_bytes, stream>>>(
        projected.data_ptr<scalar_t>(),
        workspace.data_ptr<float>(),
        eta_raw.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        workspace_layout,
        frame_offset,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, tape, pivots};
}

template <typename scalar_t, bool record_tape>
ForwardResult dispatch_expanded_forward(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& centered_positions,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    switch (shape.rank) {
        case 16:
            return launch_generic_forward<scalar_t, 16, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                centered_positions, valid_counts, shape);
        case 32:
            return launch_generic_forward<scalar_t, 32, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                centered_positions, valid_counts, shape);
        case 48:
            return launch_generic_forward<scalar_t, 48, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                centered_positions, valid_counts, shape);
        case 64:
            return launch_generic_forward<scalar_t, 64, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                centered_positions, valid_counts, shape);
        default:
            TORCH_CHECK(false, "unreachable supported rank");
    }
}

}  // namespace

at::Tensor rank_rotary_phase_table_cuda(
    const at::Tensor& projected,
    const c10::optional<at::Tensor>& centered_positions,
    FastPathShape shape) {
    switch (shape.rank) {
        case 16:
            return launch_rank_rotary_phase_table<16>(
                projected, centered_positions, shape);
        case 32:
            return launch_rank_rotary_phase_table<32>(
                projected, centered_positions, shape);
        case 48:
            return launch_rank_rotary_phase_table<48>(
                projected, centered_positions, shape);
        case 64:
            return launch_rank_rotary_phase_table<64>(
                projected, centered_positions, shape);
        default:
            TORCH_CHECK(false, "unreachable supported rank");
    }
}

at::Tensor forward_inference_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& centered_positions,
    const c10::optional<at::Tensor>& valid_counts) {
    const auto shape = validate_fast_inputs(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        centered_positions,
        valid_counts);
    c10::cuda::CUDAGuard guard(projected.device());
    (void)supported_sm();
    return dispatch_expanded_forward<float, false>(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        centered_positions,
        valid_counts,
        shape).output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_train_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& centered_positions,
    const c10::optional<at::Tensor>& valid_counts) {
    const auto shape = validate_fast_inputs(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        centered_positions,
        valid_counts);
    c10::cuda::CUDAGuard guard(projected.device());
    (void)supported_sm();
    ForwardResult result = dispatch_expanded_forward<float, true>(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        centered_positions,
        valid_counts,
        shape);
    return {result.output, result.tape, result.pivots};
}

}  // namespace lsso_equilibrium
