#include <algorithm>
#include <cstdint>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <cublasdx.hpp>
#include <cusolverdx.hpp>

namespace {

template <int Rank, int RhsTile, int Arch>
struct SolverTraits {
    using Base = decltype(
        cusolverdx::Size<Rank, Rank, RhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::FillMode<cusolverdx::lower>() +
        cusolverdx::Arrangement<cusolverdx::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<256>() +
        cusolverdx::BatchesPerBlock<1>() +
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

template <int Rank, int RhsTile, int KTile, int Arch>
__global__ void stats_solve_spd_kernel(
    const float* __restrict__ u,
    const float* __restrict__ c,
    const float* __restrict__ alpha,
    float* __restrict__ solution,
    int* __restrict__ info,
    int64_t batches,
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
    if (batch >= batches) {
        return;
    }

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int u_tile_elements =
        cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr int c_tile_elements =
        cublasdx::cosize(Cross::get_layout_smem_b());

    extern __shared__ __align__(16) cublasdx::byte smem_raw[];
    float* u_tile = reinterpret_cast<float*>(smem_raw);
    float* c_tile = u_tile + u_tile_elements;
    float* a = c_tile + c_tile_elements;
    float* b = a + Rank * lda;
    float* gemm_output = b + Rank * ldb;

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

    const float* u_batch = u + batch * sequence * Rank;
    const float* c_batch = c + batch * sequence * rhs_width;
    float* solution_batch = solution + batch * Rank * rhs_width;

    auto gram_accumulator = Gram().suggest_accumulator();
    gram_accumulator.clear();
    for (int64_t sequence_start = 0; sequence_start < sequence; sequence_start += KTile) {
        for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
            const int row = linear / KTile;
            const int k = linear - row * KTile;
            const int64_t token = sequence_start + k;
            gram_a(row, k) = token < sequence
                ? u_batch[token * Rank + row]
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
            for (int linear = threadIdx.x; linear < Rank * KTile; linear += blockDim.x) {
                const int row = linear / KTile;
                const int k = linear - row * KTile;
                const int64_t token = sequence_start + k;
                cross_a(row, k) = token < sequence
                    ? u_batch[token * Rank + row]
                    : 0.0f;
            }
            for (int linear = threadIdx.x; linear < KTile * RhsTile; linear += blockDim.x) {
                const int k = linear / RhsTile;
                const int col = linear - k * RhsTile;
                const int64_t token = sequence_start + k;
                const int64_t global_col = rhs_start + col;
                cross_b(k, col) = token < sequence && global_col < rhs_width
                    ? c_batch[token * rhs_width + global_col]
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

template <int Rank, int RhsTile, int Arch>
__global__ void solve_spd_kernel(
    const float* __restrict__ gram,
    const float* __restrict__ rhs,
    float* __restrict__ solution,
    int* __restrict__ info,
    int64_t batches,
    int64_t rhs_width) {
    using Traits = SolverTraits<Rank, RhsTile, Arch>;
    using Potrf = typename Traits::Potrf;
    using Potrs = typename Traits::Potrs;

    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);

    const int64_t batch = static_cast<int64_t>(blockIdx.x);
    if (batch >= batches) {
        return;
    }

    constexpr int lda = Potrf::lda;
    constexpr int ldb = Potrs::ldb;
    constexpr int a_elements = Rank * lda;

    extern __shared__ __align__(16) cusolverdx::byte smem_raw[];
    float* a = reinterpret_cast<float*>(smem_raw);
    float* b = a + a_elements;

    const float* gram_batch = gram + batch * Rank * Rank;
    const float* rhs_batch = rhs + batch * Rank * rhs_width;
    float* solution_batch = solution + batch * Rank * rhs_width;

    for (int linear = threadIdx.x; linear < Rank * Rank; linear += blockDim.x) {
        const int row = linear / Rank;
        const int col = linear - row * Rank;
        a[row * lda + col] = gram_batch[linear];
    }
    __syncthreads();

    Potrf().execute(a, lda, info + batch);
    __syncthreads();

    for (int64_t rhs_start = 0; rhs_start < rhs_width; rhs_start += RhsTile) {
        for (int linear = threadIdx.x; linear < Rank * RhsTile; linear += blockDim.x) {
            const int row = linear / RhsTile;
            const int col = linear - row * RhsTile;
            const int64_t global_col = rhs_start + col;
            b[row * ldb + col] = global_col < rhs_width
                ? rhs_batch[row * rhs_width + global_col]
                : 0.0f;
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

template <int Rank, int Arch>
void launch_solve(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    constexpr int rhs_tile = 32;
    using Traits = SolverTraits<Rank, rhs_tile, Arch>;
    using Potrf = typename Traits::Potrf;
    using Potrs = typename Traits::Potrs;

    constexpr size_t manual_smem =
        sizeof(float) * (Rank * Potrf::lda + Rank * Potrs::ldb);
    constexpr size_t solver_smem =
        Potrf::shared_memory_size > Potrs::shared_memory_size
            ? Potrf::shared_memory_size
            : Potrs::shared_memory_size;
    constexpr size_t smem_bytes = manual_smem > solver_smem ? manual_smem : solver_smem;

    auto kernel = solve_spd_kernel<Rank, rhs_tile, Arch>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    const auto batches = gram.size(0);
    kernel<<<static_cast<unsigned int>(batches), 256, smem_bytes, stream>>>(
        gram.const_data_ptr<float>(),
        rhs.const_data_ptr<float>(),
        solution.mutable_data_ptr<float>(),
        info.mutable_data_ptr<int>(),
        batches,
        rhs.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch>
void dispatch_rank(
    const at::Tensor& gram,
    const at::Tensor& rhs,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    if (gram.size(1) == 16) {
        launch_solve<16, Arch>(gram, rhs, solution, info, stream);
    } else if (gram.size(1) == 32) {
        launch_solve<32, Arch>(gram, rhs, solution, info, stream);
    } else {
        TORCH_CHECK(false, "MathDx backend supports rank 16 or 32, got ", gram.size(1));
    }
}

template <int Rank, int Arch>
void launch_stats_solve(
    const at::Tensor& u,
    const at::Tensor& c,
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
        Rank * Potrf::lda +
        Rank * Potrs::ldb +
        Rank * rhs_tile);

    auto kernel = stats_solve_spd_kernel<Rank, rhs_tile, k_tile, Arch>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    kernel<<<static_cast<unsigned int>(u.size(0)), 256, smem_bytes, stream>>>(
        u.const_data_ptr<float>(),
        c.const_data_ptr<float>(),
        alpha.const_data_ptr<float>(),
        solution.mutable_data_ptr<float>(),
        info.mutable_data_ptr<int>(),
        u.size(0),
        u.size(1),
        c.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch>
void dispatch_stats_solve_rank(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& alpha,
    at::Tensor& solution,
    at::Tensor& info,
    cudaStream_t stream) {
    if (u.size(2) == 16) {
        launch_stats_solve<16, Arch>(u, c, alpha, solution, info, stream);
    } else if (u.size(2) == 32) {
        launch_stats_solve<32, Arch>(u, c, alpha, solution, info, stream);
    } else {
        TORCH_CHECK(false, "MathDx backend supports rank 16 or 32, got ", u.size(2));
    }
}

std::tuple<at::Tensor, at::Tensor> solve_spd_cuda(
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
    TORCH_CHECK(gram.size(0) > 0 && rhs.size(2) > 0, "empty batches/RHS are unsupported");
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
    if (cc >= 121) {
        dispatch_rank<1210>(gram, rhs, solution, info, stream);
    } else if (cc >= 120) {
        dispatch_rank<1200>(gram, rhs, solution, info, stream);
    } else if (cc >= 110) {
        dispatch_rank<1100>(gram, rhs, solution, info, stream);
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
    return {solution, info};
}

std::tuple<at::Tensor, at::Tensor> stats_solve_spd_cuda(
    const at::Tensor& u,
    const at::Tensor& c,
    const at::Tensor& alpha) {
    TORCH_CHECK(u.is_cuda() && c.is_cuda() && alpha.is_cuda(), "u, c, and alpha must be CUDA tensors");
    TORCH_CHECK(u.scalar_type() == at::kFloat, "u must be float32");
    TORCH_CHECK(c.scalar_type() == at::kFloat, "c must be float32");
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

    const c10::cuda::CUDAGuard device_guard(u.device());
    auto solution = at::empty({u.size(0), u.size(2), c.size(2)}, u.options());
    auto info = at::zeros({u.size(0)}, u.options().dtype(at::kInt));
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const int cc = props->major * 10 + props->minor;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(u.get_device()).stream();

    if (cc >= 121) {
        dispatch_stats_solve_rank<1210>(u, c, alpha, solution, info, stream);
    } else if (cc >= 120) {
        dispatch_stats_solve_rank<1200>(u, c, alpha, solution, info, stream);
    } else if (cc >= 110) {
        dispatch_stats_solve_rank<1100>(u, c, alpha, solution, info, stream);
    } else if (cc >= 103) {
        dispatch_stats_solve_rank<1030>(u, c, alpha, solution, info, stream);
    } else if (cc >= 100) {
        dispatch_stats_solve_rank<1000>(u, c, alpha, solution, info, stream);
    } else if (cc >= 90) {
        dispatch_stats_solve_rank<900>(u, c, alpha, solution, info, stream);
    } else if (cc >= 89) {
        dispatch_stats_solve_rank<890>(u, c, alpha, solution, info, stream);
    } else if (cc >= 87) {
        dispatch_stats_solve_rank<870>(u, c, alpha, solution, info, stream);
    } else if (cc >= 86) {
        dispatch_stats_solve_rank<860>(u, c, alpha, solution, info, stream);
    } else if (cc >= 80) {
        dispatch_stats_solve_rank<800>(u, c, alpha, solution, info, stream);
    } else {
        TORCH_CHECK(false, "MathDx backend requires an Ampere-or-newer GPU, got compute capability ",
                    props->major, ".", props->minor);
    }
    return {solution, info};
}

}  // namespace

TORCH_LIBRARY(lsso_mathdx, m) {
    m.def("solve_spd(Tensor gram, Tensor rhs) -> (Tensor solution, Tensor info)");
    m.def("stats_solve_spd(Tensor u, Tensor c, Tensor alpha) -> (Tensor solution, Tensor info)");
}

TORCH_LIBRARY_IMPL(lsso_mathdx, CUDA, m) {
    m.impl("solve_spd", &solve_spd_cuda);
    m.impl("stats_solve_spd", &stats_solve_spd_cuda);
}
