#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
mathdx_root="${MATHDX_ROOT:-/opt/nvidia/nvidia-mathdx-26.06.0-cuda13/nvidia/mathdx/26.06}"
cuda_root="${CUDA_HOME:-/usr/local/cuda-13.0}"
build_dir="${LSSO_MATHDX_BUILD_DIR:-${HOME}/.cache/lsso-mathdx-build}"
artifact_dir="${LSSO_MATHDX_ARTIFACT_DIR:-${repo_root}/build/mathdx/lib}"
release_architectures="80-real;86-real;87-real;89-real;90-real;100-real;120-real"
release_torch_architectures="8.0;8.6;8.7;8.9;9.0;10.0;12.0"
if [[ "${LSSO_MATHDX_RELEASE:-0}" == "1" ]]; then
    architectures="${LSSO_CUDA_ARCHITECTURES:-${release_architectures}}"
    lto_architectures="${LSSO_MATHDX_LTO_ARCHITECTURES:-80;86;87;89;90;100;120}"
    torch_architectures="${LSSO_TORCH_CUDA_ARCH_LIST:-${release_torch_architectures}}"
else
    architectures="${LSSO_CUDA_ARCHITECTURES:-native}"
    lto_architectures="${LSSO_MATHDX_LTO_ARCHITECTURES:-}"
    torch_architectures="${LSSO_TORCH_CUDA_ARCH_LIST:-}"
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Python virtual environment not found: ${python_bin}" >&2
    exit 1
fi
torch_abi_tag="$(${python_bin} -c 'import torch; print(torch.__version__.split("+")[0])')"
if [[ -z "${LSSO_MATHDX_BUILD_DIR:-}" ]]; then
    build_dir="${HOME}/.cache/lsso-mathdx-build-${torch_abi_tag}"
fi
torch_include_cache="${LSSO_TORCH_INCLUDE_CACHE:-${HOME}/.cache/lsso-torch-include-${torch_abi_tag}}"
if [[ ! -f "${mathdx_root}/lib/cmake/mathdx/mathdx-config.cmake" ]]; then
    echo "MathDx CMake package not found under: ${mathdx_root}" >&2
    exit 1
fi

mkdir -p "${torch_include_cache}"
if [[ ! -f "${torch_include_cache}/torch/extension.h" ]]; then
    torch_include="$(${python_bin} -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')"
    cp -a "${torch_include}/." "${torch_include_cache}/"
fi

if [[ ! -f "${build_dir}/build.ninja" || "${LSSO_MATHDX_RECONFIGURE:-0}" == "1" ]]; then
    mapfile -t torch_config < <(${python_bin} - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
major, minor = torch.cuda.get_device_capability()
print(f"{major}{minor}")
PY
    )
    torch_prefix="${torch_config[0]}"
    if [[ -z "${lto_architectures}" ]]; then
        lto_architectures="${torch_config[1]}"
    fi
    cmake_args=(
        -S "${repo_root}/csrc/mathdx"
        -B "${build_dir}"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc"
        -DCMAKE_CUDA_ARCHITECTURES="${architectures}"
        -DLSSO_MATHDX_LTO_ARCHITECTURES="${lto_architectures}"
        -DCMAKE_PREFIX_PATH="${torch_prefix};${mathdx_root}/lib/cmake"
        -DLSSO_TORCH_INCLUDE_DIR="${torch_include_cache}"
        -DCUDAToolkit_ROOT="${cuda_root}"
    )
    if [[ -n "${torch_architectures}" ]]; then
        cmake_args+=("-DTORCH_CUDA_ARCH_LIST=${torch_architectures}")
    fi
    cmake "${cmake_args[@]}"
fi
cmake --build "${build_dir}" --parallel

mkdir -p "${artifact_dir}"
cp "${build_dir}/lib/lsso_mathdx.so" "${artifact_dir}/lsso_mathdx.so"
echo "Built ${artifact_dir}/lsso_mathdx.so"
