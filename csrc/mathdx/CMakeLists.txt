cmake_minimum_required(VERSION 3.24)

project(lsso_mathdx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(CUDAToolkit 13.0 REQUIRED)
find_package(mathdx REQUIRED CONFIG)

add_library(lsso_mathdx SHARED lsso_mathdx.cu)
if(DEFINED LSSO_TORCH_INCLUDE_DIR)
    target_include_directories(lsso_mathdx SYSTEM BEFORE PRIVATE
        "${LSSO_TORCH_INCLUDE_DIR}"
        "${LSSO_TORCH_INCLUDE_DIR}/torch/csrc/api/include"
    )
endif()
target_compile_options(lsso_mathdx PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_options(lsso_mathdx PRIVATE $<DEVICE_LINK:-dlto>)
foreach(cuda_arch IN LISTS LSSO_MATHDX_LTO_ARCHITECTURES)
    target_compile_options(lsso_mathdx PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${cuda_arch},code=lto_${cuda_arch}>"
    )
endforeach()
target_compile_definitions(lsso_mathdx PRIVATE TORCH_EXTENSION_NAME=lsso_mathdx)
target_link_libraries(lsso_mathdx PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    mathdx::cublasdx
    mathdx::cusolverdx_fatbin
)

set_target_properties(lsso_mathdx PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# `native` is useful for iteration. Release builds should pass the complete
# Ampere-and-newer list from scripts/build_mathdx_backend.sh.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET lsso_mathdx PROPERTY CUDA_ARCHITECTURES native)
endif()
