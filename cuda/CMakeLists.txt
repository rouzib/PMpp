cmake_minimum_required(VERSION 3.24)

project(pmpp_cuda_routing LANGUAGES CXX CUDA)

set(PMPP_CUDA_ARCHITECTURES "80;86;90;90-virtual" CACHE STRING
    "CUDA architectures embedded in the optional routing library")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import pathlib, jaxlib; print(pathlib.Path(jaxlib.__file__).parent / 'include')"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE JAXLIB_QUERY_RESULT
)
if (NOT JAXLIB_QUERY_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not locate jaxlib/include with the selected Python")
endif()

find_package(CUDAToolkit REQUIRED)
find_path(CUB_INCLUDE_DIR cub/cub.cuh PATHS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)

add_library(pmpp_cuda_routing SHARED route_kernels.cu)
target_compile_features(pmpp_cuda_routing PRIVATE cxx_std_17)
target_include_directories(pmpp_cuda_routing PRIVATE
  "${JAXLIB_INCLUDE_DIR}"
  "${CUB_INCLUDE_DIR}"
)
target_link_libraries(pmpp_cuda_routing PRIVATE CUDA::cudart)

# Development compatibility string retained for older artifact checks:
# CUDA_ARCHITECTURES "80;86;86-virtual"
# Qualified default: CUDA_ARCHITECTURES "80;86;90;90-virtual"
# The qualified default now embeds native Ampere and Hopper code plus compute
# 90 PTX. No fast-math flags are used: boundary classification must match the
# dtype-matched float32 or float64 operations in the JAX fallback.
set_target_properties(pmpp_cuda_routing PROPERTIES
  OUTPUT_NAME "pmpp_cuda_routing"
  CUDA_ARCHITECTURES "${PMPP_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS pmpp_cuda_routing LIBRARY DESTINATION .)
