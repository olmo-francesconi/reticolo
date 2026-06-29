# reticolo::cuda — the only target compiled by nvcc and the only one that may
# include <cuda_runtime.h> / cuRAND / cub. CUDA headers must never leak into
# reticolo::core, mirroring how reticolo::io quarantines <hdf5.h>.
#
# Phase 0: toolchain skeleton + a self-test TU. The target is intentionally
# self-contained and does NOT yet link reticolo::core — pulling core action
# headers into an nvcc TU is blocked until M1 isolates <sleef.h> from the math
# (see docs/cuda_hmc_roadmap.md). It carries only its own public headers under
# include/reticolo/cuda/.

find_package(CUDAToolkit REQUIRED)

add_library(reticolo_cuda STATIC selftest.cu)
add_library(reticolo::cuda ALIAS reticolo_cuda)

target_include_directories(reticolo_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CUDA runtime is propagated to host consumers that link the static archive.
target_link_libraries(reticolo_cuda PUBLIC CUDA::cudart)

set_target_properties(reticolo_cuda PROPERTIES
    CUDA_STANDARD          20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)
</content>
