#pragma once

// Host/device annotation gating.
//
// The CUDA backend shares per-site *formula* code with the CPU core (a Phi4
// force functor, the SU(N) matmul, etc.). Those functor headers are compiled
// both by nvcc (for the device kernels) and by the host compiler (because an
// app that includes <reticolo/cuda/hmc.hpp> instantiates `cuda::Hmc`, which
// holds the POD action by value). When the host compiler is NOT nvcc, the
// `__host__ __device__` keywords don't exist, so they must expand to nothing.
//
// Use RETICOLO_HD on anything meant to run on both sides; RETICOLO_DEVICE /
// RETICOLO_HOST for the rare one-sided helper. This header pulls in no CUDA
// runtime — it is safe to include from a host-only translation unit.

#if defined(__CUDACC__)
    #define RETICOLO_HD     __host__ __device__
    #define RETICOLO_DEVICE __device__
    #define RETICOLO_HOST   __host__
#else
    #define RETICOLO_HD
    #define RETICOLO_DEVICE
    #define RETICOLO_HOST
#endif
