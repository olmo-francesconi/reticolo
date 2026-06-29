#pragma once

// Host/device annotation gating — shared infrastructure.
//
// Core action headers annotate their per-site *formula* functions RETICOLO_HD
// so the SAME function compiles for the CPU backend AND the CUDA device
// kernels (one source of truth — the CPU and GPU implementations cannot
// silently diverge). When the compiler is not nvcc the keywords don't exist,
// so they expand to nothing and the function is an ordinary host function.
//
// Pulls in no CUDA runtime — safe to include from any host-only TU. Use
// RETICOLO_HD for code meant to run on both sides; RETICOLO_DEVICE /
// RETICOLO_HOST for the rare one-sided helper.

#if defined(__CUDACC__)
    #define RETICOLO_HD     __host__ __device__
    #define RETICOLO_DEVICE __device__
    #define RETICOLO_HOST   __host__
#else
    #define RETICOLO_HD
    #define RETICOLO_DEVICE
    #define RETICOLO_HOST
#endif
