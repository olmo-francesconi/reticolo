#pragma once

#include <cmath>
#include <cstddef>

#include <sleef.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
#endif
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    #include <immintrin.h>
#endif

// Portable vectorised libm wrappers backed by Sleef.
//
// Each function takes a flat double buffer of `n` elements and writes the
// per-element sin / cos / sincos into a destination buffer. Internally the
// loop processes `V` doubles at a time using the widest Sleef vector
// function available for the target ISA, then drops to scalar Sleef for the
// tail. Accuracy class is u_lp_10 (1 ULP at 10 digits) — same precision
// budget as `std::sin` / `std::cos` for the argument ranges we use
// (plaquettes / angles in [-pi, pi] after the lattice-update step).
//
// Target dispatch is compile-time only — there is no runtime CPU detect.
// Build with `-march=native` (the project's default Release toolchain) and
// you get the best variant the host CPU supports. Cross-compiling to a
// smaller ISA falls back to a narrower vector or scalar Sleef.

namespace reticolo::math {

namespace detail {

// Sleef dispatch warm-up
//
// Sleef's public scalar entry points (e.g. `Sleef_sin_u10`) read a function
// pointer from a per-symbol dispatch table and tail-call it. On the *first*
// call the slot resolves through `disp_<name>`, which calls an internal
// `cpuSupportsExt` that probes CPU support using process-global
// `signal(SIGILL, ...)` + `sigsetjmp()`. If two OpenMP worker threads enter
// this probe concurrently they trash each other's signal disposition and
// jmp_buf — one thread's `siglongjmp` lands on a stack frame that no longer
// exists on its current thread and the indirect branch goes to whatever
// address happens to live in the corrupted dispatch slot (observed:
// KERN_PROTECTION_FAILURE inside `disp_sincosd1_u10+20`, the return from
// `bl _cpuSupportsExt`).
//
// Force the probe to run once before main, single-threaded, by calling each
// dispatched scalar symbol we use. `SLEEF_CONST` lets the optimizer elide
// unused return values, so we sink them through a `volatile` to keep the
// calls live. The `inline` variable guarantees a single definition across
// translation units.
inline auto const sleef_dispatch_warmup = [] {
    volatile double sink{};
    sink                   = Sleef_sin_u10(0.0);
    sink                   = Sleef_cos_u10(0.0);
    sink                   = Sleef_log_u10(1.0);
    sink                   = Sleef_exp_u10(0.0);
    sink                   = Sleef_sqrt_u05(1.0);
    sink                   = Sleef_acos_u10(1.0);
    Sleef_double2 const sc = Sleef_sincos_u10(0.0);
    sink                   = sc.x + sc.y;
    Sleef_float2 const scf = Sleef_sincosf_u10(0.0F);
    sink                   = static_cast<double>(scf.x + scf.y);
    return static_cast<double>(sink);
}();

}  // namespace detail

// Width of the vector path picked at compile time, in doubles per vector.
// Used by callers that want to size their row scratch buffer to a multiple
// of the vector width — not required for correctness (the helpers handle
// any `n`), only to keep the inner loop free of partial tail iterations.
#ifdef __AVX512F__
inline constexpr std::size_t k_vec_width_d = 8;
#elif defined(__AVX2__) || defined(__AVX__)
inline constexpr std::size_t k_vec_width_d = 4;
#elif defined(__ARM_NEON) || defined(__aarch64__)
inline constexpr std::size_t k_vec_width_d = 2;
#elifdef __SSE2__
inline constexpr std::size_t k_vec_width_d = 2;
#else
inline constexpr std::size_t k_vec_width_d = 1;
#endif

// Width of the vector path in floats per vector (= 2× the double width). Used
// to size site-batch kernels so a float batch fills whole SIMD registers.
#ifdef __AVX512F__
inline constexpr std::size_t k_vec_width_f = 16;
#elif defined(__AVX2__) || defined(__AVX__)
inline constexpr std::size_t k_vec_width_f = 8;
#elif defined(__ARM_NEON) || defined(__aarch64__)
inline constexpr std::size_t k_vec_width_f = 4;
#elifdef __SSE2__
inline constexpr std::size_t k_vec_width_f = 4;
#else
inline constexpr std::size_t k_vec_width_f = 1;
#endif

// --------------- cos_batch ---------------------------------------------------

inline void cos_batch(double* dst, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_cosd8_u10avx512f(v));
    }
#elifdef __AVX2__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_cosd4_u10avx2(v));
    }
#elifdef __AVX__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_cosd4_u10avx(v));
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 2` in the for header
    for (; i + 2 <= n; i += 2) {
        float64x2_t const v = vld1q_f64(src + i);
        vst1q_f64(dst + i, Sleef_cosd2_u10advsimd(v));
    }
#elifdef __SSE2__
    for (; i + 2 <= n; i += 2) {
        __m128d const v = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, Sleef_cosd2_u10sse2(v));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = Sleef_cos_u10(src[i]);
    }
}

// --------------- sin_batch ---------------------------------------------------

inline void sin_batch(double* dst, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_sind8_u10avx512f(v));
    }
#elifdef __AVX2__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_sind4_u10avx2(v));
    }
#elifdef __AVX__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_sind4_u10avx(v));
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 2` in the for header
    for (; i + 2 <= n; i += 2) {
        float64x2_t const v = vld1q_f64(src + i);
        vst1q_f64(dst + i, Sleef_sind2_u10advsimd(v));
    }
#elifdef __SSE2__
    for (; i + 2 <= n; i += 2) {
        __m128d const v = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, Sleef_sind2_u10sse2(v));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = Sleef_sin_u10(src[i]);
    }
}

// --------------- acos_batch --------------------------------------------------
//
// Vectorised acos with 1 ULP at 10 digits, mirroring sin/cos. Used by the
// SU(3) exp slab kernel to batch the per-site `acos(c0/c0_max)`.

inline void acos_batch(double* dst, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_acosd8_u10avx512f(v));
    }
#elifdef __AVX2__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_acosd4_u10avx2(v));
    }
#elifdef __AVX__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_acosd4_u10avx(v));
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 2` in the for header
    for (; i + 2 <= n; i += 2) {
        float64x2_t const v = vld1q_f64(src + i);
        vst1q_f64(dst + i, Sleef_acosd2_u10advsimd(v));
    }
#elifdef __SSE2__
    for (; i + 2 <= n; i += 2) {
        __m128d const v = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, Sleef_acosd2_u10sse2(v));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = Sleef_acos_u10(src[i]);
    }
}

// --------------- sincos_batch ------------------------------------------------
//
// Computes both sin and cos in a single Sleef call per chunk. ~1.4× faster
// than calling sin_batch + cos_batch separately on hot paths that need both
// (e.g. CompactU1 s_full uses cos, compute_force_and_kick uses sin — but the
// kick that follows s_full's same lattice could share a sincos pass).

inline void
sincos_batch(double* dst_sin, double* dst_cos, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= n; i += 8) {
        __m512d const v          = _mm512_loadu_pd(src + i);
        Sleef___m512d_2 const sc = Sleef_sincosd8_u10avx512f(v);
        _mm512_storeu_pd(dst_sin + i, sc.x);
        _mm512_storeu_pd(dst_cos + i, sc.y);
    }
#elifdef __AVX2__
    for (; i + 4 <= n; i += 4) {
        __m256d const v          = _mm256_loadu_pd(src + i);
        Sleef___m256d_2 const sc = Sleef_sincosd4_u10avx2(v);
        _mm256_storeu_pd(dst_sin + i, sc.x);
        _mm256_storeu_pd(dst_cos + i, sc.y);
    }
#elifdef __AVX__
    for (; i + 4 <= n; i += 4) {
        __m256d const v          = _mm256_loadu_pd(src + i);
        Sleef___m256d_2 const sc = Sleef_sincosd4_u10avx(v);
        _mm256_storeu_pd(dst_sin + i, sc.x);
        _mm256_storeu_pd(dst_cos + i, sc.y);
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 2` in the for header
    for (; i + 2 <= n; i += 2) {
        float64x2_t const v          = vld1q_f64(src + i);
        Sleef_float64x2_t_2 const sc = Sleef_sincosd2_u10advsimd(v);
        vst1q_f64(dst_sin + i, sc.x);
        vst1q_f64(dst_cos + i, sc.y);
    }
#elifdef __SSE2__
    for (; i + 2 <= n; i += 2) {
        __m128d const v          = _mm_loadu_pd(src + i);
        Sleef___m128d_2 const sc = Sleef_sincosd2_u10sse2(v);
        _mm_storeu_pd(dst_sin + i, sc.x);
        _mm_storeu_pd(dst_cos + i, sc.y);
    }
#endif
    for (; i < n; ++i) {
        Sleef_double_2 const sc = Sleef_sincos_u10(src[i]);
        dst_sin[i]              = sc.x;
        dst_cos[i]              = sc.y;
    }
}

// --------------- sincos_batch (float) ----------------------------------------
//
// Single-precision twin of the double sincos_batch — 4-wide on NEON/SSE,
// 8-wide on AVX/AVX2, 16-wide on AVX-512. Lets the mixed-precision gauge
// drift (expi_lmul_slab<float>) run its transcendentals at the same lane
// count as its matmul instead of widening to double.

inline void sincos_batch(float* dst_sin, float* dst_cos, float const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= n; i += 16) {
        __m512 const v          = _mm512_loadu_ps(src + i);
        Sleef___m512_2 const sc = Sleef_sincosf16_u10avx512f(v);
        _mm512_storeu_ps(dst_sin + i, sc.x);
        _mm512_storeu_ps(dst_cos + i, sc.y);
    }
#elifdef __AVX2__
    for (; i + 8 <= n; i += 8) {
        __m256 const v          = _mm256_loadu_ps(src + i);
        Sleef___m256_2 const sc = Sleef_sincosf8_u10avx2(v);
        _mm256_storeu_ps(dst_sin + i, sc.x);
        _mm256_storeu_ps(dst_cos + i, sc.y);
    }
#elifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 const v          = _mm256_loadu_ps(src + i);
        Sleef___m256_2 const sc = Sleef_sincosf8_u10avx(v);
        _mm256_storeu_ps(dst_sin + i, sc.x);
        _mm256_storeu_ps(dst_cos + i, sc.y);
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 4` in the for header
    for (; i + 4 <= n; i += 4) {
        float32x4_t const v          = vld1q_f32(src + i);
        Sleef_float32x4_t_2 const sc = Sleef_sincosf4_u10advsimd(v);
        vst1q_f32(dst_sin + i, sc.x);
        vst1q_f32(dst_cos + i, sc.y);
    }
#elifdef __SSE2__
    for (; i + 4 <= n; i += 4) {
        __m128 const v          = _mm_loadu_ps(src + i);
        Sleef___m128_2 const sc = Sleef_sincosf4_u10sse2(v);
        _mm_storeu_ps(dst_sin + i, sc.x);
        _mm_storeu_ps(dst_cos + i, sc.y);
    }
#endif
    for (; i < n; ++i) {
        Sleef_float_2 const sc = Sleef_sincosf_u10(src[i]);
        dst_sin[i]             = sc.x;
        dst_cos[i]             = sc.y;
    }
}

// --------------- exp_batch ---------------------------------------------------
//
// Vectorised exp with 1 ULP at 10 digits, mirroring sin/cos. Used by the
// checkerboard Metropolis sweep to batch the per-site exp(-ds) acceptance
// weight off the scalar critical path. Underflows to 0 / overflows to inf for
// out-of-range arguments — callers gate on the ds<=0 short-circuit, so those
// values are computed but never read.

inline void exp_batch(double* dst, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_expd8_u10avx512f(v));
    }
#elifdef __AVX2__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_expd4_u10avx2(v));
    }
#elifdef __AVX__
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_expd4_u10avx(v));
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // NOLINTNEXTLINE(bugprone-infinite-loop) — increment is `i += 2` in the for header
    for (; i + 2 <= n; i += 2) {
        float64x2_t const v = vld1q_f64(src + i);
        vst1q_f64(dst + i, Sleef_expd2_u10advsimd(v));
    }
#elifdef __SSE2__
    for (; i + 2 <= n; i += 2) {
        __m128d const v = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, Sleef_expd2_u10sse2(v));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = Sleef_exp_u10(src[i]);
    }
}

}  // namespace reticolo::math
