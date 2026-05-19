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

// =============================================================================
//  Portable vectorised libm wrappers backed by Sleef.
//
//  Each function takes a flat double buffer of `n` elements and writes the
//  per-element sin / cos / sincos into a destination buffer. Internally the
//  loop processes `V` doubles at a time using the widest Sleef vector
//  function available for the target ISA, then drops to scalar Sleef for the
//  tail. Accuracy class is u_lp_10 (1 ULP at 10 digits) — same precision
//  budget as `std::sin` / `std::cos` for the argument ranges we use
//  (plaquettes / angles in [-pi, pi] after the lattice-update step).
//
//  Target dispatch is compile-time only — there is no runtime CPU detect.
//  Build with `-march=native` (the project's default Release toolchain) and
//  you get the best variant the host CPU supports. Cross-compiling to a
//  smaller ISA falls back to a narrower vector or scalar Sleef.
// =============================================================================

namespace reticolo::math {

// Width of the vector path picked at compile time, in doubles per vector.
// Used by callers that want to size their row scratch buffer to a multiple
// of the vector width — not required for correctness (the helpers handle
// any `n`), only to keep the inner loop free of partial tail iterations.
#if defined(__AVX512F__)
inline constexpr std::size_t k_vec_width_d = 8;
#elif defined(__AVX2__) || defined(__AVX__)
inline constexpr std::size_t k_vec_width_d = 4;
#elif defined(__ARM_NEON) || defined(__aarch64__)
inline constexpr std::size_t k_vec_width_d = 2;
#elif defined(__SSE2__)
inline constexpr std::size_t k_vec_width_d = 2;
#else
inline constexpr std::size_t k_vec_width_d = 1;
#endif

// --------------- cos_batch ---------------------------------------------------

inline void cos_batch(double* dst, double const* src, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_cosd8_u10avx512f(v));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_cosd4_u10avx2(v));
    }
#elif defined(__AVX__)
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
#elif defined(__SSE2__)
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
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_sind8_u10avx512f(v));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_sind4_u10avx2(v));
    }
#elif defined(__AVX__)
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
#elif defined(__SSE2__)
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
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8) {
        __m512d const v = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, Sleef_acosd8_u10avx512f(v));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d const v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, Sleef_acosd4_u10avx2(v));
    }
#elif defined(__AVX__)
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
#elif defined(__SSE2__)
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
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8) {
        __m512d const v          = _mm512_loadu_pd(src + i);
        Sleef___m512d_2 const sc = Sleef_sincosd8_u10avx512f(v);
        _mm512_storeu_pd(dst_sin + i, sc.x);
        _mm512_storeu_pd(dst_cos + i, sc.y);
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d const v          = _mm256_loadu_pd(src + i);
        Sleef___m256d_2 const sc = Sleef_sincosd4_u10avx2(v);
        _mm256_storeu_pd(dst_sin + i, sc.x);
        _mm256_storeu_pd(dst_cos + i, sc.y);
    }
#elif defined(__AVX__)
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
#elif defined(__SSE2__)
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

}  // namespace reticolo::math
