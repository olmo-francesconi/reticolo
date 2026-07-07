#pragma once

#include <reticolo/core/hd.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>

// Counter-based RNG (Philox-4×32-10, Salmon et al. 2011 / Random123) as a pure
// RETICOLO_HD bijection — one source of truth shared by the CPU PhiloxRng and
// the CUDA momentum sampler. Stateless: output = f(key, counter), so it is
// embarrassingly parallel, reproducible independent of thread count / order,
// and a checkpoint is just the counter. Integer-only, so the uniform stream is
// bit-identical CPU↔GPU; the Box-Muller normals match only to ~1 ULP (the
// transcendental differs: libdevice vs libm), the same caveat the CPU RNGs
// already have between their scalar and SIMD paths.

namespace reticolo {

struct Philox4x32 {
    static constexpr std::uint32_t kMul0  = 0xD2511F53U;
    static constexpr std::uint32_t kMul1  = 0xCD9E8D57U;
    static constexpr std::uint32_t kWeyl0 = 0x9E3779B9U;  // golden ratio
    static constexpr std::uint32_t kWeyl1 = 0xBB67AE85U;  // sqrt(2) frac

    // Plain C-array members (not std::array): std::array::operator[] is a
    // constexpr __host__ function that nvcc refuses to call from device code
    // (warning #20015-D) without --expt-relaxed-constexpr, silently producing
    // wrong values on the GPU. The builtin subscript on a C array is
    // device-native. Matches DeviceTopology's layout style.
    struct U32x4 {
        std::uint32_t v[4];
        [[nodiscard]] RETICOLO_HD std::uint32_t& operator[](int i) { return v[i]; }
        [[nodiscard]] RETICOLO_HD std::uint32_t operator[](int i) const { return v[i]; }
    };
    struct U32x2 {
        std::uint32_t v[2];
        [[nodiscard]] RETICOLO_HD std::uint32_t& operator[](int i) { return v[i]; }
        [[nodiscard]] RETICOLO_HD std::uint32_t operator[](int i) const { return v[i]; }
    };

    [[nodiscard]] RETICOLO_HD static U32x4 single_round(U32x4 ctr, U32x2 key) {
        std::uint64_t const p0  = static_cast<std::uint64_t>(kMul0) * ctr[0];
        std::uint64_t const p1  = static_cast<std::uint64_t>(kMul1) * ctr[2];
        std::uint32_t const hi0 = static_cast<std::uint32_t>(p0 >> 32U);
        std::uint32_t const lo0 = static_cast<std::uint32_t>(p0);
        std::uint32_t const hi1 = static_cast<std::uint32_t>(p1 >> 32U);
        std::uint32_t const lo1 = static_cast<std::uint32_t>(p1);
        return U32x4{hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
    }

    // 10 rounds, key bumped before rounds 2..10 (canonical Random123 schedule).
    [[nodiscard]] RETICOLO_HD static U32x4 bijection(U32x4 ctr, U32x2 key) {
        ctr = single_round(ctr, key);
        for (int r = 1; r < 10; ++r) {
            key[0] += kWeyl0;
            key[1] += kWeyl1;
            ctr = single_round(ctr, key);
        }
        return ctr;
    }
};

// 2^-53 — a 53-bit integer scaled by this lands exactly in [0, 1) (no rounding).
inline constexpr double k_u53_scale = 1.0 / 9007199254740992.0;

// Two uniform doubles in [0, 1) from (seed, traj, index). Bit-identical across
// any device (integer ops + an exact 53-bit→double scaling).
RETICOLO_HD inline void philox_uniform2(
    std::uint64_t seed, std::uint64_t traj, std::uint64_t index, double& u0, double& u1) {
    Philox4x32::U32x2 const key{static_cast<std::uint32_t>(seed),
                                static_cast<std::uint32_t>(seed >> 32U)};
    Philox4x32::U32x4 const ctr{static_cast<std::uint32_t>(traj),
                                static_cast<std::uint32_t>(traj >> 32U),
                                static_cast<std::uint32_t>(index),
                                static_cast<std::uint32_t>(index >> 32U)};
    Philox4x32::U32x4 const o = Philox4x32::bijection(ctr, key);
    std::uint64_t const b0    = (static_cast<std::uint64_t>(o[1]) << 32U) | o[0];
    std::uint64_t const b1    = (static_cast<std::uint64_t>(o[3]) << 32U) | o[2];
    u0                        = static_cast<double>(b0 >> 11U) * k_u53_scale;
    u1                        = static_cast<double>(b1 >> 11U) * k_u53_scale;
}

// Two standard normals via Box-Muller. ~1 ULP across devices (transcendental).
RETICOLO_HD inline void philox_normal2(
    std::uint64_t seed, std::uint64_t traj, std::uint64_t index, double& n0, double& n1) {
    double u0 = 0.0;
    double u1 = 0.0;
    philox_uniform2(seed, traj, index, u0, u1);
    constexpr double k_two_pi = 2.0 * std::numbers::pi;
    double const r            = std::sqrt(-2.0 * std::log(u0 > 1.0e-300 ? u0 : 1.0e-300));
    double const theta        = k_two_pi * u1;
    n0                        = r * std::cos(theta);
    n1                        = r * std::sin(theta);
}

}  // namespace reticolo
