#pragma once

#include <reticolo/core/sys/hd.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/math/group/formula/su3_formula.hpp>
#include <reticolo/math/group/su3.hpp>

#include <numbers>

// Device SU(3) matrix algebra — register-local 3×3 complex ops on the 18-real
// link layout (slot 2·(3i+j) = Re U_{ij}, +1 = Im U_{ij}; the same convention as
// math::su3). RETICOLO_HD so they also run on the host for the validation
// probe. The per-link formulas (mul/mul_adj/adj_mul/traceless_antiherm/exp_su3)
// are single-sourced in math/group/formula/su3_formula.hpp and forwarded here —
// no re-implementation, so the device path cannot drift from the CPU scalar
// reference; the generic gauge kernels (gauge_sun.cuh) drive these forwards
// through the GD = SU3Device traits interface. The matrix exponential is
// Morningstar-Peardon Cayley-Hamilton, exactly as on host.

namespace reticolo::cuda {

struct SU3Device {
    using scalar_t               = double;
    static constexpr int nc      = 18;  // reals per link (9 complex entries)
    static constexpr int n_gen   = 8;   // algebra generators N²-1
    static constexpr int n_color = 3;

    RETICOLO_HD static constexpr int idx_re(int i, int j) { return 2 * ((3 * i) + j); }
    RETICOLO_HD static constexpr int idx_im(int i, int j) { return (2 * ((3 * i) + j)) + 1; }

    // SoA gather/scatter: component k of link (mu, x) at field[(mu*nc + k)*ns + x].
    RETICOLO_HD static void
    load(double const* __restrict__ field, int mu, long x, long ns, double* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            m[k] = field[base + (static_cast<long>(k) * ns)];
        }
    }
    RETICOLO_HD static void
    store(double* __restrict__ field, int mu, long x, long ns, double const* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = m[k];
        }
    }
    RETICOLO_HD static void store_scaled(
        double* __restrict__ field, int mu, long x, long ns, double const* m, double scale) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = scale * m[k];
        }
    }

    // The 3×3 identity in the 18-real layout: Re on the diagonal, zero elsewhere.
    RETICOLO_HD static void identity(double* m) {
        for (int k = 0; k < nc; ++k) {
            m[k] = 0.0;
        }
        for (int i = 0; i < 3; ++i) {
            m[idx_re(i, i)] = 1.0;
        }
    }

    // out = a · b
    RETICOLO_HD static void mul(double* out, double const* a, double const* b) {
        math::su3::mul_3x3(out, a, b);
    }
    // out = a · b†   (out_{ij} = Σ_k a_{ik} conj(b_{jk}))
    RETICOLO_HD static void mul_adj(double* out, double const* a, double const* b) {
        math::su3::mul_adj_3x3(out, a, b);
    }
    // out = a† · b   (out_{ij} = Σ_k conj(a_{ki}) b_{kj})
    RETICOLO_HD static void adj_mul(double* out, double const* a, double const* b) {
        math::su3::adj_mul_3x3(out, a, b);
    }

    // TA(M) = (M − M†)/2 − (1/3)·Tr((M−M†)/2)·I, into the 18-real layout.
    RETICOLO_HD static void traceless_antiherm(double* out, double const* in) {
        math::su3::traceless_antiherm_3x3(out, in);
    }

    // Re Tr(M) = Σ_i Re M_ii — the real diagonal entries.
    RETICOLO_HD static double retr(double const* m) {
        return m[idx_re(0, 0)] + m[idx_re(1, 1)] + m[idx_re(2, 2)];
    }

    // Re Tr(X · Y†) = Σ_k X[k]·Y[k] over the 2N² reals (generic identity).
    RETICOLO_HD static double retr_mul_adj(double const* x, double const* y) {
        double acc = 0.0;
        for (int k = 0; k < nc; ++k) {
            acc += x[k] * y[k];
        }
        return acc;
    }

    // Group exp V = exp(dt·P), P anti-hermitian traceless 3×3. Morningstar-Peardon
    // Cayley-Hamilton, forwarded to math::su3::exp_su3 — NOTE the argument-order
    // swap (exp_su3 takes (v, p, dt), this trait takes (dt, p, v)).
    RETICOLO_HD static void expi(double dt, double const* p, double* v) {
        math::su3::exp_su3(v, p, dt);
    }

    // Scatter the n_gen Gell-Mann coords h=(h1..h8) into the anti-hermitian
    // 18-real layout (matches math::su3::sample_algebra_slab).
    RETICOLO_HD static void pack_algebra(double const* h, double* p) {
        constexpr double k_inv_sqrt3 = std::numbers::inv_sqrt3;
        double const h8_over_sqrt3   = h[7] * k_inv_sqrt3;
        for (int k = 0; k < 18; ++k) {
            p[k] = 0.0;
        }
        p[idx_im(0, 0)] = h[2] + h8_over_sqrt3;
        p[idx_im(1, 1)] = -h[2] + h8_over_sqrt3;
        p[idx_im(2, 2)] = -2.0 * h8_over_sqrt3;
        p[idx_re(0, 1)] = h[1];
        p[idx_im(0, 1)] = h[0];
        p[idx_re(1, 0)] = -h[1];
        p[idx_im(1, 0)] = h[0];
        p[idx_re(0, 2)] = h[4];
        p[idx_im(0, 2)] = h[3];
        p[idx_re(2, 0)] = -h[4];
        p[idx_im(2, 0)] = h[3];
        p[idx_re(1, 2)] = h[6];
        p[idx_im(1, 2)] = h[5];
        p[idx_re(2, 1)] = -h[6];
        p[idx_im(2, 1)] = h[5];
    }
};

template <>
struct group_device<math::group::SU3> {
    using type = SU3Device;
};

}  // namespace reticolo::cuda
