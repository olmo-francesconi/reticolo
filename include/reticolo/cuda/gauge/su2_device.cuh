#pragma once

#include <reticolo/core/sys/hd.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/math/group/formula/su2_formula.hpp>
#include <reticolo/math/group/su2.hpp>

// Device SU(2) matrix algebra — register-local 2×2 complex ops on the 8-real
// link layout (k=0..7 = 00r,00i,01r,01i,10r,10i,11r,11i; the same convention as
// math::su2). RETICOLO_HD so they also run on the host for the validation
// probe. The per-link formulas (mul/mul_adj/adj_mul/traceless_antiherm/exp_su2)
// are single-sourced in math/group/formula/su2_formula.hpp and forwarded here —
// no re-implementation, so the device path cannot drift from the CPU scalar
// reference; the generic gauge kernels (gauge_sun.cuh) drive these forwards
// through the GD = SU2Device traits interface.

namespace reticolo::cuda {

struct SU2Device {
    using scalar_t               = double;
    static constexpr int nc      = 8;  // reals per link
    static constexpr int n_gen   = 3;  // algebra generators N²-1
    static constexpr int n_color = 2;

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

    // The 2×2 identity in the 8-real (row-major, re/im-interleaved) layout:
    // Re U_00 = comp 0, Re U_11 = comp 6.
    RETICOLO_HD static void identity(double* m) {
        for (int k = 0; k < nc; ++k) {
            m[k] = 0.0;
        }
        m[0] = 1.0;
        m[6] = 1.0;
    }

    // out = a · b
    RETICOLO_HD static void mul(double* o, double const* a, double const* b) {
        math::su2::mul_2x2(o, a, b);
    }
    // out = a · b†   (out_{ij} = Σ_k a_{ik} conj(b_{jk}))
    RETICOLO_HD static void mul_adj(double* o, double const* a, double const* b) {
        math::su2::mul_adj_2x2(o, a, b);
    }
    // out = a† · b   (out_{ij} = Σ_k conj(a_{ki}) b_{kj})
    RETICOLO_HD static void adj_mul(double* o, double const* a, double const* b) {
        math::su2::adj_mul_2x2(o, a, b);
    }

    // TA(M) = (M − M†)/2 − ½Tr(...)·I, into the anti-hermitian 8-real layout.
    RETICOLO_HD static void traceless_antiherm(double* out, double const* in) {
        math::su2::traceless_antiherm_2x2(out, in);
    }

    // Re Tr(M) = Σ_i Re M_ii — the real diagonal entries (comps 0, 6 for 2×2).
    RETICOLO_HD static double retr(double const* m) { return m[0] + m[6]; }

    // Re Tr(X · Y†) = Σ_k X[k]·Y[k] over the 2N² reals (generic identity).
    RETICOLO_HD static double retr_mul_adj(double const* x, double const* y) {
        double acc = 0.0;
        for (int k = 0; k < nc; ++k) {
            acc += x[k] * y[k];
        }
        return acc;
    }

    // Group exp: V = exp(dt·P), P the anti-hermitian algebra element. Closed form
    // V = cos(dt‖h‖)·I + i·sin(dt‖h‖)/‖h‖·(h·σ); h = (Im P01, Re P01, Im P00).
    // Forwarded to math::su2::exp_su2 — NOTE the argument-order swap (exp_su2
    // takes (v, p, dt), this trait takes (dt, p, v)).
    RETICOLO_HD static void expi(double dt, double const* p, double* v) {
        math::su2::exp_su2(v, p, dt);
    }

    // Scatter the n_gen algebra coords h=(h1,h2,h3) into the anti-hermitian
    // 8-real layout (matches math::su2::sample_algebra_slab).
    RETICOLO_HD static void pack_algebra(double const* h, double* p) {
        p[0] = 0.0;
        p[1] = h[2];
        p[2] = h[1];
        p[3] = h[0];
        p[4] = -h[1];
        p[5] = h[0];
        p[6] = 0.0;
        p[7] = -h[2];
    }
};

template <>
struct group_device<math::group::SU2> {
    using type = SU2Device;
};

}  // namespace reticolo::cuda
