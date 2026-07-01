#pragma once

#include <reticolo/action/detail/gauge/gauge_group/su2.hpp>
#include <reticolo/core/hd.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>

#include <cmath>

// Device SU(2) matrix algebra — register-local 2×2 complex ops on the 8-real
// link layout (k=0..7 = 00r,00i,01r,01i,10r,10i,11r,11i; the same convention as
// math::su2). These are a DEVICE-ONLY re-implementation of the per-link math
// (NOT the CPU Sleef-batched slab loops), RETICOLO_HD so they can also run on
// the host for the validation probe. The generic gauge kernels (gauge_sun.cuh)
// drive them through the GD = SU2Device traits interface.

namespace reticolo::cuda {

struct SU2Device {
    using scalar_t               = double;
    static constexpr int nc      = 8;  // reals per link
    static constexpr int n_gen   = 3;  // algebra generators N²-1
    static constexpr int n_color = 2;

    // SoA gather/scatter: component k of link (mu, x) at field[(mu*nc + k)*ns + x].
    RETICOLO_HD static void load(double const* __restrict__ field, int mu, long x, long ns, double* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            m[k] = field[base + (static_cast<long>(k) * ns)];
        }
    }
    RETICOLO_HD static void store(double* __restrict__ field, int mu, long x, long ns, double const* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = m[k];
        }
    }
    RETICOLO_HD static void
    store_scaled(double* __restrict__ field, int mu, long x, long ns, double const* m, double scale) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = scale * m[k];
        }
    }

    // out = a · b
    RETICOLO_HD static void mul(double* o, double const* a, double const* b) {
        cset(o, 0, a[0], a[1], b[0], b[1], a[2], a[3], b[4], b[5]);
        cset(o, 1, a[0], a[1], b[2], b[3], a[2], a[3], b[6], b[7]);
        cset(o, 2, a[4], a[5], b[0], b[1], a[6], a[7], b[4], b[5]);
        cset(o, 3, a[4], a[5], b[2], b[3], a[6], a[7], b[6], b[7]);
    }
    // out = a · b†   (out_{ij} = Σ_k a_{ik} conj(b_{jk}))
    RETICOLO_HD static void mul_adj(double* o, double const* a, double const* b) {
        cset_badj(o, 0, a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]);
        cset_badj(o, 1, a[0], a[1], b[4], b[5], a[2], a[3], b[6], b[7]);
        cset_badj(o, 2, a[4], a[5], b[0], b[1], a[6], a[7], b[2], b[3]);
        cset_badj(o, 3, a[4], a[5], b[4], b[5], a[6], a[7], b[6], b[7]);
    }
    // out = a† · b   (out_{ij} = Σ_k conj(a_{ki}) b_{kj})
    RETICOLO_HD static void adj_mul(double* o, double const* a, double const* b) {
        cset_aadj(o, 0, a[0], a[1], b[0], b[1], a[4], a[5], b[4], b[5]);
        cset_aadj(o, 1, a[0], a[1], b[2], b[3], a[4], a[5], b[6], b[7]);
        cset_aadj(o, 2, a[2], a[3], b[0], b[1], a[6], a[7], b[4], b[5]);
        cset_aadj(o, 3, a[2], a[3], b[2], b[3], a[6], a[7], b[6], b[7]);
    }

    // TA(M) = (M − M†)/2 − ½Tr(...)·I, into the anti-hermitian 8-real layout.
    RETICOLO_HD static void traceless_antiherm(double* out, double const* in) {
        double const diag_im = 0.5 * (in[1] - in[7]);
        out[0]               = 0.0;
        out[1]               = diag_im;
        out[2]               = 0.5 * (in[2] - in[4]);
        out[3]               = 0.5 * (in[3] + in[5]);
        out[4]               = -out[2];
        out[5]               = out[3];
        out[6]               = 0.0;
        out[7]               = -diag_im;
    }

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
    RETICOLO_HD static void expi(double dt, double const* p, double* v) {
        double const h3   = p[1];
        double const h2   = p[2];
        double const h1   = p[3];
        double const h    = std::sqrt((h1 * h1) + (h2 * h2) + (h3 * h3));
        double const beta = dt * h;
        double const c    = std::cos(beta);
        double const gamma =
            (h > 1.0e-12) ? (std::sin(beta) / h) : (dt - ((dt * beta * beta) / 6.0));
        v[0] = c;
        v[1] = gamma * h3;
        v[2] = gamma * h2;
        v[3] = gamma * h1;
        v[4] = -gamma * h2;
        v[5] = gamma * h1;
        v[6] = c;
        v[7] = -gamma * h3;
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

private:
    // o_{entry} = (ar1+i ai1)(br1+i bi1) + (ar2+i ai2)(br2+i bi2)
    RETICOLO_HD static void cset(double* o,
                                 int e,
                                 double ar1,
                                 double ai1,
                                 double br1,
                                 double bi1,
                                 double ar2,
                                 double ai2,
                                 double br2,
                                 double bi2) {
        o[2 * e]       = (ar1 * br1) - (ai1 * bi1) + (ar2 * br2) - (ai2 * bi2);
        o[(2 * e) + 1] = (ar1 * bi1) + (ai1 * br1) + (ar2 * bi2) + (ai2 * br2);
    }
    // with conj(b) on each term
    RETICOLO_HD static void cset_badj(double* o,
                                      int e,
                                      double ar1,
                                      double ai1,
                                      double br1,
                                      double bi1,
                                      double ar2,
                                      double ai2,
                                      double br2,
                                      double bi2) {
        o[2 * e]       = (ar1 * br1) + (ai1 * bi1) + (ar2 * br2) + (ai2 * bi2);
        o[(2 * e) + 1] = (ai1 * br1) - (ar1 * bi1) + (ai2 * br2) - (ar2 * bi2);
    }
    // with conj(a) on each term
    RETICOLO_HD static void cset_aadj(double* o,
                                      int e,
                                      double ar1,
                                      double ai1,
                                      double br1,
                                      double bi1,
                                      double ar2,
                                      double ai2,
                                      double br2,
                                      double bi2) {
        o[2 * e]       = (ar1 * br1) + (ai1 * bi1) + (ar2 * br2) + (ai2 * bi2);
        o[(2 * e) + 1] = (ar1 * bi1) - (ai1 * br1) + (ar2 * bi2) - (ai2 * br2);
    }
};

template <>
struct group_device<gauge_group::SU2> {
    using type = SU2Device;
};

}  // namespace reticolo::cuda
