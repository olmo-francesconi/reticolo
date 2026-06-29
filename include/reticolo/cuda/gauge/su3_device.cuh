#pragma once

#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/core/hd.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>

#include <cmath>

// Device SU(3) matrix algebra — register-local 3×3 complex ops on the 18-real
// link layout (slot 2·(3i+j) = Re U_{ij}, +1 = Im U_{ij}; the same convention as
// math::su3). These are a DEVICE-ONLY re-implementation of the per-link math
// (NOT the CPU Sleef-batched slab loops), RETICOLO_HD so they can also run on
// the host for the validation probe. The per-link formulas are copied verbatim
// from math::su3 (mul/mul_adj/adj_mul/traceless_antiherm/exp_su3) so the device
// path is bit-faithful to the CPU scalar reference; the generic gauge kernels
// (gauge_sun.cuh) drive them through the GD = SU3Device traits interface. The
// matrix exponential is Morningstar-Peardon Cayley-Hamilton, exactly as on host.

namespace reticolo::cuda {

struct SU3Device {
    using scalar_t               = double;
    static constexpr int nc      = 18;  // reals per link (9 complex entries)
    static constexpr int n_gen   = 8;   // algebra generators N²-1
    static constexpr int n_color = 3;

    RETICOLO_HD static constexpr int idx_re(int i, int j) { return 2 * ((3 * i) + j); }
    RETICOLO_HD static constexpr int idx_im(int i, int j) { return (2 * ((3 * i) + j)) + 1; }

    // SoA gather/scatter: component k of link (mu, x) at field[(mu*nc + k)*ns + x].
    RETICOLO_HD static void load(double const* field, int mu, long x, long ns, double* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            m[k] = field[base + (static_cast<long>(k) * ns)];
        }
    }
    RETICOLO_HD static void store(double* field, int mu, long x, long ns, double const* m) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = m[k];
        }
    }
    RETICOLO_HD static void
    store_scaled(double* field, int mu, long x, long ns, double const* m, double scale) {
        long const base = ((static_cast<long>(mu) * nc) * ns) + x;
        for (int k = 0; k < nc; ++k) {
            field[base + (static_cast<long>(k) * ns)] = scale * m[k];
        }
    }

    // out = a · b
    RETICOLO_HD static void mul(double* out, double const* a, double const* b) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double cr = 0.0;
                double ci = 0.0;
                for (int k = 0; k < 3; ++k) {
                    double const ar = a[idx_re(i, k)];
                    double const ai = a[idx_im(i, k)];
                    double const br = b[idx_re(k, j)];
                    double const bi = b[idx_im(k, j)];
                    cr += (ar * br) - (ai * bi);
                    ci += (ar * bi) + (ai * br);
                }
                out[idx_re(i, j)] = cr;
                out[idx_im(i, j)] = ci;
            }
        }
    }
    // out = a · b†   (out_{ij} = Σ_k a_{ik} conj(b_{jk}))
    RETICOLO_HD static void mul_adj(double* out, double const* a, double const* b) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double cr = 0.0;
                double ci = 0.0;
                for (int k = 0; k < 3; ++k) {
                    double const ar = a[idx_re(i, k)];
                    double const ai = a[idx_im(i, k)];
                    double const br = b[idx_re(j, k)];
                    double const bi = b[idx_im(j, k)];
                    cr += (ar * br) + (ai * bi);
                    ci += (ai * br) - (ar * bi);
                }
                out[idx_re(i, j)] = cr;
                out[idx_im(i, j)] = ci;
            }
        }
    }
    // out = a† · b   (out_{ij} = Σ_k conj(a_{ki}) b_{kj})
    RETICOLO_HD static void adj_mul(double* out, double const* a, double const* b) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double cr = 0.0;
                double ci = 0.0;
                for (int k = 0; k < 3; ++k) {
                    double const ar = a[idx_re(k, i)];
                    double const ai = a[idx_im(k, i)];
                    double const br = b[idx_re(k, j)];
                    double const bi = b[idx_im(k, j)];
                    cr += (ar * br) + (ai * bi);
                    ci += (ar * bi) - (ai * br);
                }
                out[idx_re(i, j)] = cr;
                out[idx_im(i, j)] = ci;
            }
        }
    }

    // TA(M) = (M − M†)/2 − (1/3)·Tr((M−M†)/2)·I, into the 18-real layout.
    RETICOLO_HD static void traceless_antiherm(double* out, double const* in) {
        double const im00     = in[idx_im(0, 0)];
        double const im11     = in[idx_im(1, 1)];
        double const im22     = in[idx_im(2, 2)];
        double const t_over_3 = (im00 + im11 + im22) / 3.0;
        out[idx_re(0, 0)]     = 0.0;
        out[idx_im(0, 0)]     = im00 - t_over_3;
        out[idx_re(1, 1)]     = 0.0;
        out[idx_im(1, 1)]     = im11 - t_over_3;
        out[idx_re(2, 2)]     = 0.0;
        out[idx_im(2, 2)]     = im22 - t_over_3;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double const r_ij = in[idx_re(i, j)];
                double const i_ij = in[idx_im(i, j)];
                double const r_ji = in[idx_re(j, i)];
                double const i_ji = in[idx_im(j, i)];
                double const re   = 0.5 * (r_ij - r_ji);
                double const im   = 0.5 * (i_ij + i_ji);
                out[idx_re(i, j)] = re;
                out[idx_im(i, j)] = im;
                out[idx_re(j, i)] = -re;
                out[idx_im(j, i)] = im;
            }
        }
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
    // Cayley-Hamilton, copied verbatim from math::su3::exp_su3 (Q = -i·dt·P).
    RETICOLO_HD static void expi(double dt, double const* p, double* v) {
        double q[18];
        for (int k = 0; k < 9; ++k) {
            q[2 * k]       = dt * p[(2 * k) + 1];
            q[(2 * k) + 1] = -dt * p[2 * k];
        }

        double c1 = 0.0;
        for (int k = 0; k < 18; ++k) {
            c1 += q[k] * q[k];
        }
        c1 *= 0.5;

        constexpr double k_small_c1 = 1.0e-8;
        if (c1 < k_small_c1) {
            double q2[18];
            mul(q2, q, q);
            double q3[18];
            mul(q3, q2, q);
            for (int k = 0; k < 18; ++k) {
                v[k] = -0.5 * q2[k];
            }
            v[idx_re(0, 0)] += 1.0;
            v[idx_re(1, 1)] += 1.0;
            v[idx_re(2, 2)] += 1.0;
            for (int k = 0; k < 9; ++k) {
                v[2 * k] += -q[(2 * k) + 1];
                v[(2 * k) + 1] += q[2 * k];
            }
            constexpr double k_inv6 = 1.0 / 6.0;
            for (int k = 0; k < 9; ++k) {
                v[2 * k] += k_inv6 * q3[(2 * k) + 1];
                v[(2 * k) + 1] -= k_inv6 * q3[2 * k];
            }
            return;
        }

        double q2[18];
        mul(q2, q, q);
        double const tr_q3 =
            (q2[idx_re(0, 0)] * q[idx_re(0, 0)] - q2[idx_im(0, 0)] * q[idx_im(0, 0)]) +
            (q2[idx_re(0, 1)] * q[idx_re(1, 0)] - q2[idx_im(0, 1)] * q[idx_im(1, 0)]) +
            (q2[idx_re(0, 2)] * q[idx_re(2, 0)] - q2[idx_im(0, 2)] * q[idx_im(2, 0)]) +
            (q2[idx_re(1, 0)] * q[idx_re(0, 1)] - q2[idx_im(1, 0)] * q[idx_im(0, 1)]) +
            (q2[idx_re(1, 1)] * q[idx_re(1, 1)] - q2[idx_im(1, 1)] * q[idx_im(1, 1)]) +
            (q2[idx_re(1, 2)] * q[idx_re(2, 1)] - q2[idx_im(1, 2)] * q[idx_im(2, 1)]) +
            (q2[idx_re(2, 0)] * q[idx_re(0, 2)] - q2[idx_im(2, 0)] * q[idx_im(0, 2)]) +
            (q2[idx_re(2, 1)] * q[idx_re(1, 2)] - q2[idx_im(2, 1)] * q[idx_im(1, 2)]) +
            (q2[idx_re(2, 2)] * q[idx_re(2, 2)] - q2[idx_im(2, 2)] * q[idx_im(2, 2)]);
        double const c0 = tr_q3 / 3.0;

        double const c1_over_3 = c1 / 3.0;
        double const c0_max    = 2.0 * c1_over_3 * std::sqrt(c1_over_3);
        double ratio           = 0.0;
        if (c0_max > 0.0) {
            ratio = c0 / c0_max;
            ratio = (ratio < -1.0) ? -1.0 : ((ratio > 1.0) ? 1.0 : ratio);
        }
        double const theta = std::acos(ratio);

        double const sqrt_c1_3 = std::sqrt(c1_over_3);
        double const u         = sqrt_c1_3 * std::cos(theta / 3.0);
        double const w         = std::sqrt(c1) * std::sin(theta / 3.0);

        double const cw  = std::cos(w);
        double const cu  = std::cos(u);
        double const su_ = std::sin(u);
        double const c2u = std::cos(2.0 * u);
        double const s2u = std::sin(2.0 * u);
        double const xi  = (std::fabs(w) > 1.0e-10) ? (std::sin(w) / w) : 1.0;

        double const u2  = u * u;
        double const w2  = w * w;
        double const den = (9.0 * u2) - w2;

        double const t0a_re = (u2 - w2) * c2u;
        double const t0a_im = (u2 - w2) * s2u;
        double const t0b_re = (cu * (8.0 * u2 * cw)) + (su_ * (2.0 * u * (3.0 * u2 + w2) * xi));
        double const t0b_im = -(su_ * (8.0 * u2 * cw)) + (cu * (2.0 * u * (3.0 * u2 + w2) * xi));
        double const h0_re  = t0a_re + t0b_re;
        double const h0_im  = t0a_im + t0b_im;

        double const t1a_re    = 2.0 * u * c2u;
        double const t1a_im    = 2.0 * u * s2u;
        double const inner1_re = 2.0 * u * cw;
        double const inner1_im = -((3.0 * u2) - w2) * xi;
        double const t1b_re    = (cu * inner1_re) + (su_ * inner1_im);
        double const t1b_im    = (cu * inner1_im) - (su_ * inner1_re);
        double const h1_re     = t1a_re - t1b_re;
        double const h1_im     = t1a_im - t1b_im;

        double const inner2_re = cw;
        double const inner2_im = 3.0 * u * xi;
        double const t2b_re    = (cu * inner2_re) + (su_ * inner2_im);
        double const t2b_im    = (cu * inner2_im) - (su_ * inner2_re);
        double const h2_re     = c2u - t2b_re;
        double const h2_im     = s2u - t2b_im;

        double const inv_den = 1.0 / den;
        double const f0_re   = h0_re * inv_den;
        double const f0_im   = h0_im * inv_den;
        double const f1_re   = h1_re * inv_den;
        double const f1_im   = h1_im * inv_den;
        double const f2_re   = h2_re * inv_den;
        double const f2_im   = h2_im * inv_den;

        for (int k = 0; k < 9; ++k) {
            int const kr       = 2 * k;
            int const ki       = kr + 1;
            double const q_re  = q[kr];
            double const q_im  = q[ki];
            double const q2_re = q2[kr];
            double const q2_im = q2[ki];
            v[kr] = ((f1_re * q_re) - (f1_im * q_im)) + ((f2_re * q2_re) - (f2_im * q2_im));
            v[ki] = ((f1_re * q_im) + (f1_im * q_re)) + ((f2_re * q2_im) + (f2_im * q2_re));
        }
        v[idx_re(0, 0)] += f0_re;
        v[idx_im(0, 0)] += f0_im;
        v[idx_re(1, 1)] += f0_re;
        v[idx_im(1, 1)] += f0_im;
        v[idx_re(2, 2)] += f0_re;
        v[idx_im(2, 2)] += f0_im;
    }

    // Scatter the n_gen Gell-Mann coords h=(h1..h8) into the anti-hermitian
    // 18-real layout (matches math::su3::sample_algebra_slab).
    RETICOLO_HD static void pack_algebra(double const* h, double* p) {
        constexpr double k_inv_sqrt3 = 0.57735026918962576451;
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
struct group_device<gauge_group::SU3> {
    using type = SU3Device;
};

}  // namespace reticolo::cuda
