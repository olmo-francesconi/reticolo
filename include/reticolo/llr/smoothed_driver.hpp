#pragma once

#include <reticolo/io/writer.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/log.hpp>
#include <reticolo/llr/update_a.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <vector>

namespace reticolo::llr::smoothed {

// Smoothed LLR driver: vanilla per-replica NR warm-up, then an RM phase
// in which a local-polynomial fit â(E) across replicas is shrunk into
// each replica's iterate at every step.
//
//   a_rm[n]   = a[n] + ⟨dE⟩[n] / (δ² (s+1))               (per-replica RM)
//   â         = local_poly_fit({(E_m, a_rm[m])}, K, deg) (serial)
//   λ_s       = λ₀ / (s+1)^α                              (shrinkage weight)
//   a[n] ← (1 − λ_s) · a_rm[n] + λ_s · â[n]
//
// With α > 1 the cumulative shrinkage perturbation is summable, so the
// fixed point is unchanged from vanilla LLR (Kushner-Yin perturbed-SA
// theorem). The smoother only accelerates the transient and cannot
// introduce asymptotic bias even if the smoothness assumption fails.
//
// Output schema mirrors `llr::run`'s /replica_NNN/a, /dE plus four extra
// per-replica series: /a_pre_shrink (a_rm before the convex combination),
// /a_hat (the smoothed prediction), /da_rm (|a_rm − a_old|) and /da_sm
// (|λ·(â − a_rm)|, the shrinkage step magnitude). At convergence the
// three a-series must agree to the noise floor.

namespace impl {

// In-place Gauss elimination with partial pivoting. `mat` is `n × n`
// row-major; `vec` is `n`. Returns false if the matrix is singular to
// within `k_tol`; otherwise `vec` holds the solution on return.
inline bool gauss_solve(std::vector<double>& mat, std::vector<double>& vec, int n) {
    constexpr double k_tol = 1e-12;
    auto const ni          = static_cast<std::size_t>(n);
    auto idx               = [ni](int i, int j) {
        return (static_cast<std::size_t>(i) * ni) + static_cast<std::size_t>(j);
    };
    for (int k = 0; k < n; ++k) {
        int pivot          = k;
        double pivot_value = std::abs(mat[idx(k, k)]);
        for (int i = k + 1; i < n; ++i) {
            double const v = std::abs(mat[idx(i, k)]);
            if (v > pivot_value) {
                pivot_value = v;
                pivot       = i;
            }
        }
        if (pivot_value < k_tol) {
            return false;
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(mat[idx(k, j)], mat[idx(pivot, j)]);
            }
            std::swap(vec[static_cast<std::size_t>(k)], vec[static_cast<std::size_t>(pivot)]);
        }
        double const inv_pk = 1.0 / mat[idx(k, k)];
        for (int i = k + 1; i < n; ++i) {
            double const factor = mat[idx(i, k)] * inv_pk;
            for (int j = k; j < n; ++j) {
                mat[idx(i, j)] -= factor * mat[idx(k, j)];
            }
            vec[static_cast<std::size_t>(i)] -= factor * vec[static_cast<std::size_t>(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = vec[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= mat[idx(i, j)] * vec[static_cast<std::size_t>(j)];
        }
        vec[static_cast<std::size_t>(i)] = sum / mat[idx(i, i)];
    }
    return true;
}

// Local-polynomial fit of degree `deg` through the (2K+1) replicas
// centred on index `n` (clamped at the array ends), evaluated at E = e[n].
// Centering at e[n] means the smoothed value is just the constant term.
// If fewer than `deg + 1` points are available the smoother falls back
// to the raw value `a[n]`.
// NOLINTNEXTLINE(readability-identifier-naming, bugprone-easily-swappable-parameters)
[[nodiscard]] inline double local_poly_fit_at(std::vector<double> const& e,
                                              std::vector<double> const& a,
                                              std::size_t n,
                                              int K,  // NOLINT
                                              int deg) {
    int const total = static_cast<int>(e.size());
    int const ni    = static_cast<int>(n);
    int const lo    = std::max(0, ni - K);
    int const hi    = std::min(total - 1, ni + K);
    int const npts  = (hi - lo) + 1;
    int const dim   = deg + 1;
    if (npts < dim) {
        return a[n];
    }

    double const e0 = e[n];
    std::vector<double> moms(static_cast<std::size_t>((2 * deg) + 1), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
    for (int m = lo; m <= hi; ++m) {
        double const x = e[static_cast<std::size_t>(m)] - e0;
        double const y = a[static_cast<std::size_t>(m)];
        double xp      = 1.0;
        for (int p = 0; p <= 2 * deg; ++p) {
            moms[static_cast<std::size_t>(p)] += xp;
            if (p <= deg) {
                rhs[static_cast<std::size_t>(p)] += xp * y;
            }
            xp *= x;
        }
    }
    auto const dim_z = static_cast<std::size_t>(dim);
    std::vector<double> mat(dim_z * dim_z, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            mat[(static_cast<std::size_t>(i) * dim_z) + static_cast<std::size_t>(j)] =
                moms[static_cast<std::size_t>(i) + static_cast<std::size_t>(j)];
        }
    }
    if (!gauss_solve(mat, rhs, dim)) {
        return a[n];
    }
    return rhs[0];
}

inline void local_poly_fit(std::vector<double> const& e,
                           std::vector<double> const& a,
                           int K,  // NOLINT(readability-identifier-naming)
                           int deg,
                           std::vector<double>& out) {
    std::size_t const total = e.size();
    out.assign(total, 0.0);
    for (std::size_t n = 0; n < total; ++n) {
        out[n] = local_poly_fit_at(e, a, n, K, deg);
    }
}

}  // namespace impl

struct DriverSpec {
    int n_nr{};
    int n_therm_nr{};
    int n_meas_nr{};
    int n_rm{};
    int n_therm_rm{};
    int n_meas_rm{};
    double delta{};
    double e_min{};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    double E_max{};
    double d_e{};
    bool exchange = true;
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    int smooth_K          = 4;
    int smooth_degree     = 2;
    double smooth_lambda0 = 1.0;
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    double smooth_lambda_exp = 2.0;
};

template <class Replica, class ExchRng>
void run(std::vector<std::unique_ptr<Replica>>& reps,
         ExchRng& exch_rng,
         DriverSpec const& spec,
         io::Writer& out) {
    int const n_rep           = static_cast<int>(reps.size());
    std::size_t const n_rep_u = reps.size();

    out.attr<int>("/cfg@n_rep", n_rep);
    out.attr<int>("/cfg@n_nr", spec.n_nr);
    out.attr<int>("/cfg@n_rm", spec.n_rm);
    out.attr<double>("/cfg@delta", spec.delta);
    out.attr<double>("/cfg@E_min", spec.e_min);
    out.attr<double>("/cfg@E_max", spec.E_max);
    out.attr<double>("/cfg@dE", spec.d_e);
    out.attr<int>("/cfg@smooth_K", spec.smooth_K);
    out.attr<int>("/cfg@smooth_degree", spec.smooth_degree);
    out.attr<double>("/cfg@smooth_lambda0", spec.smooth_lambda0);
    out.attr<double>("/cfg@smooth_lambda_exp", spec.smooth_lambda_exp);

    auto e_n_series = out.series<double>("/cfg/E_n");
    std::vector<double> e_n_vec;
    e_n_vec.reserve(n_rep_u);
    for (auto const& r : reps) {
        auto const en = r->E_n();
        e_n_series.append(en);
        e_n_vec.push_back(static_cast<double>(en));
    }

    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    std::vector<io::Series<double>> a_pre_series;
    std::vector<io::Series<double>> a_hat_series;
    std::vector<io::Series<double>> drm_series;
    std::vector<io::Series<double>> dsm_series;
    a_series.reserve(n_rep_u);
    de_series.reserve(n_rep_u);
    a_pre_series.reserve(n_rep_u);
    a_hat_series.reserve(n_rep_u);
    drm_series.reserve(n_rep_u);
    dsm_series.reserve(n_rep_u);
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a", n)));
        de_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/dE", n)));
        a_pre_series.emplace_back(
            out.series<double>(std::format("/replica_{:03d}/a_pre_shrink", n)));
        a_hat_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a_hat", n)));
        drm_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/da_rm", n)));
        dsm_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/da_sm", n)));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    std::vector<double> de_buf(n_rep_u);
    std::vector<double> a_rm(n_rep_u);
    std::vector<double> a_hat(n_rep_u);
    std::vector<double> a_buf(n_rep_u);
    std::vector<double> drm_buf(n_rep_u);
    std::vector<double> dsm_buf(n_rep_u);

    log::info("llr", "smoothed NR phase  {} iters × {} replicas", spec.n_nr, n_rep_u);
    for (int k = 0; k < spec.n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto& r = *reps[n];
            r.thermalize(spec.n_therm_nr);
            de_buf[n] = r.sample(spec.n_meas_nr);
            a_buf[n]  = nr_update(r.a(), de_buf[n], spec.delta);
            r.set_a(a_buf[n]);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
            a_pre_series[n].append(a_buf[n]);
            a_hat_series[n].append(a_buf[n]);
            drm_series[n].append(0.0);
            dsm_series[n].append(0.0);
        }
        log::info("llr", "NR iter  {:>3}/{}  done", k + 1, spec.n_nr);
    }

    log::info("llr",
              "smoothed RM phase  {} iters × {} replicas  (K={}, deg={}, λ0={:.2g}, exp={:.2g})",
              spec.n_rm,
              n_rep_u,
              spec.smooth_K,
              spec.smooth_degree,
              spec.smooth_lambda0,
              spec.smooth_lambda_exp);
    for (int s = 0; s < spec.n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto& r = *reps[n];
            r.thermalize(spec.n_therm_rm, log::Mode::silent);
            de_buf[n] = r.sample(spec.n_meas_rm, log::Mode::silent);
            a_rm[n]   = rm_update(r.a(), de_buf[n], spec.delta, s);
        }

        impl::local_poly_fit(e_n_vec, a_rm, spec.smooth_K, spec.smooth_degree, a_hat);
        double const lam =
            spec.smooth_lambda0 / std::pow(static_cast<double>(s + 1), spec.smooth_lambda_exp);
        double rm_step_sum = 0.0;
        double rm_step_max = 0.0;
        double sm_step_sum = 0.0;
        double sm_step_max = 0.0;
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto const a_old = static_cast<double>(reps[n]->a());
            drm_buf[n]       = std::abs(a_rm[n] - a_old);
            dsm_buf[n]       = std::abs(lam * (a_hat[n] - a_rm[n]));
            rm_step_sum += drm_buf[n];
            sm_step_sum += dsm_buf[n];
            if (drm_buf[n] > rm_step_max) {
                rm_step_max = drm_buf[n];
            }
            if (dsm_buf[n] > sm_step_max) {
                sm_step_max = dsm_buf[n];
            }
            a_buf[n] = ((1.0 - lam) * a_rm[n]) + (lam * a_hat[n]);
            reps[n]->set_a(a_buf[n]);
            auto _ = log::scope(reps[n]->id());
            iter("sRM",
                 static_cast<std::size_t>(s) + 1,
                 static_cast<std::size_t>(spec.n_rm),
                 a_buf[n],
                 de_buf[n],
                 spec.delta);
        }
        double const inv_n      = 1.0 / static_cast<double>(n_rep_u);
        double const rm_mean    = rm_step_sum * inv_n;
        double const sm_mean    = sm_step_sum * inv_n;
        double const ratio_mean = (rm_mean > 0.0) ? (sm_mean / rm_mean) : 0.0;
        double const ratio_max  = (rm_step_max > 0.0) ? (sm_step_max / rm_step_max) : 0.0;
        log::info("llr",
                  "sRM step  {:>3}/{}  λ={:.3e}  |Δa_rm| mean={:.2e} max={:.2e}  "
                  "|Δa_sm| mean={:.2e} max={:.2e}  sm/rm mean={:.2f} max={:.2f}",
                  s + 1,
                  spec.n_rm,
                  lam,
                  rm_mean,
                  rm_step_max,
                  sm_mean,
                  sm_step_max,
                  ratio_mean,
                  ratio_max);

        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
            a_pre_series[n].append(a_rm[n]);
            a_hat_series[n].append(a_hat[n]);
            drm_series[n].append(drm_buf[n]);
            dsm_series[n].append(dsm_buf[n]);
        }

        int accepted = 0;
        int attempts = 0;
        if (spec.exchange) {
            auto const off = static_cast<std::size_t>(s & 1);
            for (std::size_t i = off; i + 1 < reps.size(); i += 2) {
                ++attempts;
                if (try_exchange(*reps[i], *reps[i + 1], exch_rng)) {
                    ++accepted;
                }
            }
        }
        exch_series.append(accepted);
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
    }
}

}  // namespace reticolo::llr::smoothed
