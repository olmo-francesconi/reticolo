// Nightly GPU physics harness (app-level, NOT a ctest gate — it runs full
// simulations, which the deterministic suite must never do). It links the
// umbrella, so it runs the CPU alg::Hmc and the GPU cuda::Hmc side-by-side in
// ONE process and asserts the physics identities the per-phase roadmap tags
// [nightly]:
//
//   * <exp(-dH)> = 1  (Kennedy-Pendleton) — joint check of momentum
//     normalisation + integrator + Metropolis accept, CPU and GPU.
//   * CPU-vs-GPU observable equivalence within Monte-Carlo error — same action,
//     different RNG streams => same ensemble, not the same chain.
//   * SU(2) plaquette at beta, CPU vs GPU.
//
// Exits non-zero if any check fails. Built only with RETICOLO_ENABLE_CUDA;
// driven on a GPU host by tools/cuda_build_test.sh with RETICOLO_NIGHTLY=1.

#include <reticolo/cuda/cuda.hpp>
#include <reticolo/reticolo.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

namespace {

// NB: no namespace-scope `using namespace reticolo;` — it pulls reticolo::log
// into global lookup and collides with CUDA's ::log in the device math headers.
// Each block scopes the using-directive inside its own function body instead.

struct Stat {
    double mean = 0.0;
    double err  = 0.0;
};

// Standard error of the mean via blocking (contiguous blocks absorb the HMC
// autocorrelation). Falls back to the naive estimator for short series.
Stat blocked(std::vector<double> const& x, int n_blocks = 20) {
    auto const n = static_cast<int>(x.size());
    if (n == 0) {
        return {};
    }
    if (n < 2 * n_blocks) {
        n_blocks = n;
    }
    int const w = n / n_blocks;
    double gm   = 0.0;
    std::vector<double> bm(static_cast<std::size_t>(n_blocks), 0.0);
    for (int b = 0; b < n_blocks; ++b) {
        double s = 0.0;
        for (int i = 0; i < w; ++i) {
            s += x[static_cast<std::size_t>((b * w) + i)];
        }
        bm[static_cast<std::size_t>(b)] = s / w;
        gm += bm[static_cast<std::size_t>(b)];
    }
    gm /= n_blocks;
    double var = 0.0;
    for (double const m : bm) {
        var += (m - gm) * (m - gm);
    }
    var /= (n_blocks - 1);
    return {.mean = gm, .err = std::sqrt(var / n_blocks)};
}

bool g_ok = true;

void check(char const* name, bool pass, char const* detail) {
    std::printf("  [%s] %-34s %s\n", pass ? "PASS" : "FAIL", name, detail);
    g_ok = g_ok && pass;
}

// <exp(-dH)> = 1 within a lenient nightly band (fat-tailed estimator).
void check_exp_dH(char const* tag, std::vector<double> const& edh) {
    Stat const s   = blocked(edh);
    double const d = std::abs(s.mean - 1.0);
    char buf[128];
    std::snprintf(buf, sizeof buf, "<exp(-dH)>=%.4f +/- %.4f", s.mean, s.err);
    check(tag, d < std::max(0.05, 4.0 * s.err), buf);
}

// CPU and GPU means agree within combined error (smoke, not a tight gate).
void check_equiv(char const* tag, std::vector<double> const& cpu, std::vector<double> const& gpu) {
    Stat const c     = blocked(cpu);
    Stat const g     = blocked(gpu);
    double const sig = std::hypot(c.err, g.err);
    double const nd  = sig > 0.0 ? std::abs(c.mean - g.mean) / sig : 0.0;
    char buf[160];
    std::snprintf(buf, sizeof buf, "cpu=%.6f gpu=%.6f  |d|/sigma=%.2f", c.mean, g.mean, nd);
    check(tag, nd < 5.0, buf);
}

void cold_identity_su2(reticolo::MatrixLinkLattice<reticolo::gauge_group::SU2, double>& u,
                       int ndim) {
    std::size_t const ns = u.nsites();
    std::fill(u.data(), u.data() + u.ncomponents(), 0.0);
    for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
        double* const blk = u.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;  // Re U_00
            blk[(6 * ns) + s] = 1.0;  // Re U_11
        }
    }
}

constexpr int k_n_therm = 1000;
constexpr int k_n_meas  = 2000;
constexpr double k_tau  = 1.0;
constexpr int k_n_md    = 20;

void phi4_block() {
    using namespace reticolo;
    std::printf("phi4 (L=8^4, kappa=0.18, lambda=1.0):\n");
    using DField = cuda::DeviceField<double>;
    using DAct   = cuda::DeviceAction<action::Phi4<double>, DField>;
    Lattice<double>::SizeVec const shape(4, 8);
    action::Phi4<double> const a{.kappa = 0.18, .lambda = 1.0};

    // GPU chain.
    std::vector<double> g_edh;
    std::vector<double> g_phi2;
    {
        DField f{shape};
        std::vector<double> const zero(f.size(), 0.0);
        f.copy_from_host(zero.data());
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        DAct meas{a, f.topology()};
        cuda::Hmc<DAct, alg::integ::Leapfrog, DField> hmc{
            DAct{a, f.topology()}, f, k_tau, k_n_md, 0x9114ULL};
        hmc.run(k_n_therm);
        hmc.sync();
        auto const v = static_cast<double>(f.size());
        auto const n = static_cast<long>(f.size());
        for (int i = 0; i < k_n_meas; ++i) {
            auto const r = hmc.step();
            g_edh.push_back(std::exp(-r.dH));
            g_phi2.push_back(cuda::reduce_sumsq_f64(f.data(), n) / v);
        }
    }

    // CPU chain (different RNG seed => independent realisation of same ensemble).
    std::vector<double> c_edh;
    std::vector<double> c_phi2;
    {
        Lattice<double> phi{shape};
        std::fill(phi.data(), phi.data() + phi.nsites(), 0.0);
        FastRng rng{0xC0DEULL};
        action::Phi4<double> ac{.kappa = 0.18, .lambda = 1.0};
        alg::Hmc hmc{ac, phi, rng, {.tau = k_tau, .n_md = k_n_md}, alg::integ::leapfrog};
        for (int i = 0; i < k_n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
        }
        for (int i = 0; i < k_n_meas; ++i) {
            auto const s = hmc.step(log::Mode::silent);
            c_edh.push_back(std::exp(-s.dH));
            c_phi2.push_back(obs::sq(phi));
        }
    }

    check_exp_dH("gpu <exp(-dH)>", g_edh);
    check_exp_dH("cpu <exp(-dH)>", c_edh);
    check_equiv("cpu-vs-gpu <phi^2>", c_phi2, g_phi2);
}

void su2_block() {
    using namespace reticolo;
    std::printf("su2 Wilson (L=6^4, beta=2.3):\n");
    using G           = gauge_group::SU2;
    using A           = action::Wilson<G, double>;
    using HField      = MatrixLinkLattice<G, double>;
    using DField      = cuda::DeviceField<double, cuda::MatrixLayout<G>>;
    using DAct        = cuda::DeviceAction<A, DField>;
    int const ndim    = 4;
    int const L       = 6;
    double const beta = 2.3;
    HField::SizeVec const shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    A const a{.beta = beta};

    HField cold{shape};
    cold_identity_su2(cold, ndim);
    std::size_t const ns = cold.nsites();
    auto const n_plaq    = (static_cast<std::size_t>(ndim) * (ndim - 1) / 2U) * ns;
    double const norm    = beta * static_cast<double>(n_plaq);

    std::vector<double> g_edh;
    std::vector<double> g_plaq;
    {
        DField f{shape};
        f.copy_from_host(cold.data());
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        DAct meas{a, f.topology()};
        // Omelyan2 (as in the canonical su2_hmc app): Leapfrog at this dt is past
        // the SU(2) stability edge from a cold start and rejects every trajectory.
        cuda::Hmc<DAct, alg::integ::Omelyan2, DField> hmc{
            DAct{a, f.topology()}, f, k_tau, k_n_md, 0x5202ULL};
        hmc.run(k_n_therm);
        hmc.sync();
        for (int i = 0; i < k_n_meas; ++i) {
            auto const r = hmc.step();
            g_edh.push_back(std::exp(-r.dH));
            g_plaq.push_back(1.0 - (meas.s_full(f) / norm));
        }
    }

    std::vector<double> c_edh;
    std::vector<double> c_plaq;
    {
        HField u{shape};
        cold_identity_su2(u, ndim);
        FastRng rng{0xBEEFULL};
        A ac{.beta = beta};
        alg::Hmc hmc{ac, u, rng, {.tau = k_tau, .n_md = k_n_md}, alg::integ::omelyan2};
        for (int i = 0; i < k_n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
        }
        for (int i = 0; i < k_n_meas; ++i) {
            auto const s = hmc.step(log::Mode::silent);
            c_edh.push_back(std::exp(-s.dH));
            c_plaq.push_back(1.0 - (ac.s_full(u) / norm));
        }
    }

    check_exp_dH("gpu <exp(-dH)>", g_edh);
    check_exp_dH("cpu <exp(-dH)>", c_edh);
    check_equiv("cpu-vs-gpu <plaq>", c_plaq, g_plaq);
}

void bose_gas_block() {
    using namespace reticolo;
    std::printf("BoseGas (L=6^4, m=1, lambda=0.5, mu=0.3):\n");
    using DField = cuda::DeviceField<cplx<double>>;
    using DAct   = cuda::DeviceAction<action::BoseGas<double>, DField>;
    Lattice<std::complex<double>>::SizeVec const shape(4, 6);
    action::BoseGas<double> const a{.mass = 1.0, .lambda = 0.5, .mu = 0.3};

    // GPU chain. ⟨|φ|²⟩ = Σ(re²+im²)/V over the 2·V underlying reals.
    std::vector<double> g_edh;
    std::vector<double> g_phi2;
    {
        Lattice<std::complex<double>> cold{shape};
        std::fill(cold.data(), cold.data() + cold.nsites(), std::complex<double>{0.0, 0.0});
        DField f{shape};
        f.copy_from_host(reinterpret_cast<cplx<double> const*>(cold.data()));
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        DAct meas{a, f.topology()};
        // Omelyan2 (as in the canonical bose_gas_hmc app): leapfrog at this
        // trajectory length is past the BoseGas stability edge from a cold start.
        cuda::Hmc<DAct, alg::integ::Omelyan2, DField> hmc{
            DAct{a, f.topology()}, f, k_tau, k_n_md, 0xB05EULL};
        hmc.run(k_n_therm);
        hmc.sync();
        auto const v = static_cast<double>(f.size());
        auto const n = 2 * static_cast<long>(f.size());
        for (int i = 0; i < k_n_meas; ++i) {
            auto const r = hmc.step();
            g_edh.push_back(std::exp(-r.dH));
            g_phi2.push_back(cuda::reduce_sumsq_f64(reinterpret_cast<double const*>(f.data()), n) /
                             v);
        }
    }

    // CPU chain.
    std::vector<double> c_edh;
    std::vector<double> c_phi2;
    {
        Lattice<std::complex<double>> phi{shape};
        std::fill(phi.data(), phi.data() + phi.nsites(), std::complex<double>{0.0, 0.0});
        FastRng rng{0xB1A5ULL};
        action::BoseGas<double> ac{.mass = 1.0, .lambda = 0.5, .mu = 0.3};
        alg::Hmc hmc{ac, phi, rng, {.tau = k_tau, .n_md = k_n_md}, alg::integ::omelyan2};
        auto const v = static_cast<double>(phi.nsites());
        for (int i = 0; i < k_n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
        }
        for (int i = 0; i < k_n_meas; ++i) {
            auto const s = hmc.step(log::Mode::silent);
            c_edh.push_back(std::exp(-s.dH));
            double acc = 0.0;
            for (std::size_t k = 0; k < phi.nsites(); ++k) {
                acc += std::norm(phi.data()[k]);
            }
            c_phi2.push_back(acc / v);
        }
    }

    check_exp_dH("gpu <exp(-dH)>", g_edh);
    check_exp_dH("cpu <exp(-dH)>", c_edh);
    check_equiv("cpu-vs-gpu <|phi|^2>", c_phi2, g_phi2);
}

}  // namespace

int main() {
    reticolo::log::off();
    std::printf("=== reticolo CUDA nightly physics harness ===\n");
    phi4_block();
    su2_block();
    bose_gas_block();
    std::printf("=== nightly harness: %s ===\n", g_ok ? "GREEN" : "RED");
    return g_ok ? 0 : 1;
}
