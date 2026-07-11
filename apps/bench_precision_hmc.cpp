// Single- vs double-precision phi^4 HMC, side by side.
//
// Mixed-precision HMC stores and integrates the field in `float` (half the
// memory traffic, double-width SIMD) but accumulates the action reduction and
// the Hamiltonian / ΔH in `double`, so the Metropolis acceptance still targets
// exp(-S). This rig quantifies the trade: per-trajectory wall time and
// throughput (the speed win, largest in the bandwidth-bound large-V regime),
// against acceptance and the ΔH spread (the cost — single-precision MD
// conserves energy slightly worse, so ΔH widens and acceptance drops at fixed
// n_md). The <phi^2> columns are the correctness check: both must agree.

#include <reticolo/reticolo.hpp>

#include "_bench/timing.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using namespace reticolo;

struct Case {
    int ndim;
    int L;
    int n_md;  // scaled ~V^{1/4} per case so acceptance stays comparable across volumes
               // (leapfrog <ΔH> ∝ V·dt⁴, so fixed n_md would collapse acceptance at large V)
};

struct Result {
    double per_traj;  // s / trajectory
    double thru;      // site-MD-updates / s
    double acc;       // acceptance rate
    double dh_mean;   // <ΔH>
    double dh_std;    // std(ΔH)
    double phi2;      // equilibrium <phi^2>
};

template <class T>
Result measure(Case c, double kappa, double lambda, double tau) {
    int const n_md = c.n_md;
    typename Lattice<T>::SizeVec shape(static_cast<std::size_t>(c.ndim),
                                       static_cast<std::size_t>(c.L));
    Lattice<T> phi{shape};
    act::Phi4<T> const action{.kappa = static_cast<T>(kappa), .lambda = static_cast<T>(lambda)};
    updater::Hmc hmc{action, phi, FastRng{1234}, {.tau = tau, .n_md = n_md}};

    for (int i = 0; i < 200; ++i) {
        (void)hmc.step(log::Mode::silent);  // thermalise
    }

    // Physics + ΔH statistics.
    constexpr int n_meas = 300;
    long accepted        = 0;
    double sum_dh        = 0.0;
    double sum_dh2       = 0.0;
    double sum2          = 0.0;
    for (int m = 0; m < n_meas; ++m) {
        auto const st = hmc.step(log::Mode::silent);
        accepted += st.accepted ? 1 : 0;
        sum_dh += st.dH;
        sum_dh2 += st.dH * st.dH;
        sum2 += obs::sq(phi);
    }
    double const n       = static_cast<double>(n_meas);
    double const dh_mean = sum_dh / n;
    double const dh_var  = (sum_dh2 / n) - (dh_mean * dh_mean);

    // Timing.
    double const per  = bench::time_per_call([&] { (void)hmc.step(log::Mode::silent); });
    double const thru = static_cast<double>(phi.nsites()) * static_cast<double>(n_md) / per;

    return {.per_traj = per,
            .thru     = thru,
            .acc      = static_cast<double>(accepted) / n,
            .dh_mean  = dh_mean,
            .dh_std   = std::sqrt(dh_var > 0.0 ? dh_var : 0.0),
            .phi2     = sum2 / n};
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();

    std::vector<Case> const cases{{3, 24, 20}, {3, 48, 40}, {4, 12, 24}, {4, 16, 32}};
    double const kappa  = 0.12;
    double const lambda = 1.0;
    double const tau    = 1.0;

    std::printf("Phi4 HMC: double vs float  (kappa=%.2f lambda=%.2f tau=%.1f, n_md ~ V^1/4)\n\n",
                kappa,
                lambda,
                tau);
    std::printf("%-11s %-9s  %-11s %-11s %-8s  %-8s %-8s  %-9s %-9s\n",
                "ndim x L",
                "V",
                "f64 [s]",
                "f32 [s]",
                "speedup",
                "f64 acc",
                "f32 acc",
                "f64 σ(ΔH)",
                "f32 σ(ΔH)");
    std::printf("%-11s %-9s  %-11s %-11s %-8s  %-8s %-8s  %-9s %-9s\n",
                "--------",
                "-",
                "-------",
                "-------",
                "-------",
                "-------",
                "-------",
                "-------",
                "-------");

    for (auto const& c : cases) {
        auto const d  = measure<double>(c, kappa, lambda, tau);
        auto const f  = measure<float>(c, kappa, lambda, tau);
        std::size_t v = 1;
        for (int i = 0; i < c.ndim; ++i) {
            v *= static_cast<std::size_t>(c.L);
        }
        std::printf("%-2dD x %-5d %-9zu  %-11.3e %-11.3e %-8.2f  %-8.3f %-8.3f  %-9.3e %-9.3e\n",
                    c.ndim,
                    c.L,
                    v,
                    d.per_traj,
                    f.per_traj,
                    d.per_traj / f.per_traj,
                    d.acc,
                    f.acc,
                    d.dh_std,
                    f.dh_std);
        std::printf("    throughput:  f64=%.1f M/s  f32=%.1f M/s    <phi^2>:  f64=%.4f  f32=%.4f\n",
                    d.thru / 1e6,
                    f.thru / 1e6,
                    d.phi2,
                    f.phi2);
    }
}
