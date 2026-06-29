// Sequential vs checkerboard Metropolis on Phi4, side by side.
//
// The two are distinct (valid) Markov chains: the sequential sweep updates
// sites in lexicographic order with each site seeing its predecessors'
// accepts; the checkerboard sweep updates the even sublattice then the odd,
// each against the frozen opposite colour, which lets it batch the Gaussian
// proposal (normal_fill) and the exp(-ds) acceptance weight (Sleef exp_batch)
// off the scalar critical path.
//
// This rig prints, per volume: per-sweep wall time, throughput, the speedup,
// the acceptance rate, and the equilibrium <phi^2> for both schemes. The last
// column is the correctness check — both must land on the same <phi^2>.

#include <reticolo/reticolo.hpp>

#include "_bench/timing.hpp"

#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using namespace reticolo;

struct Case {
    int ndim;
    int L;
};

struct Result {
    double per_sweep;  // s / sweep
    double thru;       // site-updates / s
    double acc;        // acceptance rate
    double phi2;       // equilibrium <phi^2>
};

Result measure(alg::Sweep sweep, Case c, double kappa, double lambda, double sigma) {
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(c.ndim), static_cast<std::size_t>(c.L));
    Lattice<double> phi{shape};
    FastRng rng{42};
    act::Phi4<double> const action{.kappa = kappa, .lambda = lambda};
    for (Site x : phi.sites()) {
        phi[x] = 0.1 * rng.normal();
    }

    alg::Metropolis<act::Phi4<double>, FastRng> mc{
        action, phi, rng, alg::MetropolisSpec{.sigma = sigma, .sweep = sweep}};

    for (int i = 0; i < 500; ++i) {
        (void)mc.step();  // thermalise
    }

    // Physics: acceptance + <phi^2> over a measurement window.
    constexpr int n_meas = 400;
    std::size_t acc      = 0;
    std::size_t att      = 0;
    double sum2          = 0.0;
    for (int m = 0; m < n_meas; ++m) {
        auto const st = mc.step();
        acc += st.n_accepted;
        att += st.n_attempts;
        for (Site x : phi.sites()) {
            sum2 += phi[x] * phi[x];
        }
    }
    double const phi2 = sum2 / (static_cast<double>(n_meas) * static_cast<double>(phi.nsites()));
    double const acc_rate = att == 0 ? 0.0 : static_cast<double>(acc) / static_cast<double>(att);

    // Timing: best-of-N per-sweep wall time.
    double const per  = bench::time_per_call([&] { (void)mc.step(); });
    double const thru = static_cast<double>(phi.nsites()) / per;

    return {.per_sweep = per, .thru = thru, .acc = acc_rate, .phi2 = phi2};
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();  // keep step-format cost out of the timing

    std::vector<Case> const cases{{2, 64}, {2, 256}, {3, 32}, {4, 12}};
    double const kappa  = 0.10;
    double const lambda = 0.5;
    double const sigma  = 0.7;

    std::printf("Phi4 Metropolis: sequential vs checkerboard "
                "(kappa=%.2f lambda=%.2f sigma=%.2f)\n\n",
                kappa,
                lambda,
                sigma);
    std::printf("%-12s %-10s  %-12s %-12s %-8s  %-11s %-11s  %-9s %-9s\n",
                "ndim x L",
                "V",
                "seq [s]",
                "chk [s]",
                "speedup",
                "seq M/s",
                "chk M/s",
                "seq acc",
                "chk acc");
    std::printf("%-12s %-10s  %-12s %-12s %-8s  %-11s %-11s  %-9s %-9s\n",
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
        auto const s  = measure(alg::Sweep::Sequential, c, kappa, lambda, sigma);
        auto const k  = measure(alg::Sweep::Checkerboard, c, kappa, lambda, sigma);
        std::size_t v = 1;
        for (int d = 0; d < c.ndim; ++d) {
            v *= static_cast<std::size_t>(c.L);
        }
        std::printf("%-2dD x %-6d %-10zu  %-12.3e %-12.3e %-8.2f  %-11.1f %-11.1f  %-9.3f %-9.3f\n",
                    c.ndim,
                    c.L,
                    v,
                    s.per_sweep,
                    k.per_sweep,
                    s.per_sweep / k.per_sweep,
                    s.thru / 1e6,
                    k.thru / 1e6,
                    s.acc,
                    k.acc);
        std::printf("    <phi^2>:  seq=%.4f  chk=%.4f  (must match)\n", s.phi2, k.phi2);
    }
}
