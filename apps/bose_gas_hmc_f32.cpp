// Single-precision (mixed) phase-quenched HMC for the 4D self-interacting
// relativistic lattice Bose gas at finite chemical potential — the
// mixed-precision counterpart of bose_gas_hmc. The complex field, momentum,
// force and MD integration run in `float`; the action reductions S_R / S_I and
// the Hamiltonian / ΔH accumulate in `double`. Precision is derived from the
// lattice type: swap `complex<float>` for `complex<double>` for full double.

#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<float>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_hmc_f32",
                  "Single-precision (mixed) phase-quenched HMC for the 4D Bose gas"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "bose_gas_hmc_f32.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass       = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu         = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 10, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 500, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 5000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: float complex lattice, RNG, float action ----
    Lattice<std::complex<float>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                static_cast<std::size_t>(cf.L));
    Lattice<std::complex<float>> phi{shape};
    Action const action{.mass   = static_cast<float>(mass),
                        .lambda = static_cast<float>(lambda),
                        .mu     = static_cast<float>(mu)};
    log::act(action);

    // ---- Output: writer + series (all observables are double) ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_r      = out.series<double>("/prod/obs/s_r");
    auto s_i      = out.series<double>("/prod/obs/s_i");

    // ---- Updater (precision deduced from the float complex lattice + action) ----
    updater::Hmc hmc{action, phi, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(action.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_r.append(action.s_full(phi));
            s_i.append(action.s_imag(phi));
        }
    }
}
