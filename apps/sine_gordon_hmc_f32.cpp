// Single-precision (mixed) HMC for the sine-Gordon scalar field — the
// mixed-precision counterpart of sine_gordon_hmc. The field, momentum, force
// and MD integration run in `float`; the action reduction S and the
// Hamiltonian / ΔH accumulate in `double`. Precision is derived from the
// lattice type: swap `Lattice<float>` for `Lattice<double>` for full double.
//
//  S = sum_x [ -2 kappa phi(x) Σ_{mu>0} phi(x+mu)  +  phi(x)^2
//              - alpha cos(phi(x)) ]
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory (double)
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<phi>|
//  /prod/obs/cos_phi     — <cos(phi)>

#include <reticolo/reticolo.hpp>

#include <cmath>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;

    // ---- CLI ----
    cli::Parser p{"sine_gordon_hmc_f32", "Single-precision (mixed) HMC for the sine-Gordon field"};
    auto const cf          = app::common_flags(p, {.out = "sine_gordon_f32.h5"});
    auto const& kappa      = p.opt<double>("kappa", 1.0, "hopping parameter");
    auto const& alpha      = p.opt<double>("alpha", 1.0, "cosine-potential strength");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: float lattice, RNG, float action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<float> phi{shape};
    act::SineGordon<float> sg{.kappa = static_cast<float>(kappa),
                              .alpha = static_cast<float>(alpha)};
    log::act(sg);

    // ---- Output: writer + series (all observables are double) ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto cos_phi  = out.series<double>("/prod/obs/cos_phi");

    // ---- Updater (precision deduced from the float lattice + action) ----
    updater::Hmc hmc{sg, phi, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(sg.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            auto const V        = static_cast<double>(phi.nsites());
            auto const [s1, sc] = obs::reduce(
                phi, obs::kernel::phi, [](auto z) { return std::cos(static_cast<double>(z)); });
            s_prod.append(sg.s_full(phi));
            mag.append(obs::mag_abs_of(s1, V));
            cos_phi.append(obs::mean_of(sc, V));
        }
    }
}
