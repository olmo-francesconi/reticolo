// HMC for the phi^4 scalar field on a `ndim`-dimensional hypercubic lattice
// (default ndim = 4; the example sweeps use ndim = 3 for cleaner FSS in the
// 3D Ising universality class).
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<phi>|           (per-config magnetization magnitude)
//  /prod/obs/mag_sq      — (<phi>)^2         (per-config magnetization squared; chi input)
//  /prod/obs/m2          — <phi^2>           (per-site field-squared average)

#include <reticolo/reticolo.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>

namespace {

std::string cfg_path(std::string const& out, long long i) {
    std::string stem = out;
    if (auto const pos = stem.rfind(".h5"); pos != std::string::npos && pos == stem.size() - 3) {
        stem.resize(pos);
    }
    std::array<char, 256> buf{};
    std::snprintf(buf.data(), buf.size(), "%s.cfg.%05lld.h5", stem.c_str(), i);
    return buf.data();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace reticolo;

    // ---- CLI ----
    cli::Parser p{"phi4_hmc", "Hybrid Monte Carlo for the phi^4 scalar field"};
    auto const& L          = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& kappa      = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath    = p.opt<std::string>("out", std::string{"phi4.h5"}, "HDF5 output path");
    auto const& ckpt_every =
        p.opt<int>("checkpoint_every", 0, "write a config every N prod trajectories (0 = off)");
    auto const& resume_path =
        p.opt<std::string>("resume", std::string{}, "resume from a previous config (.h5)");
    if (!p.parse(argc, argv))
        return 0;

    log::start(outpath);

    // ---- State: lattice, RNG, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Lattice<double> phi{shape};
    FastRng rng{seed};
    long long start_i   = 0;
    bool const resuming = !resume_path.empty();
    if (resuming) {
        auto const file_shape = io::load_field_shape(resume_path);
        if (file_shape != shape) {
            throw std::runtime_error{"--resume shape mismatch with --L/--ndim"};
        }
        start_i = io::load_config(resume_path, phi, rng);
        log::info("hmc", "resumed from {} at traj {}", resume_path, start_i);
    }

    act::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};
    log::act(phi4);

    // ---- Output: writer + series ----
    io::Writer out{outpath, argc, argv, &p};
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto mag_sq   = out.series<double>("/prod/obs/mag_sq");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    // ---- Updater ----
    alg::Hmc hmc{phi4, phi, rng, {.tau = tau, .n_md = n_md}};

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", n_therm);
        for (int i = 0; i < n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(phi4.s_full(phi));
        }
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories (from {})", n_prod, start_i);
    for (long long i = start_i; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(phi4.s_full(phi));
            mag.append(obs::mag::abs(phi));
            mag_sq.append(obs::sq_of_mean(phi));
            m_sq.append(obs::sq(phi));
        }
        if (ckpt_every > 0 && (i + 1) % ckpt_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, rng, i + 1, argc, argv, &p);
        }
    }
}
