// SU(2) Wilson gauge HMC on the CUDA backend. The GPU twin of su2_hmc.cpp: same
// physics and flags, the trajectory for-loop plainly here in main(). A .cu
// compiled by nvcc — cuda::Hmc over a MatrixLayout<SU2> field for the device
// work, io::Writer (HDF5 PIMPL'd, never seen by nvcc) for output. Trajectories
// run host-free in blocks of `block`; only the scalar action crosses PCIe,
// and ⟨P⟩ = 1 − S_W/(β·n_plaq) is derived in-app.
//
// Output schema:
//  /run@*, /vars@*        — Writer reproducibility metadata + resolved flags
//  /therm/stats/s         — S_full per thermalisation block
//  /prod/obs/s            — S_full
//  /prod/obs/plaq         — ⟨P⟩
//  /prod/stats@acceptance — cumulative production acceptance
//
// TWO DELIBERATE DIFFERENCES from the CPU twin, both consequences of host-free
// execution rather than oversights:
//   * `--block`, not the CPU's `--meas_every`. Trajectories replay inside one
//     CUDA graph with no host sync, so this is the count BETWEEN syncs — a loop
//     stride, not a measurement filter. Measurements happen once per block, so
//     the series hold n_prod/block rows, not n_prod.
//   * No per-trajectory `/prod/stats/dH` or `/prod/stats/accepted`. `run(k)`
//     overwrites a single 4-double energy buffer per trajectory, so only the
//     block's last trajectory survives; emitting them per trajectory would need
//     a sync per trajectory — precisely the floor this backend exists to
//     remove. The cumulative acceptance is written as an attribute instead.

#include <reticolo/cuda/cuda.hpp>
#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

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
    using Group  = math::group::SU2;
    using Action = action::Wilson<Group, double>;
    using HField = MatrixLinkLattice<Group, double>;
    using DField = cuda::DeviceField<double, cuda::MatrixLayout<Group>>;
    using DAct   = cuda::DeviceAction<Action, DField>;

    cli::Parser p{"su2_metropolis_cuda", "SU(2) Wilson action HMC on the CUDA backend"};
    auto const cf       = app::common_flags(p, {.L = 4, .out = "su2_cuda.h5"});
    auto const& ndim    = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta    = p.opt<double>("beta", 2.3, "Wilson coupling");
    auto const& sigma   = p.opt<double>("sigma", 0.3, "Metropolis proposal width");
    auto const& n_therm = p.opt<int>("n_therm", 200, "thermalisation sweeps");
    auto const& n_prod  = p.opt<int>("n_prod", 2000, "production sweeps");
    auto const& block =
        p.opt<int>("block", 10, "sweeps per host-free block (graph replays between syncs)");
    auto const& ckpt_every =
        p.opt<int>("checkpoint_every", 0, "write a config every N prod sweeps (0 = off)");
    auto const& resume_path =
        p.opt<std::string>("resume", std::string{}, "resume from a previous config (.h5)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out);
    std::string const outpath = app::out_path(cf);

    HField::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    HField host{shape};
    std::size_t const ns = host.nsites();

    std::uint64_t seed     = cf.seed;
    std::uint64_t counter0 = 0;
    long long start_i      = 0;
    bool const resuming    = !resume_path.empty();
    if (resuming) {
        if (io::load_field_shape(resume_path) != shape) {
            throw std::runtime_error{"--resume shape mismatch with --L/--ndim"};
        }
        start_i = io::load_config_counter(resume_path, host, seed, counter0);
        log::info("metr", "resumed from {} at sweep {}", resume_path, start_i);
    } else {
        // Cold start: every link = 2×2 identity (Re U_00 = Re U_11 = 1).
        set_cold_identity(host);
    }

    Action const action{.beta = beta};
    log::act(action);

    DField field{shape};
    field.copy_from_host(host.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    DAct meas{action, field.topology()};
    cuda::Metropolis<DAct> metro{DAct{action, field.topology()}, field, sigma, seed};
    if (resuming) {
        metro.set_rng_counter(counter0);
    }

    io::Writer out{outpath, argc, argv, &p};
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto plaq    = out.series<double>("/prod/obs/plaq");

    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * ns;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    if (!resuming) {
        log::info("metr", "therm  {} sweeps", n_therm);
        for (int i = 0; i < n_therm; i += block) {
            metro.run(std::min(block, n_therm - i));
            metro.sync();
            s_therm.append(meas.s_full(field));
        }
    }

    metro.reset_acceptance_stats();
    log::info("metr", "prod   {} sweeps (from {})", n_prod, start_i);
    for (long long i = start_i; i < n_prod; i += block) {
        int const k = static_cast<int>(std::min<long long>(block, n_prod - i));
        metro.run(k);
        metro.sync();
        double const s = meas.s_full(field);
        s_prod.append(s);
        plaq.append(1.0 - (s / plaq_norm));
        long long const done = i + k;
        if (ckpt_every > 0 && done % ckpt_every == 0) {
            field.copy_to_host(host.data());
            RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
            io::save_config_counter(cfg_path(outpath, done),
                                    host,
                                    metro.seed(),
                                    metro.rng_counter(),
                                    done,
                                    argc,
                                    argv,
                                    &p);
        }
    }
    out.attr<double>("/prod/stats@acceptance", metro.acceptance());
}
