// HMC for the phi^4 scalar field on the CUDA backend. The GPU twin of
// phi4_hmc.cpp: same physics and flags, the trajectory for-loop plainly here
// in main(). This is a .cu compiled by nvcc — it uses cuda::Hmc and
// DeviceField directly for the device work, and io::Writer directly for output
// (io::Writer PIMPLs HDF5, so nvcc never sees <hdf5.h>; it just links the
// prebuilt reticolo::io archive). Trajectories run host-free in blocks of
// `block`; observables are reduced on-device and only scalars cross PCIe.
//
// Output schema:
//  /run@*, /vars@*        — Writer reproducibility metadata + resolved flags
//  /therm/stats/s         — S_full per thermalisation block
//  /prod/obs/s            — S_full
//  /prod/obs/mag          — |<phi>|
//  /prod/obs/mag_sq       — (<phi>)^2
//  /prod/obs/m2           — <phi^2>
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
#include <cmath>
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
    using DField = cuda::DeviceField<double>;
    using DAct   = cuda::DeviceAction<act::Phi4<double>, DField>;

    cli::Parser p{"phi4_hmc_cuda", "Hybrid Monte Carlo for phi^4 on the CUDA backend"};
    auto const cf       = app::common_flags(p, {.out = "phi4_cuda.h5"});
    auto const& kappa   = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda  = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim    = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau     = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md    = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod  = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& block =
        p.opt<int>("block", 10, "trajectories per host-free block (graph replays between syncs)");
    auto const& ckpt_every =
        p.opt<int>("checkpoint_every", 0, "write a config every N prod trajectories (0 = off)");
    auto const& resume_path =
        p.opt<std::string>("resume", std::string{}, "resume from a previous config (.h5)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out);
    std::string const outpath = app::out_path(cf);

    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> host{shape};  // cold/resume staging + checkpoint copy-out

    std::uint64_t seed     = cf.seed;
    std::uint64_t counter0 = 0;
    long long start_i      = 0;
    bool const resuming    = !resume_path.empty();
    if (resuming) {
        if (io::load_field_shape(resume_path) != shape) {
            throw std::runtime_error{"--resume shape mismatch with --L/--ndim"};
        }
        start_i = io::load_config_counter(resume_path, host, seed, counter0);
        log::info("hmc", "resumed from {} at traj {}", resume_path, start_i);
    } else {
        std::fill(host.data(), host.data() + host.nsites(), 0.0);  // cold start phi = 0
    }

    act::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};
    log::act(phi4);

    DField field{shape};
    field.copy_from_host(host.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    DAct meas{phi4, field.topology()};  // measurement action (own scratch)
    cuda::Hmc<DAct, updater::integ::Leapfrog, DField> hmc{
        DAct{phi4, field.topology()}, field, tau, n_md, seed};
    if (resuming) {
        hmc.set_rng_counter(counter0);
    }
    auto const v = static_cast<double>(field.size());

    io::Writer out{outpath, argc, argv, &p};
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto mag     = out.series<double>("/prod/obs/mag");
    auto mag_sq  = out.series<double>("/prod/obs/mag_sq");
    auto m_sq    = out.series<double>("/prod/obs/m2");

    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", n_therm);
        for (int i = 0; i < n_therm; i += block) {
            hmc.run(std::min(block, n_therm - i));
            hmc.sync();
            s_therm.append(meas.s_full(field));
        }
    }

    log::info("hmc", "prod   {} trajectories (from {})", n_prod, start_i);
    for (long long i = start_i; i < n_prod; i += block) {
        int const k = static_cast<int>(std::min<long long>(block, n_prod - i));
        hmc.run(k);
        hmc.sync();
        s_prod.append(meas.s_full(field));
        // One fused pass: Σφ and Σφ² share the single read of the field.
        auto const [sum_phi, sum_phi_sq] =
            cuda::reduce(field, obs::kernel::phi, obs::kernel::phi_sq);
        double const mean = sum_phi / v;
        mag.append(std::abs(mean));
        mag_sq.append(mean * mean);
        m_sq.append(sum_phi_sq / v);
        long long const done = i + k;
        if (ckpt_every > 0 && done % ckpt_every == 0) {
            field.copy_to_host(host.data());
            RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
            io::save_config_counter(
                cfg_path(outpath, done), host, hmc.seed(), hmc.rng_counter(), done, argc, argv, &p);
        }
    }
    out.attr<double>("/prod/stats@acceptance", hmc.acceptance());
}
