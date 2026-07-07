// SU(2) Wilson gauge HMC on the CUDA backend. The GPU twin of su2_hmc.cpp: same
// CLI + output schema, the trajectory for-loop plainly here in main(). A .cu
// compiled by nvcc — cuda::Hmc over a MatrixLayout<SU2> field for the device
// work, io::Writer (HDF5 PIMPL'd, never seen by nvcc) for output. Trajectories
// run host-free in blocks of `meas_every`; only the scalar action crosses PCIe,
// and ⟨P⟩ = 1 − S_W/(β·n_plaq) is derived in-app.
//
// Output schema:
//  /run@*, /vars@*        — Writer reproducibility metadata + resolved flags
//  /therm/stats/s         — S_full per thermalisation block
//  /prod/obs/s            — S_full
//  /prod/obs/plaq         — ⟨P⟩
//  /prod/stats@acceptance — cumulative production acceptance

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

    cli::Parser p{"su2_hmc_cuda", "SU(2) Wilson action HMC on the CUDA backend"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "su2_cuda.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 2.3, "Wilson coupling");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 10, "trajectories per host-free block");
    auto const& ckpt_every =
        p.opt<int>("checkpoint_every", 0, "write a config every N prod trajectories (0 = off)");
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
        log::info("hmc", "resumed from {} at traj {}", resume_path, start_i);
    } else {
        // Cold start: every link = 2×2 identity (Re U_00 = Re U_11 = 1).
        std::fill(host.data(), host.data() + host.ncomponents(), 0.0);
        for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
            double* const blk = host.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
                blk[(0 * ns) + s] = 1.0;
                blk[(6 * ns) + s] = 1.0;
            }
        }
    }

    Action const action{.beta = beta};
    log::act(action);

    DField field{shape};
    field.copy_from_host(host.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    DAct meas{action, field.topology()};
    cuda::Hmc<DAct, alg::integ::Leapfrog, DField> hmc{
        DAct{action, field.topology()}, field, tau, n_md, seed};
    if (resuming) {
        hmc.set_rng_counter(counter0);
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
        log::info("hmc", "therm  {} trajectories", n_therm);
        for (int i = 0; i < n_therm; i += meas_every) {
            hmc.run(std::min(meas_every, n_therm - i));
            hmc.sync();
            s_therm.append(meas.s_full(field));
        }
    }

    log::info("hmc", "prod   {} trajectories (from {})", n_prod, start_i);
    for (long long i = start_i; i < n_prod; i += meas_every) {
        int const k = static_cast<int>(std::min<long long>(meas_every, n_prod - i));
        hmc.run(k);
        hmc.sync();
        double const s = meas.s_full(field);
        s_prod.append(s);
        plaq.append(1.0 - (s / plaq_norm));
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
