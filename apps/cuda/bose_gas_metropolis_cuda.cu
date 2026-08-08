// Phase-quenched local Metropolis for the 4D relativistic lattice Bose gas at
// finite chemical potential, on the CUDA backend. The GPU twin of
// bose_gas_metropolis.cpp: samples the real (phase-quenched) part
// S_R = base.s_full by a checkerboard sweep instead of a trajectory, and records
// BOTH S_R and the imaginary observable S_I = base.s_imag per measurement — the
// reference the complex-LLR run reconstructs. The complex field is
// cplx<double> (2 reals/site), flat-copy-compatible with the host
// Lattice<std::complex<double>>.
//
// Output schema (the observable series match bose_gas_metropolis.cpp so
// examples/05_bose_gas_llr/analyze.py reads it unchanged):
//  /run@*, /vars@*        — Writer reproducibility metadata + resolved flags
//  /therm/stats/s         — S_R per thermalisation block
//  /prod/obs/s_r          — S_R
//  /prod/obs/s_i          — S_I
//  /prod/stats@acceptance — cumulative production acceptance
//
// TWO DELIBERATE DIFFERENCES from the CPU twin, both consequences of host-free
// execution rather than oversights:
//   * `--block`, not the CPU's `--meas_every`. Sweeps replay inside one CUDA
//     graph with no host sync, so this is the count BETWEEN syncs — a loop
//     stride, not a measurement filter. Measurements happen once per block, so
//     the series hold n_prod/block rows, not n_prod.
//   * No per-sweep `/prod/stats/acceptance` series (the CPU twin's
//     `step().acceptance()` per sweep). `cuda::Metropolis` carries {S, accepted,
//     attempts} incrementally on the device across every sweep in the block;
//     reading a per-sweep value would need a sync per sweep — precisely the
//     floor this backend exists to remove. The cumulative acceptance over the
//     whole run is written as an attribute instead.

#include <reticolo/cuda/cuda.hpp>
#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

int main(int argc, char** argv) {
    using namespace reticolo;
    using DField = cuda::DeviceField<cplx<double>>;
    using DAct   = cuda::DeviceAction<act::BoseGas<double>, DField>;

    cli::Parser p{"bose_gas_metropolis_cuda",
                  "Phase-quenched HMC for the 4D Bose gas on CUDA (S_R + S_I)"};
    auto const cf       = app::common_flags(p, {.L = 4, .out = "bose_gas_cuda.h5"});
    auto const& ndim    = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass    = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda  = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu      = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& sigma   = p.opt<double>("sigma", 0.3, "Metropolis proposal width");
    auto const& n_therm = p.opt<int>("n_therm", 500, "thermalisation sweeps");
    auto const& n_prod  = p.opt<int>("n_prod", 5000, "production sweeps");
    auto const& block =
        p.opt<int>("block", 10, "sweeps per host-free block (graph replays between syncs)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out);
    std::string const outpath = app::out_path(cf);

    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(cf.L));
    Lattice<std::complex<double>> host{shape};  // cold-start staging (phi = 0)
    std::fill(host.data(), host.data() + host.nsites(), std::complex<double>{0.0, 0.0});

    act::BoseGas<double> const action{.mass = mass, .lambda = lambda, .mu = mu};
    log::act(action);

    DField field{shape};
    field.copy_from_host(reinterpret_cast<cplx<double> const*>(host.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    DAct meas{action, field.topology()};  // measurement action (own scratch)
    cuda::Metropolis<DAct> metro{DAct{action, field.topology()}, field, sigma, cf.seed};

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm = out.series<double>("/therm/stats/s");
    auto s_r     = out.series<double>("/prod/obs/s_r");
    auto s_i     = out.series<double>("/prod/obs/s_i");

    log::info("metr", "therm  {} sweeps", n_therm);
    for (int i = 0; i < n_therm; i += block) {
        metro.run(std::min(block, n_therm - i));
        metro.sync();
        s_therm.append(meas.s_full(field));
    }

    metro.reset_acceptance_stats();
    log::info("metr", "prod   {} sweeps", n_prod);
    for (int i = 0; i < n_prod; i += block) {
        metro.run(std::min(block, n_prod - i));
        metro.sync();
        s_r.append(meas.s_full(field));
        s_i.append(meas.s_imag(field));
    }
    out.attr<double>("/prod/stats@acceptance", metro.acceptance());
}
