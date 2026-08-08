// Parameter-span orchestration: one binary that runs many phi^4 HMC chains
// concurrently, each pinned to a different hopping parameter kappa across
// [kappa_min, kappa_max]. Thin app over the generic `orch::span` orchestrator —
// it builds the kappa grid (`Spec::points`) and stamps /cfg, then calls
// `setup()`/`run()`, which build the workers and drive the schedule. No LLR, no
// window, no exchange; an in-process replacement for an outer bash sweep,
// writing one HDF5 file.
//
// Output schema:
//  /run@*, /vars@*        — reproducibility metadata + resolved flags
//  /cfg@n_workers, @lambda, @kappa_min, @kappa_max
//  /cfg/kappa             — series, length n_workers (each worker's kappa)
//  /worker_NNN/stats/dH   — per-trajectory ΔH
//  /worker_NNN/stats/accepted
//  /worker_NNN/obs/s      — S_full (recorded by the driver for any action)
//  /worker_NNN/obs/mag    — |<phi>|
//  /worker_NNN/obs/mag_sq — (<phi>)^2
//  /worker_NNN/obs/m2     — <phi^2>

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <format>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::Phi4<double>;
    using Field  = Lattice<double>;
    using Span   = orch::span::Orchestrator<Action, FastRng, updater::HmcSampler<>>;

    // ---- CLI ----
    cli::Parser p{"param_span_hmc", "Concurrent phi^4 HMC chains over a kappa span"};
    auto const cf          = app::common_flags(p, {.L = 8, .out = "param_span.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling (shared)");
    auto const& kappa_min  = p.opt<double>("kappa_min", 0.10, "lower hopping parameter");
    auto const& kappa_max  = p.opt<double>("kappa_max", 0.26, "upper hopping parameter");
    auto const& n_workers  = p.opt<int>("n_workers", 8, "number of kappa points (workers)");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& worker_threads =
        p.opt<int>("worker_threads", 1, "HMC threads per worker (1 = serial; 0 = auto-balance)");
    auto const& slabs =
        p.opt<int>("slabs_per_thread", 0, "HMC slab granularity per thread (0 = 1)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv);
    auto const n_w = static_cast<std::size_t>(n_workers < 1 ? 1 : n_workers);

    // ---- The grid: one action per worker, kappa linear across the span. That is
    //      the app's whole job here; ids, per-worker seeds and the thread plan are
    //      the orchestrator's, and it derives them in setup(). ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    auto points = orch::span::scan(
        Action{.lambda = lambda}, &Action::kappa, kappa_min, kappa_max, static_cast<int>(n_w));

    // ---- /cfg metadata: the swept kappa values ----
    out.attr<int>("/cfg@n_workers", static_cast<int>(n_w));
    out.attr<double>("/cfg@lambda", lambda);
    out.attr<double>("/cfg@kappa_min", kappa_min);
    out.attr<double>("/cfg@kappa_max", kappa_max);
    auto kappa_series = out.series<double>("/cfg/kappa");
    for (auto const& a : points) {
        kappa_series.append(a.kappa);
    }

    // ---- Drive: concurrent therm + prod, recording phi^4 observables ----
    std::vector<orch::span::Observable<Field>> obs{
        {.name = "mag",
         .measure =
             [](Field const& f) {
                 return obs::mag_abs_of(std::get<0>(obs::reduce(f, obs::kernel::phi)),
                                        static_cast<double>(f.nsites()));
             }},
        {.name = "mag_sq",
         .measure =
             [](Field const& f) {
                 return obs::sq_of_mean_of(std::get<0>(obs::reduce(f, obs::kernel::phi)),
                                           static_cast<double>(f.nsites()));
             }},
        {.name = "m2", .measure = [](Field const& f) {
             return obs::mean_of(std::get<0>(obs::reduce(f, obs::kernel::phi_sq)),
                                 static_cast<double>(f.nsites()));
         }}};
    Span runner{Span::Spec{.shape          = shape,
                           .seed           = cf.seed,
                           .points         = std::move(points),
                           .sampler        = {.tau = tau, .n_md = n_md, .slabs_per_thread = slabs},
                           .worker_threads = worker_threads},
                std::move(obs)};
    runner.setup(out);
    runner.run(
        orch::span::Schedule{.n_therm = n_therm, .n_prod = n_prod, .meas_every = meas_every});
}
