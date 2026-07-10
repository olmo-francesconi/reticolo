#include <reticolo/action/site/phi4.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/cuda/actions/site/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/graph.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Eager device HMC end-to-end on Phi4 (f64).
//
// NOTE on the "zero integrator-specific CUDA code" lint gate: this is the one
// TU that NAMES the integrator tags (alg::integ::Leapfrog / Omelyan2 /
// Omelyan4) — to instantiate the GENERIC integrator over DeviceField and prove
// reuse. That is the opposite of integrator-specific kernel code. The gate
// (tests/cuda/check_no_integrator_kernels.cmake) greps the kernel sources and
// excludes this probe by name.

namespace reticolo::cuda {

namespace {

using Phi4   = action::Phi4<double>;
using DField = DeviceField<double>;
using DAct   = DeviceAction<Phi4, DField>;

constexpr double kKappa  = 0.18;
constexpr double kLambda = 1.0;

Phi4 make_action() {
    Phi4 a{};
    a.kappa  = kKappa;
    a.lambda = kLambda;
    return a;
}

Lattice<double> smooth_field(std::vector<std::size_t> const& shape) {
    Lattice<double> l{shape};
    double* const d = l.data();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        d[i] = 0.4 * std::sin(0.3 * static_cast<double>(i) + 1.0);
    }
    return l;
}

double hamiltonian(DAct const& act, DField const& field, DField const& mom) {
    double const kinetic = 0.5 * reduce_sumsq_f64(mom.data(), static_cast<long>(mom.size()));
    return kinetic + act.s_full(field);
}

}  // namespace

bool hmc_reversibility_ok() {
    std::vector<std::size_t> const shape{6, 4, 5};
    Lattice<double> const init = smooth_field(shape);
    auto const n               = init.nsites();

    DField field{shape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{make_action(), field.topology()};

    field.copy_from_host(init);
    std::vector<double> p(n);
    FastRng rng{7};
    rng.normal_fill(p.data(), n);
    mom.copy_from_host(p.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    Lattice<double> f0{shape};
    field.copy_to_host(f0);
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    using Integ          = alg::integ::Leapfrog;
    constexpr double tau = 1.0;
    constexpr int n_md   = 12;

    Integ::run(act, field, mom, force, tau, n_md);
    axpy_f64(-2.0, mom.data(), mom.data(), static_cast<long>(n));  // p -> -p
    Integ::run(act, field, mom, force, tau, n_md);

    Lattice<double> f1{shape};
    field.copy_to_host(f1);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(f1.data()[i] - f0.data()[i]) > 1e-9) {
            return false;
        }
    }
    return true;
}

namespace {

template <class Integ>
double mean_abs_dH(double tau, int n_md, int n_samples, unsigned seed) {
    std::vector<std::size_t> const shape{4, 4, 4, 4};
    Lattice<double> const init = smooth_field(shape);
    auto const n               = init.nsites();

    DField field{shape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{make_action(), field.topology()};

    FastRng rng{seed};
    std::vector<double> p(n);
    double sum = 0.0;
    for (int k = 0; k < n_samples; ++k) {
        field.copy_from_host(init);  // reset q0
        rng.normal_fill(p.data(), n);
        mom.copy_from_host(p.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

        double const h0 = hamiltonian(act, field, mom);
        Integ::run(act, field, mom, force, tau, n_md);
        double const h1 = hamiltonian(act, field, mom);
        sum += std::abs(h1 - h0);
    }
    return sum / n_samples;
}

// Observed order p in |ΔH| ~ dt^p between dt = tau/n_md and tau/(2·n_md).
template <class Integ>
double observed_order(double tau, int n_md_coarse, int n_samples, unsigned seed) {
    double const dH_coarse = mean_abs_dH<Integ>(tau, n_md_coarse, n_samples, seed);
    double const dH_fine   = mean_abs_dH<Integ>(tau, 2 * n_md_coarse, n_samples, seed);
    return std::log2(dH_coarse / dH_fine);  // dt halves coarse -> fine
}

}  // namespace

bool integrator_order_ok() {
    constexpr double tau  = 1.0;
    constexpr int n_md    = 8;  // coarse dt = 1/8; fine = 1/16 (both above the f64 floor)
    constexpr int samples = 16;
    constexpr unsigned sd = 2026;

    double const p_lf = observed_order<alg::integ::Leapfrog>(tau, n_md, samples, sd);
    double const p_o2 = observed_order<alg::integ::Omelyan2>(tau, n_md, samples, sd);
    double const p_o4 = observed_order<alg::integ::Omelyan4>(tau, n_md, samples, sd);

    return (std::abs(p_lf - 2.0) < 0.4) && (std::abs(p_o2 - 2.0) < 0.4) &&
           (std::abs(p_o4 - 4.0) < 0.6);
}

bool hmc_step_runs() {
    std::vector<std::size_t> const shape{6, 4, 5};
    Lattice<double> const init = smooth_field(shape);

    DField field{shape};
    field.copy_from_host(init);
    DAct act{make_action(), field.topology()};

    Hmc<DAct> hmc{std::move(act), field, 1.0, 10};
    for (int step = 0; step < 5; ++step) {
        HmcResult const r = hmc.step();
        if (!std::isfinite(r.dH)) {
            return false;
        }
    }

    Lattice<double> out{shape};
    field.copy_to_host(out);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < out.nsites(); ++i) {
        if (!std::isfinite(out.data()[i])) {
            return false;
        }
    }
    return true;
}

// Run K trajectories fully host-free (one graph replay each, no per-step sync,
// device-side Metropolis accept). Two independent runs from the
// same seed and same initial config must agree bit-for-bit — the whole chain
// (momenta, MD, accept uniform, rollback, counter bump) is deterministic on the
// device — and the acceptance must be a sane fraction with a finite field.
bool hmc_device_run_deterministic() {
    std::vector<std::size_t> const shape{6, 4, 5};
    Lattice<double> const init = smooth_field(shape);
    constexpr int kTraj        = 32;

    auto run_chain = [&](Lattice<double>& out) -> double {
        DField field{shape};
        field.copy_from_host(init);
        DAct act{make_action(), field.topology()};
        Hmc<DAct> hmc{std::move(act), field, 1.0, 10, 2026ULL};
        hmc.run(kTraj);  // host-free: K replays, no sync between them
        double const acc = hmc.acceptance();
        hmc.sync();
        field.copy_to_host(out);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        return acc;
    };

    Lattice<double> a{shape};
    Lattice<double> b{shape};
    double const acc_a = run_chain(a);
    double const acc_b = run_chain(b);

    if (acc_a <= 0.0 || acc_a > 1.0 || acc_a != acc_b) {
        return false;
    }
    for (std::size_t i = 0; i < a.nsites(); ++i) {
        if (!std::isfinite(a.data()[i]) || a.data()[i] != b.data()[i]) {
            return false;
        }
    }
    return true;
}

bool graph_replay_matches_eager() {
    std::vector<std::size_t> const shape{6, 4, 5};
    Lattice<double> const q0 = smooth_field(shape);
    auto const n             = q0.nsites();
    std::vector<double> p0(n);
    FastRng rng{5};
    rng.normal_fill(p0.data(), n);

    constexpr double tau = 1.0;
    constexpr int n_md   = 12;
    using Integ          = alg::integ::Leapfrog;

    // Eager reference MD from (q0, p0), on the default stream.
    Lattice<double> eager_out{shape};
    {
        DField field{shape};
        DField mom{field.topology()};
        DField force{field.topology()};
        DAct const act{make_action(), field.topology()};
        field.copy_from_host(q0);
        mom.copy_from_host(p0.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        Integ::run(act, field, mom, force, tau, n_md);
        field.copy_to_host(eager_out);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Captured MD on a dedicated stream. Iter 0 captures + launches; iter 1
    // replays the cached graphExec (body not re-run). Both reset to (q0, p0)
    // and must reproduce the eager field bit-for-bit.
    cudaStream_t s = nullptr;
    RETICOLO_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    DField field{shape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{make_action(), field.topology()};
    TrajectoryGraph graph{s};

    bool ok = true;
    for (int iter = 0; iter < 2; ++iter) {
        field.copy_from_host(q0);
        mom.copy_from_host(p0.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        {
            ScopedStream const scope{s};
            graph.run([&] { Integ::run(act, field, mom, force, tau, n_md); });
        }
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(s));

        Lattice<double> g_out{shape};
        field.copy_to_host(g_out);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        for (std::size_t i = 0; i < n; ++i) {
            if (g_out.data()[i] != eager_out.data()[i]) {
                ok = false;
            }
        }
    }
    cudaStreamDestroy(s);
    return ok;
}

}  // namespace reticolo::cuda

// Leapfrog MD on DeviceField is time-reversible to roundoff.
TEST_CASE("cuda HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_reversibility_ok());
}

// The reused alg::integ tags give |dH| ~ dt^p with p = 2/2/4 on the device.
TEST_CASE("cuda integrator order is 2/2/4", "[cuda]") {
    REQUIRE(reticolo::cuda::integrator_order_ok());
}

// cuda::Hmc::step() (sample -> MD -> dH -> host MH) runs and stays finite.
TEST_CASE("cuda Hmc step runs end-to-end", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_step_runs());
}

// A graph-captured MD trajectory + its replay reproduce eager MD bit-for-bit.
TEST_CASE("cuda graph replay matches eager MD", "[cuda]") {
    REQUIRE(reticolo::cuda::graph_replay_matches_eager());
}

// Host-free trajectory streaming (device-side Metropolis accept) is
// deterministic and produces a sane chain.
TEST_CASE("cuda host-free HMC run is deterministic", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_device_run_deterministic());
}
