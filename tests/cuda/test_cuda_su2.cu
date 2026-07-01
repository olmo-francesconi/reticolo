#include <reticolo/action/detail/gauge/gauge_group/su2.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/wilson.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/gauge/su2_device.cuh>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/math/su2_ops.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// SU(2) Wilson gauge HMC on the device. The matrix link field is [ndim][nc][nsites]
// (MatrixLayout<SU2>, nc=8), identical to the host MatrixLinkLattice<SU2> order —
// a flat copy round-trips. The force is the per-link staple gather + TA[U·V], the
// action a per-site forward-plane sum, the drift a group exponential. This TU
// validates the device matrix ops against math::su2, the device action/force against
// the CPU Wilson<SU2>, MD energy conservation + reversibility, the Gell-Mann
// momentum moments, and the generic host-free Hmc instantiation. (Excluded from the
// no-integrator-kernels lint gate: it names alg::integ::Leapfrog to instantiate the
// generic integrator over the matrix field, as hmc_probe.cu / u1_probe.cu.)

namespace reticolo::cuda {

namespace {

using SU2    = gauge_group::SU2;
using Wil    = action::Wilson<SU2, double>;
using DField = DeviceField<double, MatrixLayout<SU2>>;
using DAct   = DeviceAction<Wil, DField>;

std::vector<std::size_t> const kShape{4, 4, 4, 4};
constexpr double kBeta = 2.4;

// Identity per link, then drift each direction by a random algebra element —
// a non-trivial but valid SU(2) on every link (the canonical host hot start).
void hot_start(MatrixLinkLattice<SU2, double>& u, FastRng& rng) {
    std::size_t const d     = u.ndims();
    std::size_t const ns    = u.nsites();
    std::size_t const total = d * SU2::n_real_components * ns;
    double* const data      = u.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = u.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;  // Re U_{00}
            blk[(6 * ns) + s] = 1.0;  // Re U_{11}
        }
    }
    std::vector<double> scratch(SU2::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su2::sample_algebra_slab(scratch.data(), rng, ns);
        math::su2::expi_lmul_slab(u.mu_block_data(mu), scratch.data(), 0.5, ns);
    }
}

MatrixLinkLattice<SU2, double> make_links() {
    MatrixLinkLattice<SU2, double> u{Indexing::SizeVec(kShape.begin(), kShape.end())};
    FastRng rng{2718};
    hot_start(u, rng);
    return u;
}

}  // namespace

bool su2_device_ops_match_cpu() {
    // Two arbitrary 2×2 complex matrices (8 reals each) — not unitary; the ops
    // are linear-algebra primitives, validated on generic input.
    FastRng rng{12345};
    double a[8];
    double b[8];
    for (double& x : a) {
        x = rng.normal();
    }
    for (double& x : b) {
        x = rng.normal();
    }

    auto close = [](double const* p, double const* q) {
        for (int k = 0; k < 8; ++k) {
            if (std::abs(p[k] - q[k]) > 1e-12) {
                return false;
            }
        }
        return true;
    };

    double dev[8];
    double cpu[8];

    SU2Device::mul(dev, a, b);
    math::su2::mul_2x2(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    SU2Device::mul_adj(dev, a, b);
    math::su2::mul_adj_2x2(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    SU2Device::adj_mul(dev, a, b);
    math::su2::adj_mul_2x2(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    // traceless_antiherm on a hermitian-ish argument (U·V product shape).
    double prod[8];
    math::su2::mul_2x2(prod, a, b);
    SU2Device::traceless_antiherm(dev, prod);
    math::su2::traceless_antiherm_2x2(cpu, prod);
    if (!close(dev, cpu)) {
        return false;
    }
    // Group exponential V = exp(dt·P) for an anti-hermitian P (= TA of prod).
    double p_alg[8];
    math::su2::traceless_antiherm_2x2(p_alg, prod);
    SU2Device::expi(0.37, p_alg, dev);
    math::su2::exp_su2(cpu, p_alg, 0.37);
    return close(dev, cpu);
}

bool su2_cpu_matches_device() {
    MatrixLinkLattice<SU2, double> const host = make_links();

    Wil cpu{};
    cpu.beta           = kBeta;
    double const s_cpu = cpu.s_full(host);
    MatrixLinkLattice<SU2, double> f_cpu{host.indexing()};
    cpu.compute_force(host, f_cpu);

    DField dfield{kShape};
    DField dforce{dfield.topology()};
    dfield.copy_from_host(host.data());

    DAct const act{cpu, dfield.topology()};
    double const s_dev = act.s_full(dfield);
    act.compute_force(dfield, dforce);

    std::vector<double> f_dev(dforce.size());
    dforce.copy_to_host(f_dev.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > 1e-9 * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < f_dev.size(); ++i) {
        double const a = f_cpu.data()[i];
        if (std::abs(a - f_dev[i]) > 1e-9 * (1.0 + std::abs(a))) {
            return false;
        }
    }
    return true;
}

bool su2_energy_conserved_ok() {
    MatrixLinkLattice<SU2, double> const init = make_links();
    MatrixLinkLattice<SU2, double> p_host{init.indexing()};
    FastRng rng{77};
    std::size_t const d  = p_host.ndims();
    std::size_t const ns = p_host.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su2::sample_algebra_slab(p_host.mu_block_data(mu), rng, ns);
    }

    DAct const act{[] {
                       Wil a{};
                       a.beta = kBeta;
                       return a;
                   }(),
                   make_device_topology(kShape)};

    // |ΔH| of one trajectory from the SAME start, at step counts n and 2n.
    // H = ½·Σp² + S; for a 2nd-order integrator |ΔH| ∝ dt², so halving the step
    // (doubling n_md) should cut |ΔH| by ≈4×.
    auto delta_h = [&](int n_md) {
        DField field{kShape};
        DField mom{field.topology()};
        DField force{field.topology()};
        field.copy_from_host(init.data());
        mom.copy_from_host(p_host.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

        auto const n    = static_cast<long>(mom.size());
        double const h0 = (0.5 * reduce_sumsq_f64(mom.data(), n)) + act.s_full(field);
        alg::integ::Leapfrog::run(act, field, mom, force, 0.8, n_md);
        double const h1 = (0.5 * reduce_sumsq_f64(mom.data(), n)) + act.s_full(field);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        return std::abs(h1 - h0);
    };

    double const dh_coarse = delta_h(16);
    double const dh_fine   = delta_h(32);

    if (!std::isfinite(dh_coarse) || !std::isfinite(dh_fine)) {
        return false;
    }
    // Refinement must reduce |ΔH| toward the 2nd-order ratio of 4 (allow a wide
    // band — the leading-error regime is approximate at finite step).
    double const ratio = dh_coarse / dh_fine;
    return ratio > 2.5 && ratio < 6.0;
}

bool su2_hmc_reversibility_ok() {
    MatrixLinkLattice<SU2, double> const init = make_links();

    DField field{kShape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{[] {
                       Wil a{};
                       a.beta = kBeta;
                       return a;
                   }(),
                   field.topology()};

    field.copy_from_host(init.data());

    // Sample a valid algebra momentum on the host (per direction), stage it in.
    MatrixLinkLattice<SU2, double> p_host{init.indexing()};
    FastRng rng{99};
    std::size_t const d  = p_host.ndims();
    std::size_t const ns = p_host.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su2::sample_algebra_slab(p_host.mu_block_data(mu), rng, ns);
    }
    mom.copy_from_host(p_host.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    std::vector<double> f0(field.size());
    field.copy_to_host(f0.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    using Integ          = alg::integ::Leapfrog;
    constexpr double tau = 0.6;
    constexpr int n_md   = 12;

    Integ::run(act, field, mom, force, tau, n_md);
    axpy_f64(-2.0, mom.data(), mom.data(), static_cast<long>(field.size()));  // p -> -p
    Integ::run(act, field, mom, force, tau, n_md);

    std::vector<double> f1(field.size());
    field.copy_to_host(f1.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < f1.size(); ++i) {
        if (std::abs(f1[i] - f0[i]) > 1e-9) {
            return false;
        }
    }
    return true;
}

bool su2_momentum_moments_ok() {
    std::vector<std::size_t> const shape{8, 8, 8, 8};
    DField mom{shape};
    auto const n      = static_cast<long>(mom.size());
    long const nlinks = static_cast<long>(mom.topology().ndim) * mom.topology().nsites;

    Wil w{};
    w.beta = kBeta;
    DAct const act{w, mom.topology()};

    DeviceBuffer<std::uint64_t> traj{1};
    std::uint64_t const zero = 0;
    traj.copy_from_host(&zero, nullptr);

    act.sample_momenta(mom.data(), n, /*seed=*/0xC0FFEEULL, traj.data(), nullptr);

    std::vector<double> h(static_cast<std::size_t>(n));
    mom.copy_to_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    double sum   = 0.0;
    double sumsq = 0.0;
    for (double v : h) {
        if (!std::isfinite(v)) {
            return false;
        }
        sum += v;
        sumsq += v * v;
    }
    // Mean of all packed reals ≈ 0 (the structural zeros and the ±-paired
    // generators cancel in expectation).
    double const mean = sum / static_cast<double>(n);
    if (std::abs(mean) > 0.02) {
        return false;
    }
    // Per-link second moment: Σ(packed reals)² = 2·Σ_a h_a², h_a ~ N(0,½), so
    // E = 2·n_gen·½ = n_gen = 3. (The device MH reads ½·Σreals² as the kinetic
    // energy, i.e. ½·n_gen per link — the correct N(0,½)-per-generator measure.)
    double const per_link = sumsq / static_cast<double>(nlinks);
    double const expected = static_cast<double>(SU2Device::n_gen);
    return std::abs(per_link - expected) < 0.1 * expected;
}

bool su2_hmc_runs() {
    DField field{kShape};
    field.copy_from_host(make_links().data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    Wil w{};
    w.beta = kBeta;
    DAct dact{w, field.topology()};

    Hmc<DAct, alg::integ::Leapfrog, DField> hmc{std::move(dact), field, 0.4, 10};
    hmc.run(8);  // host-free: 8 SU(2) trajectories over the matrix link field
    double const acc = hmc.acceptance();
    hmc.sync();

    std::vector<double> out(field.size());
    field.copy_to_host(out.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (acc <= 0.0 || acc > 1.0) {
        return false;
    }
    for (double v : out) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda

// SU(2) Wilson gauge on the device through the SAME unified DeviceAction.
TEST_CASE("cuda SU2Device matrix ops match math::su2", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_device_ops_match_cpu());
}

TEST_CASE("cuda DeviceAction<Wilson<SU2>> matches CPU action::Wilson<SU2>", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_cpu_matches_device());
}

TEST_CASE("cuda SU2 Leapfrog MD conserves energy (2nd order)", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_energy_conserved_ok());
}

TEST_CASE("cuda SU2 HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_hmc_reversibility_ok());
}

TEST_CASE("cuda SU2 Gell-Mann momenta have the right moments", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_momentum_moments_ok());
}

TEST_CASE("cuda SU2 host-free HMC runs on the matrix link field", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_hmc_runs());
}
