#include <reticolo/math/gauge_group/su3.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/gauge/wilson.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/gauge/su3_device.cuh>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/math/su3_ops.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// SU(3) Wilson gauge HMC on the device. Same generic SU(N) kernels (gauge_sun.cuh)
// as SU(2) — only GD = SU3Device differs (nc=18, n_gen=8, 3×3 Morningstar-Peardon
// group exp). The matrix link field is MatrixLayout<SU3>, identical [ndim][nc][nsites]
// order to the host MatrixLinkLattice<SU3>. Gates: device ops vs math::su3,
// action/force vs CPU Wilson<SU3>, MD energy conservation + reversibility, Gell-Mann
// momentum moments, host-free HMC. (Excluded from the no-integrator-kernels lint
// gate: names alg::integ::Leapfrog to instantiate the generic integrator over the
// matrix field.)

namespace reticolo::cuda {

namespace {

using SU3    = gauge_group::SU3;
using Wil    = action::Wilson<SU3, double>;
using DField = DeviceField<double, MatrixLayout<SU3>>;
using DAct   = DeviceAction<Wil, DField>;

std::vector<std::size_t> const kShape{4, 4, 4, 4};
constexpr double kBeta = 6.0;

// Identity per link, then drift each direction by a random algebra element —
// a non-trivial but valid SU(3) on every link (the canonical host hot start).
void hot_start(MatrixLinkLattice<SU3, double>& u, FastRng& rng) {
    std::size_t const d     = u.ndims();
    std::size_t const ns    = u.nsites();
    std::size_t const total = d * SU3::n_real_components * ns;
    double* const data      = u.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = u.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;  // Re U_{00}
            blk[(8 * ns) + s]  = 1.0;  // Re U_{11}
            blk[(16 * ns) + s] = 1.0;  // Re U_{22}
        }
    }
    std::vector<double> scratch(SU3::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su3::sample_algebra_slab(scratch.data(), rng, ns);
        math::su3::expi_lmul_slab(u.mu_block_data(mu), scratch.data(), 0.5, ns);
    }
}

MatrixLinkLattice<SU3, double> make_links() {
    MatrixLinkLattice<SU3, double> u{Indexing::SizeVec(kShape.begin(), kShape.end())};
    FastRng rng{2718};
    hot_start(u, rng);
    return u;
}

}  // namespace

bool su3_device_ops_match_cpu() {
    FastRng rng{12345};
    double a[18];
    double b[18];
    for (double& x : a) {
        x = rng.normal();
    }
    for (double& x : b) {
        x = rng.normal();
    }

    auto close = [](double const* p, double const* q) {
        for (int k = 0; k < 18; ++k) {
            if (std::abs(p[k] - q[k]) > 1e-11) {
                return false;
            }
        }
        return true;
    };

    double dev[18];
    double cpu[18];

    SU3Device::mul(dev, a, b);
    math::su3::mul_3x3(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    SU3Device::mul_adj(dev, a, b);
    math::su3::mul_adj_3x3(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    SU3Device::adj_mul(dev, a, b);
    math::su3::adj_mul_3x3(cpu, a, b);
    if (!close(dev, cpu)) {
        return false;
    }
    double prod[18];
    math::su3::mul_3x3(prod, a, b);
    SU3Device::traceless_antiherm(dev, prod);
    math::su3::traceless_antiherm_3x3(cpu, prod);
    if (!close(dev, cpu)) {
        return false;
    }
    // Group exponential of a traceless anti-hermitian P (= TA of the product).
    double p_alg[18];
    math::su3::traceless_antiherm_3x3(p_alg, prod);
    SU3Device::expi(0.37, p_alg, dev);
    math::su3::exp_su3(cpu, p_alg, 0.37);
    return close(dev, cpu);
}

bool su3_cpu_matches_device() {
    MatrixLinkLattice<SU3, double> const host = make_links();

    Wil cpu{};
    cpu.beta           = kBeta;
    double const s_cpu = cpu.s_full(host);
    MatrixLinkLattice<SU3, double> f_cpu{host.indexing()};
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

// Fused s_full_and_force vs two-pass compute_force + s_full (see the SU(2) twin):
// force bit-identical (same staple sum), action to roundoff (staples summed then
// one ReTr). The LLR WindowedAction uses the fused path.
bool su3_fused_matches_twopass() {
    MatrixLinkLattice<SU3, double> const host = make_links();
    Wil a{};
    a.beta = kBeta;

    DField dfield{kShape};
    dfield.copy_from_host(host.data());
    DAct const act{a, dfield.topology()};

    DField f_two{dfield.topology()};
    act.compute_force(dfield, f_two);
    double const s_ref = act.s_full(dfield);

    DeviceBuffer<double> partials{static_cast<std::size_t>(k_reduce_max_grid)};
    DeviceBuffer<double> s_dev{1};
    DField f_fused{dfield.topology()};
    act.s_full_and_force(s_dev.data(), dfield, f_fused, partials.data(), nullptr);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    double s_fused = 0.0;
    s_dev.copy_to_host(&s_fused);
    std::vector<double> h_two(f_two.size());
    std::vector<double> h_fused(f_fused.size());
    f_two.copy_to_host(h_two.data());
    f_fused.copy_to_host(h_fused.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_ref - s_fused) > 1e-9 * (1.0 + std::abs(s_ref))) {
        return false;
    }
    for (std::size_t i = 0; i < h_two.size(); ++i) {
        if (std::abs(h_two[i] - h_fused[i]) > 1e-10 * (1.0 + std::abs(h_two[i]))) {
            return false;
        }
    }
    return true;
}

bool su3_energy_conserved_ok() {
    MatrixLinkLattice<SU3, double> const init = make_links();
    MatrixLinkLattice<SU3, double> p_host{init.indexing()};
    FastRng rng{77};
    std::size_t const d  = p_host.ndims();
    std::size_t const ns = p_host.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su3::sample_algebra_slab(p_host.mu_block_data(mu), rng, ns);
    }

    DAct const act{[] {
                       Wil a{};
                       a.beta = kBeta;
                       return a;
                   }(),
                   make_device_topology(kShape)};

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
    double const ratio = dh_coarse / dh_fine;
    return ratio > 2.5 && ratio < 6.0;
}

bool su3_hmc_reversibility_ok() {
    MatrixLinkLattice<SU3, double> const init = make_links();

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

    MatrixLinkLattice<SU3, double> p_host{init.indexing()};
    FastRng rng{99};
    std::size_t const d  = p_host.ndims();
    std::size_t const ns = p_host.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        math::su3::sample_algebra_slab(p_host.mu_block_data(mu), rng, ns);
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

bool su3_momentum_moments_ok() {
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
    double const mean = sum / static_cast<double>(n);
    if (std::abs(mean) > 0.02) {
        return false;
    }
    // Σ(packed reals)² = 2·Σ_a h_a², h_a ~ N(0,½), so per link E = 2·n_gen·½ =
    // n_gen = 8 (the device MH reads ½·Σreals² as the kinetic energy).
    double const per_link = sumsq / static_cast<double>(nlinks);
    double const expected = static_cast<double>(SU3Device::n_gen);
    return std::abs(per_link - expected) < 0.1 * expected;
}

bool su3_hmc_runs() {
    DField field{kShape};
    field.copy_from_host(make_links().data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    Wil w{};
    w.beta = kBeta;
    DAct dact{w, field.topology()};

    Hmc<DAct, alg::integ::Leapfrog, DField> hmc{std::move(dact), field, 0.4, 10};
    hmc.run(8);  // host-free: 8 SU(3) trajectories over the matrix link field
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

// SU(3) Wilson gauge — the same generic SU(N) kernels as SU(2) with GD = SU3Device.
TEST_CASE("cuda SU3Device matrix ops match math::su3", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_device_ops_match_cpu());
}

TEST_CASE("cuda DeviceAction<Wilson<SU3>> matches CPU action::Wilson<SU3>", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_cpu_matches_device());
}

TEST_CASE("cuda SU3 fused s_full_and_force matches two-pass", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_fused_matches_twopass());
}

TEST_CASE("cuda SU3 Leapfrog MD conserves energy (2nd order)", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_energy_conserved_ok());
}

TEST_CASE("cuda SU3 HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_hmc_reversibility_ok());
}

TEST_CASE("cuda SU3 Gell-Mann momenta have the right moments", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_momentum_moments_ok());
}

TEST_CASE("cuda SU3 host-free HMC runs on the matrix link field", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_hmc_runs());
}
