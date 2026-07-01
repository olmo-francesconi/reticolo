#include <reticolo/action/detail/gauge/gauge_group/u1.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/wilson.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Wilson<U(1)> on the device via the abelian specialization. The link is a
// single angle (LinkLayout); device_functors<Wilson<U1>> reuses the gauge_u1.cuh
// kernels (the CompactU1 device path) since the two actions are bit-identical
// (n_color=1). The host MatrixLinkLattice<U1> stores ndim·nsites angles in the
// same [ndim][nsites] order as LinkLayout, so a flat copy round-trips — no
// matrix conversion. (Excluded from the no-integrator-kernels lint gate: it names
// alg::integ::Leapfrog to instantiate the generic integrator over the link field.)

namespace reticolo::cuda {

namespace {

using U1G    = gauge_group::U1;
using Wil    = action::Wilson<U1G, double>;
using DField = DeviceField<double, LinkLayout>;
using DAct   = DeviceAction<Wil, DField>;

std::vector<std::size_t> const kShape{4, 4, 4, 4};
constexpr double kBeta = 1.1;

// Host U(1) gauge config: a real angle per link, in the [ndim][nsites] order
// shared by MatrixLinkLattice<U1> and the device LinkLayout.
MatrixLinkLattice<U1G, double> make_links() {
    MatrixLinkLattice<U1G, double> u{Indexing::SizeVec(kShape.begin(), kShape.end())};
    double* const d        = u.data();
    std::size_t const ncmp = u.ncomponents();  // ndim·nsites (1 angle per link)
    for (std::size_t i = 0; i < ncmp; ++i) {
        d[i] = 0.6 * std::sin((0.4 * static_cast<double>(i)) + 0.7);
    }
    return u;
}

}  // namespace

bool wilson_u1_cpu_matches_device() {
    MatrixLinkLattice<U1G, double> const host = make_links();

    Wil cpu{};
    cpu.beta           = kBeta;
    double const s_cpu = cpu.s_full(host);
    MatrixLinkLattice<U1G, double> f_cpu{host.indexing()};
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

    if (std::abs(s_cpu - s_dev) > 1e-10 * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < f_dev.size(); ++i) {
        double const a = f_cpu.data()[i];
        if (std::abs(a - f_dev[i]) > 1e-7 * (1.0 + std::abs(a))) {
            return false;
        }
    }
    return true;
}

bool wilson_u1_hmc_reversibility_ok() {
    MatrixLinkLattice<U1G, double> const init = make_links();

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
    std::vector<double> p(field.size());
    FastRng rng{11};
    rng.normal_fill(p.data(), p.size());
    mom.copy_from_host(p.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    std::vector<double> f0(field.size());
    field.copy_to_host(f0.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    using Integ          = alg::integ::Leapfrog;
    constexpr double tau = 1.0;
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

bool wilson_u1_hmc_runs() {
    DField field{kShape};
    field.copy_from_host(make_links().data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    Wil act{};
    act.beta = kBeta;
    DAct dact{act, field.topology()};

    Hmc<DAct, alg::integ::Leapfrog, DField> hmc{std::move(dact), field, 1.0, 10};
    hmc.run(8);
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

// Wilson<U(1)> on the device via the SPECIALIZED abelian path (reuses the
// CompactU1 angle kernels on a 1-angle LinkLayout field).
TEST_CASE("cuda DeviceAction<Wilson<U1>> matches CPU action::Wilson<U1>", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_cpu_matches_device());
}

TEST_CASE("cuda Wilson<U1> HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_hmc_reversibility_ok());
}

TEST_CASE("cuda Wilson<U1> host-free HMC runs on the link field", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_hmc_runs());
}
