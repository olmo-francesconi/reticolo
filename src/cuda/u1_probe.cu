#include <reticolo/action/compact_u1.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/compact_u1.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.hpp>
#include <reticolo/cuda/u1_probe.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

// Phase 4: compact U(1) gauge HMC on the device. The link field is direction-
// major (LinkLayout), the force a per-link gather, the action a per-site
// plaquette reduction. This TU runs the CPU action and the device path and
// compares, validates the gather force against a finite difference, and checks
// MD reversibility. (Excluded from the no-integrator-kernels lint gate: it names
// alg::integ::Leapfrog to instantiate the generic integrator over the link
// field, the same as hmc_probe.cu / f32_probe.cu.)

namespace reticolo::cuda {

namespace {

using U1     = action::CompactU1<double>;
using DField = DeviceField<double, LinkLayout>;
using DAct   = DeviceAction<U1, DField>;

std::vector<std::size_t> const kShape{4, 4, 4, 4};
constexpr double kBeta = 1.1;

LinkLattice<double> make_links() {
    LinkLattice<double> l{Indexing::SizeVec(kShape.begin(), kShape.end())};
    double* const d          = l.data();
    std::size_t const nlinks = l.nlinks();
    for (std::size_t i = 0; i < nlinks; ++i) {
        d[i] = 0.6 * std::sin((0.4 * static_cast<double>(i)) + 0.7);
    }
    return l;
}

}  // namespace

bool u1_cpu_matches_device() {
    LinkLattice<double> const host = make_links();

    U1 cpu{};
    cpu.beta           = kBeta;
    double const s_cpu = cpu.s_full(host);
    LinkLattice<double> f_cpu{host.indexing()};
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

bool u1_force_matches_fd() {
    LinkLattice<double> host = make_links();
    U1 cpu{};
    cpu.beta = kBeta;

    DField dfield{kShape};
    DField dforce{dfield.topology()};
    DAct const act{cpu, dfield.topology()};

    dfield.copy_from_host(host.data());
    act.compute_force(dfield, dforce);
    std::vector<double> force(dforce.size());
    dforce.copy_to_host(force.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    // Central FD of the device s_full w.r.t. a sample of links. F = -dS/dtheta.
    constexpr double eps     = 1e-4;
    std::size_t const nlinks = host.nlinks();
    std::size_t const stride = (nlinks / 48) + 1;  // ~48 samples, keep it cheap
    for (std::size_t i = 0; i < nlinks; i += stride) {
        double const saved = host.data()[i];

        host.data()[i] = saved + eps;
        dfield.copy_from_host(host.data());
        double const s_plus = act.s_full(dfield);

        host.data()[i] = saved - eps;
        dfield.copy_from_host(host.data());
        double const s_minus = act.s_full(dfield);

        host.data()[i] = saved;

        double const f_fd = -(s_plus - s_minus) / (2.0 * eps);
        if (std::abs(f_fd - force[i]) > 1e-5 * (1.0 + std::abs(force[i]))) {
            return false;
        }
    }
    return true;
}

bool u1_hmc_reversibility_ok() {
    LinkLattice<double> const init = make_links();

    DField field{kShape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{[] {
                       U1 a{};
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

bool u1_hmc_runs() {
    DField field{kShape};
    field.copy_from_host(make_links().data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    U1 act{};
    act.beta = kBeta;
    DAct dact{act, field.topology()};

    Hmc<DAct, alg::integ::Leapfrog, DField> hmc{std::move(dact), field, 1.0, 10};
    hmc.run(8);  // host-free: 8 gauge trajectories over the link field
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
