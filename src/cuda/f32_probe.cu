#include <reticolo/action/phi4.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/f32_probe.hpp>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

// Phase 3d: the device HMC stack in single precision. f32 momenta/fields flow
// through axpy_f32 (MD) + the double-accumulating sum-of-squares reduction; this
// TU runs the same checks Phase 2 ran in f64, but to a bounded tolerance.

namespace reticolo::cuda {

namespace {

using Phi4f   = action::Phi4<float>;
using DFieldf = DeviceField<float>;
using DActf   = DeviceAction<Phi4f, DFieldf>;

std::vector<std::size_t> const kShape{6, 4, 5};

Phi4f make_action() {
    Phi4f a{};
    a.kappa  = 0.18F;
    a.lambda = 0.55F;
    return a;
}

Lattice<float> make_field() {
    Lattice<float> l{kShape};
    float* const d = l.data();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        d[i] = static_cast<float>(0.4 * std::sin(0.3 * static_cast<double>(i) + 1.0));
    }
    return l;
}

}  // namespace

bool phi4_f32_cpu_matches_device() {
    Lattice<float> const host = make_field();

    Phi4f const cpu    = make_action();
    double const s_cpu = cpu.s_full(host);
    Lattice<float> f_cpu{kShape};
    cpu.compute_force(host, f_cpu);

    DFieldf dfield{kShape};
    DFieldf dforce{dfield.topology()};
    dfield.copy_from_host(host);

    DActf const act{cpu, dfield.topology()};
    double const s_dev = act.s_full(dfield);
    act.compute_force(dfield, dforce);

    Lattice<float> f_dev{kShape};
    dforce.copy_to_host(f_dev);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > 1e-5 * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < host.nsites(); ++i) {
        float const a = f_cpu.data()[i];
        float const b = f_dev.data()[i];
        if (std::abs(a - b) > (1e-4F * (1.0F + std::abs(a)))) {
            return false;
        }
    }
    return true;
}

bool hmc_f32_reversibility_ok() {
    Lattice<float> const init = make_field();
    auto const n              = init.nsites();

    DFieldf field{kShape};
    DFieldf mom{field.topology()};
    DFieldf force{field.topology()};
    DActf const act{make_action(), field.topology()};

    field.copy_from_host(init);
    std::vector<double> pd(n);
    FastRng rng{7};
    rng.normal_fill(pd.data(), n);
    std::vector<float> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = static_cast<float>(pd[i]);
    }
    mom.copy_from_host(p.data());
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    Lattice<float> f0{kShape};
    field.copy_to_host(f0);
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    using Integ          = alg::integ::Leapfrog;
    constexpr double tau = 1.0;
    constexpr int n_md   = 12;

    Integ::run(act, field, mom, force, tau, n_md);
    axpy_f32(-2.0F, mom.data(), mom.data(), static_cast<long>(n));  // p -> -p
    Integ::run(act, field, mom, force, tau, n_md);

    Lattice<float> f1{kShape};
    field.copy_to_host(f1);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < n; ++i) {
        float const a = f0.data()[i];
        float const b = f1.data()[i];
        if (std::abs(a - b) > (1e-3F * (1.0F + std::abs(a)))) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda
