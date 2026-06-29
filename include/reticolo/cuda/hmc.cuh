#pragma once

// Generic device HMC — nvcc-only (.cuh; transitively launches kernels).
//
// Mirrors alg::Hmc::step() once. There is NO action or integrator switch: the
// integrator is the type parameter Integ (an unchanged alg::integ::* tag), and
// the action is any type exposing s_full(Field) / compute_force(Field, Field)
// — here cuda::DeviceAction. The MD loop reuses Leapfrog/Omelyan2/Omelyan4
// verbatim through the device drift_field/kick_add atoms (cuda/integ_ops.hpp,
// resolved by ADL). Momenta are filled on the host and copied to the device
// (Phase 2b); the device Philox sampler replaces that in Phase 2c. The MH
// accept is a one-scalar host decision over ΔH; graph capture wraps H0·MD·H1
// in Phase 2d.

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace reticolo::cuda {

struct HmcResult {
    double dH     = 0.0;
    bool accepted = false;
};

template <class A, class R, class Integ = alg::integ::Leapfrog, class Field = DeviceField<double>>
class Hmc {
public:
    Hmc(A action, Field& field, R& rng, double tau, int n_md)
        : action_{std::move(action)},
          field_{field},
          rng_{rng},
          mom_{field.topology()},
          force_{field.topology()},
          old_{field.topology()},
          tau_{tau},
          n_md_{n_md},
          hbuf_(field.size()) {}

    HmcResult step() {
        sample_momenta_();
        copy_device_(old_, field_);
        double const h0 = hamiltonian_();
        Integ::run(action_, field_, mom_, force_, tau_, n_md_);
        double const h1     = hamiltonian_();
        double const dH     = h1 - h0;
        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            copy_device_(field_, old_);
        }
        return {.dH = dH, .accepted = accepted};
    }

    // Integrator-only run + accessors — the seam the reversibility / order
    // gates drive directly (no MH, no resample).
    void integrate(double tau, int n_md) { Integ::run(action_, field_, mom_, force_, tau, n_md); }
    [[nodiscard]] Field& momentum() noexcept { return mom_; }
    [[nodiscard]] double hamiltonian() { return hamiltonian_(); }

private:
    void sample_momenta_() {
        rng_.normal_fill(hbuf_.data(), hbuf_.size());
        mom_.copy_from_host(hbuf_.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    [[nodiscard]] double hamiltonian_() {
        double const kinetic   = 0.5 * reduce_sumsq_f64(mom_.data(), static_cast<long>(mom_.size()));
        double const potential = action_.s_full(field_);
        return kinetic + potential;
    }

    static void copy_device_(Field& dst, Field const& src) {
        RETICOLO_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(),
                                            src.size() * sizeof(typename Field::value_type),
                                            cudaMemcpyDeviceToDevice, nullptr));
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    A action_;
    Field& field_;
    R& rng_;
    Field mom_;
    Field force_;
    Field old_;
    double tau_;
    int n_md_;
    std::vector<double> hbuf_;
};

}  // namespace reticolo::cuda
