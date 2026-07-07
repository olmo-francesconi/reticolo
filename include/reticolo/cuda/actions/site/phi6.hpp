#pragma once

#include <reticolo/action/site/formula/phi6_formula.hpp>
#include <reticolo/action/site/phi6.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/macros.hpp>

#include <cstdint>

// Device per-site functors for Phi6 + the host-action → device-functor trait.
// Same scalar device protocol as Phi4 (init / accumulate(mu, nbr) / finalize);
// the per-site math is the shared HD formula in action::formula (phi6_formula.hpp),
// one source of truth with the CPU action::Phi6.

namespace reticolo::cuda {

template <class T>
class Phi6ForceFunctor {
public:
    using element = T;
    RETICOLO_HD Phi6ForceFunctor(T kappa, T lambda, T g6)
        : kappa_{kappa}, lambda_{lambda}, g6_{g6} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        nbrs_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD T finalize() const {
        return action::formula::phi6_force_site<T>(phi_, nbrs_, kappa_, lambda_, g6_);
    }

private:
    T kappa_;
    T lambda_;
    T g6_;
    T phi_  = T{0};
    T nbrs_ = T{0};
};

template <class T>
class Phi6EnergyFunctor {
public:
    using element = T;
    RETICOLO_HD Phi6EnergyFunctor(T kappa, T lambda, T g6)
        : kappa_{kappa}, lambda_{lambda}, g6_{g6} {}

    RETICOLO_HD void init(T self) {
        phi_ = self;
        fwd_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { fwd_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return static_cast<double>(
            action::formula::phi6_action_site<T>(phi_, fwd_, kappa_, lambda_, g6_));
    }

private:
    T kappa_;
    T lambda_;
    T g6_;
    T phi_ = T{0};
    T fwd_ = T{0};
};

// Fused force + energy in one gather (see Phi4ForceEnergyFunctor): full 2d sum for
// the force, separate forward sum for the energy → S_base bit-identical to the
// reduce_fwd path. Used by the LLR WindowedAction.
template <class T>
class Phi6ForceEnergyFunctor {
public:
    using element = T;
    RETICOLO_HD Phi6ForceEnergyFunctor(T kappa, T lambda, T g6)
        : kappa_{kappa}, lambda_{lambda}, g6_{g6} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        full_ = T{0};
        fwd_  = T{0};
    }
    RETICOLO_HD void fwd(T nbr) {
        full_ += nbr;
        fwd_ += nbr;
    }
    RETICOLO_HD void bwd(T nbr) { full_ += nbr; }
    [[nodiscard]] RETICOLO_HD T force() const {
        return action::formula::phi6_force_site<T>(phi_, full_, kappa_, lambda_, g6_);
    }
    [[nodiscard]] RETICOLO_HD double energy() const {
        return static_cast<double>(
            action::formula::phi6_action_site<T>(phi_, fwd_, kappa_, lambda_, g6_));
    }

private:
    T kappa_;
    T lambda_;
    T g6_;
    T phi_  = T{0};
    T full_ = T{0};
    T fwd_  = T{0};
};

template <class T>
struct device_functors<action::Phi6<T>> {
    static void compute_force(action::Phi6<T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        impl::site_force(Phi6ForceFunctor<T>{a.kappa, a.lambda, a.g6}, field, force, topo, s);
    }
    [[nodiscard]] static double s_full(action::Phi6<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return impl::site_s_full(
            Phi6EnergyFunctor<T>{a.kappa, a.lambda, a.g6}, field, scratch, topo, s);
    }
    static void s_full_into(double* out,
                            action::Phi6<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        impl::site_s_full_into(
            out, Phi6EnergyFunctor<T>{a.kappa, a.lambda, a.g6}, field, scratch, partials, topo, s);
    }
    static void sample_momenta(T* mom,
                               long n,
                               DeviceTopology const& topo,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        impl::site_sample_momenta(mom, n, topo, seed, traj, s);
    }
    static void s_full_and_force(double* out,
                                 action::Phi6<T> const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
        impl::site_s_full_and_force(out,
                                    Phi6ForceEnergyFunctor<T>{a.kappa, a.lambda, a.g6},
                                    field,
                                    force,
                                    scratch,
                                    partials,
                                    topo,
                                    s);
    }
};

}  // namespace reticolo::cuda
