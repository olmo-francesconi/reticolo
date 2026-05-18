#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/gauge/algorithm/hmc.hpp>
#include <reticolo/gauge/algorithm/integrators.hpp>
#include <reticolo/gauge/llr/windowed_action.hpp>
#include <reticolo/llr/replica.hpp>  // for ReplicaStats reuse

#include <utility>

namespace reticolo::gauge::llr {

using reticolo::llr::ReplicaStats;

// =============================================================================
//  Single LLR replica for a gauge action: owns its phi (LinkLattice), its RNG,
//  its tilted action, and its HMC kernel. Twin of `reticolo::llr::Replica`.
// =============================================================================

template <class Base,
          class Rng,
          class Integrator = gauge::alg::integ::Leapfrog,
          class T          = typename Base::value_type>
class Replica {
public:
    using value_type = T;
    using field_type = typename Base::field_type;
    using SizeVec    = typename field_type::SizeVec;

    Replica(SizeVec shape,
            Base const& base,
            Rng rng_init,
            T e_n_init,
            T delta_init,
            T a_init,
            gauge::alg::HmcSpec const& spec)
        : phi_{std::move(shape)}, rng_{std::move(rng_init)},
          windowed_{.base = base, .a = a_init, .E_n = e_n_init, .delta = delta_init},
          hmc_{windowed_, phi_, rng_, spec} {}

    Replica(Replica const&)            = delete;
    Replica& operator=(Replica const&) = delete;
    Replica(Replica&&)                 = delete;
    Replica& operator=(Replica&&)      = delete;
    ~Replica()                         = default;

    void thermalize(int n) {
        for (int i = 0; i < n; ++i) {
            auto const step = hmc_.trajectory();
            ++stats_.n_traj;
            if (step.accepted) {
                ++stats_.n_accepted;
            }
        }
    }

    [[nodiscard]] T sample(int n) {
        T sum = T{0};
        for (int i = 0; i < n; ++i) {
            auto const step = hmc_.trajectory();
            ++stats_.n_traj;
            if (step.accepted) {
                ++stats_.n_accepted;
            }
            sum += windowed_.base.s_full(phi_) - windowed_.E_n;
        }
        return sum / static_cast<T>(n);
    }

    [[nodiscard]] T energy() const noexcept { return windowed_.base.s_full(phi_); }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    [[nodiscard]] T a() const noexcept { return windowed_.a; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    [[nodiscard]] T E_n() const noexcept { return windowed_.E_n; }
    [[nodiscard]] T delta() const noexcept { return windowed_.delta; }

    void set_a(T v) noexcept { windowed_.a = v; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    void set_E_n(T v) noexcept { windowed_.E_n = v; }

    [[nodiscard]] field_type& phi() noexcept { return phi_; }
    [[nodiscard]] field_type const& phi() const noexcept { return phi_; }

    [[nodiscard]] ReplicaStats const& stats() const noexcept { return stats_; }

private:
    field_type phi_;
    Rng rng_;
    WindowedAction<Base, T> windowed_;
    gauge::alg::Hmc<WindowedAction<Base, T>, Rng, Integrator> hmc_;
    ReplicaStats stats_{};
};

}  // namespace reticolo::gauge::llr
