#pragma once

#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/llr/windowed_action.hpp>

#include <utility>

namespace reticolo::llr {

struct ReplicaStats {
    long n_traj     = 0;
    long n_accepted = 0;
};

// =============================================================================
//  Single LLR replica: owns its phi, its RNG, its WindowedAction wrapper (and
//  through it, its own copy of the base action), and its HMC kernel. The base
//  is copied at construction so any mutable per-action state stays
//  per-replica — required for OpenMP parallelism over replicas.
//
//  Non-moveable / non-copyable: the HMC inside holds references into the
//  replica's own members. Use `std::vector<std::unique_ptr<Replica<...>>>`
//  in the driver code.
// =============================================================================

template <class Base,
          class Rng,
          class Integrator = alg::integ::Leapfrog,
          class T          = Base::value_type>
class Replica {
public:
    using value_type = T;
    using scalar_t   = scalar_of_t<T>;

    Replica(Lattice<T>::SizeVec shape,
            Base const& base,
            Rng rng_init,
            scalar_t e_n_init,
            scalar_t delta_init,
            scalar_t a_init,
            alg::HmcSpec const& spec)
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

    // Running mean of `Q(phi) - E_n` over `n` trajectories, measured at the
    // end of each trajectory (accepted or not). Q = base.s_full when the
    // base is a real action, Q = base.s_imag in the complex-LLR case.
    [[nodiscard]] scalar_t sample(int n) {
        scalar_t sum = scalar_t{0};
        for (int i = 0; i < n; ++i) {
            auto const step = hmc_.trajectory();
            ++stats_.n_traj;
            if (step.accepted) {
                ++stats_.n_accepted;
            }
            sum += windowed_.constraint_value(phi_) - windowed_.E_n;
        }
        return sum / static_cast<scalar_t>(n);
    }

    [[nodiscard]] scalar_t energy() const noexcept { return windowed_.base.s_full(phi_); }

    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    [[nodiscard]] scalar_t a() const noexcept { return windowed_.a; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    [[nodiscard]] scalar_t E_n() const noexcept { return windowed_.E_n; }
    [[nodiscard]] scalar_t delta() const noexcept { return windowed_.delta; }

    void set_a(scalar_t v) noexcept { windowed_.a = v; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    void set_E_n(scalar_t v) noexcept { windowed_.E_n = v; }
    void set_delta(scalar_t v) noexcept { windowed_.delta = v; }

    [[nodiscard]] Lattice<T>& phi() noexcept { return phi_; }
    [[nodiscard]] Lattice<T> const& phi() const noexcept { return phi_; }

    [[nodiscard]] ReplicaStats const& stats() const noexcept { return stats_; }

private:
    Lattice<T> phi_;
    Rng rng_;
    WindowedAction<Base, T> windowed_;
    alg::Hmc<WindowedAction<Base, T>, Rng, Integrator> hmc_;
    ReplicaStats stats_{};
};

}  // namespace reticolo::llr
