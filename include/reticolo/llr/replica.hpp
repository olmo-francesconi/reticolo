#pragma once

#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/llr/windowed_action.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace reticolo::llr {

struct ReplicaStats {
    long n_traj     = 0;
    long n_accepted = 0;
};

// Single LLR replica: owns its phi, its RNG, its WindowedAction wrapper (and
// through it, its own copy of the base action), and its HMC kernel. The base
// is copied at construction so any mutable per-action state stays
// per-replica — required for OpenMP parallelism over replicas.
//
// One template for both scalar and gauge LLR: `Field` defaults to
// `Lattice<T>` so existing scalar callers compile unchanged; gauge users
// pass `LinkLattice<T>` explicitly. Window/tilt math lives in
// WindowedAction; HMC handles the field-type dispatch via flat_size.
//
// Non-moveable / non-copyable: the HMC inside holds references into the
// replica's own members. Use `std::vector<std::unique_ptr<Replica<...>>>`
// in the driver code.

template <class Base,
          class Rng,
          class Integrator = alg::integ::Leapfrog,
          class T          = typename Base::value_type,
          class Field      = Lattice<T>>
class Replica {
public:
    using value_type           = T;
    using field_type           = Field;
    using scalar_t             = scalar_of_t<T>;
    using SizeVec              = typename Field::SizeVec;
    using windowed_action_type = WindowedAction<Base, T, Field>;

    static constexpr std::string_view log_tag = "repl";

    struct Spec {
        std::string id;
        SizeVec shape;
        scalar_t e_n{};
        scalar_t delta{};
        scalar_t a_init = scalar_t{0};
    };

    Replica(Base const& base, Rng rng_init, Spec spec, alg::HmcSpec const& hmc_spec)
        : id_{std::move(spec.id)}, phi_{std::move(spec.shape)}, rng_{std::move(rng_init)},
          windowed_{.base = base, .a = spec.a_init, .E_n = spec.e_n, .delta = spec.delta},
          hmc_{windowed_, phi_, rng_, hmc_spec, Integrator{}, log::Mode::silent} {
        // Self-announce with our run id bound as scope so the line carries
        // `r0NN` automatically — apps don't have to wrap construction in
        // `log::scope` themselves. Announce the nested HMC too so its tau /
        // n_md params land under the same scope.
        auto _ = log::scope(id_);
        log::algo(*this);
        log::algo(hmc_);
    }

    Replica(Replica const&)            = delete;
    Replica& operator=(Replica const&) = delete;
    Replica(Replica&&)                 = delete;
    Replica& operator=(Replica&&)      = delete;
    ~Replica()                         = default;

    [[nodiscard]] std::string_view id() const noexcept { return id_; }

    void describe(log::Entry& e) const {
        e.line("Replica<{}>", scalar_name<scalar_t>());
        e.param("E_n={:+.3f}", static_cast<double>(windowed_.E_n));
        e.param("a={:+.3f}", static_cast<double>(windowed_.a));
        e.param("δ={:.3f}", static_cast<double>(windowed_.delta));
    }

    void thermalize(int n, log::Mode log_mode = log::Mode::normal) {
        auto _             = log::scope(id_);
        int local_accepted = 0;
        for (int i = 0; i < n; ++i) {
            auto const step = hmc_.step(log::Mode::silent);
            ++stats_.n_traj;
            if (step.accepted) {
                ++stats_.n_accepted;
                ++local_accepted;
            }
        }
        if (log_mode == log::Mode::normal) {
            double const acc =
                n > 0 ? static_cast<double>(local_accepted) / static_cast<double>(n) : 0.0;
            log::info("repl", "thermalize n={}  acc={:.3f}", n, acc);
        }
    }

    // Running mean of `Q(phi) - E_n` over `n` trajectories, measured at the
    // end of each trajectory (accepted or not). Q = base.s_full when the
    // base is a real action, Q = base.s_imag in the complex-LLR case. Reads
    // the base action's post-trajectory raw-scalar cache instead of a fresh
    // sweep — HMC's s_full call at h1 already populated it, and a reject
    // would have rolled it back to the h0 value.
    [[nodiscard]] scalar_t sample(int n, log::Mode log_mode = log::Mode::normal) {
        auto _       = log::scope(id_);
        scalar_t sum = scalar_t{0};
        for (int i = 0; i < n; ++i) {
            auto const step = hmc_.step(log::Mode::silent);
            ++stats_.n_traj;
            if (step.accepted) {
                ++stats_.n_accepted;
            }
            if constexpr (Windowed::k_complex) {
                sum += windowed_.base.last_s_imag() - windowed_.E_n;
            } else {
                sum += windowed_.base.last_s_full() - windowed_.E_n;
            }
        }
        scalar_t const dE = sum / static_cast<scalar_t>(n);
        if (log_mode == log::Mode::normal) {
            auto const dE_d    = static_cast<double>(dE);
            auto const delta   = static_cast<double>(windowed_.delta);
            double const ratio = (delta != 0.0) ? (dE_d / delta) : 0.0;
            log::info("repl", "sample n={}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}", n, dE_d, ratio);
        }
        return dE;
    }

    // Exchange energy E = the LLR constraint observable of the current config:
    // S_I (imaginary part) in the complex mode-B path, S_base in real mode A.
    // This is the observable `a` couples to, so the swap acceptance must use it
    // (mode A: S_base == s_full; mode B: S_I == s_imag — mirrors `sample()`).
    // Reads the base action's post-trajectory cache (populated by HMC, rolled
    // back on reject) instead of a fresh O(V) sweep — valid once at least one
    // trajectory has run, which the drivers guarantee (exchange only happens
    // after `sample`).
    [[nodiscard]] scalar_t energy() const noexcept {
        if constexpr (Windowed::k_complex) {
            return static_cast<scalar_t>(windowed_.base.last_s_imag());
        } else {
            return static_cast<scalar_t>(windowed_.base.last_s_full());
        }
    }

    // After an accepted exchange swaps the two fields, each base action's
    // raw-scalar cache must follow the config it described.
    void swap_energy_cache(Replica& other) noexcept {
        auto const mine    = windowed_.base.last_s_full();
        auto const other_s = other.windowed_.base.last_s_full();
        windowed_.base.restore_last_s_full(other_s);
        other.windowed_.base.restore_last_s_full(mine);
        if constexpr (Windowed::k_complex) {
            auto const mine_i  = windowed_.base.last_s_imag();
            auto const other_i = other.windowed_.base.last_s_imag();
            windowed_.base.restore_last_s_imag(other_i);
            other.windowed_.base.restore_last_s_imag(mine_i);
        }
    }

    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    [[nodiscard]] scalar_t a() const noexcept { return windowed_.a; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    [[nodiscard]] scalar_t E_n() const noexcept { return windowed_.E_n; }
    [[nodiscard]] scalar_t delta() const noexcept { return windowed_.delta; }

    void set_a(scalar_t v) noexcept { windowed_.a = v; }
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n
    void set_E_n(scalar_t v) noexcept { windowed_.E_n = v; }
    void set_delta(scalar_t v) noexcept { windowed_.delta = v; }

    [[nodiscard]] Field& phi() noexcept { return phi_; }
    [[nodiscard]] Field const& phi() const noexcept { return phi_; }
    [[nodiscard]] Rng& rng() noexcept { return rng_; }
    [[nodiscard]] windowed_action_type const& windowed_action() const noexcept { return windowed_; }

    // Random Gaussian-shift seed of the field, sigma per real component.
    // Complex fields get independent N(0, sigma²) on Re and Im.
    void hot_start(scalar_t sigma) noexcept {
        T* const data       = phi_.data();
        std::size_t const n = phi_.nsites();
        if constexpr (std::is_same_v<T, std::complex<double>> ||
                      std::is_same_v<T, std::complex<float>>) {
            using R = typename T::value_type;
            for (std::size_t i = 0; i < n; ++i) {
                data[i] = T{static_cast<R>(static_cast<scalar_t>(rng_.normal()) * sigma),
                            static_cast<R>(static_cast<scalar_t>(rng_.normal()) * sigma)};
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                data[i] = static_cast<T>(static_cast<scalar_t>(rng_.normal()) * sigma);
            }
        }
    }

    // Drive this replica's own windowed HMC in batches until
    // |S_constraint − E_n| < threshold_sigmas·δ or `max_batches` × `batch_size`
    // trajectories have been taken. The windowed force pins trajectories toward
    // E_n; deep in the S tail a stiffer integrator (more MD steps) may be needed
    // for the leapfrog to stay stable — tune via the replica's HmcSpec.
    //
    // Returns the number of batches consumed. `== max_batches` means budget
    // exhausted without convergence.
    int warm_into_window(int max_batches,
                         int batch_size          = 10,
                         double threshold_sigmas = 1.0,
                         log::Mode log_mode      = log::Mode::normal) {
        auto _ = log::scope(id_);
        for (int b = 0; b < max_batches; ++b) {
            for (int i = 0; i < batch_size; ++i) {
                (void)hmc_.step(log::Mode::silent);
            }
            scalar_t const q   = windowed_.constraint_value(phi_);
            scalar_t const dev = std::abs(q - windowed_.E_n);
            scalar_t const ratio =
                (windowed_.delta != scalar_t{0}) ? (dev / windowed_.delta) : scalar_t{0};
            if (log_mode == log::Mode::normal) {
                log::info("repl",
                          "warm  batch {:>4}  |S-E_n|/δ={:+.3f}",
                          b + 1,
                          static_cast<double>(ratio));
            }
            if (static_cast<double>(ratio) < threshold_sigmas) {
                return b + 1;
            }
        }
        return max_batches;
    }

    [[nodiscard]] ReplicaStats const& stats() const noexcept { return stats_; }

private:
    using Windowed = windowed_action_type;

    std::string id_;
    Field phi_;
    Rng rng_;
    Windowed windowed_;
    alg::Hmc<Windowed, Rng, Integrator, Field, T> hmc_;
    ReplicaStats stats_{};
};

}  // namespace reticolo::llr
