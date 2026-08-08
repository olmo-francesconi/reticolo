#pragma once

#include <reticolo/action/windowed_action.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>
#include <reticolo/orch/llr/ramp.hpp>
#include <reticolo/orch/llr/update_a.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/samplers.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace reticolo::orch::llr {

struct ReplicaStats {
    long n_traj = 0;
    // Σ of each update's `acceptance()`, not a count of accepted updates: an HMC
    // trajectory contributes 0 or 1, a Metropolis sweep contributes its site
    // acceptance fraction. Divided by n_traj it is the mean acceptance for either
    // sampler — which is exactly why `updater::Updater` keys on acceptance() and
    // not on an `accepted` flag one of the two algorithms cannot supply.
    double sum_acceptance = 0.0;
};

// Single LLR replica: owns its phi, its RNG, its WindowedAction wrapper (and
// through it, its own copy of the base action), and its sampler. The base
// is copied at construction so any mutable per-action state stays
// per-replica — required for OpenMP parallelism over replicas.
//
// One template for both scalar and gauge LLR: the field type comes from the
// ACTION (`Action::field_type`), so a gauge replica spells the same thing a
// scalar one does. Window/tilt math lives in WindowedAction; the sampler
// handles the field-type dispatch via flat_size.
//
// THE SAMPLER IS A TEMPLATE PARAMETER — an `updater::` sampler tag, in the slot
// the MD integrator used to hold, and with NO default: which updater drives a run
// is a decision worth reading at the call site. There is ONE Replica: nothing
// below reaches past `updater::Updater` (`step().acceptance()` plus the action's
// own caches is the entire surface), so swapping HMC for local Metropolis is a
// template argument, not a second class.
//
//   Replica<Action, Rng, updater::HmcSampler<>>                          Omelyan2
//   Replica<Action, Rng, updater::HmcSampler<updater::integ::Omelyan4>>
//   Replica<Action, Rng, updater::MetropolisSampler>
//
// It shares this structure with `orch::span::Chain<Action, Rng, Sampler>` — the
// two are the same kind of thing. The integrator rides on the HMC tag because it
// is meaningless for any other sampler. What genuinely differs between samplers
// is the construction spec, so the ctor takes `Sampler::spec_type` and
// `make_sampler_` reads the tag's optional `integrator` alias to decide whether
// the ctor takes one.
//
// Non-moveable / non-copyable: the sampler inside holds references into the
// replica's own members. Use `std::vector<std::unique_ptr<Replica<...>>>`
// in the driver code.

// The element and field types are NOT parameters: the action already names both
// (`value_type`, `field_type`), so a parameter for either could only ever be set
// to what the action says or to something wrong. A gauge LLR therefore looks
// exactly like a scalar one — `Replica<Wilson<SU3>, FastRng>`.
template <class Action, class Rng, class Sampler, class Constraint = void>
class Replica {
public:
    using value_type           = Action::value_type;
    using field_type           = Action::field_type;
    using scalar_t             = action::scalar_of_t<value_type>;
    using SizeVec              = field_type::SizeVec;
    using windowed_action_type = action::WindowedAction<Action, Constraint>;
    // The resolved constraint policy: SelfConstraint / ImagConstraint by default
    // (empty), or a caller-supplied ObservableConstraint<Obs> for a window on an
    // arbitrary observable. Passed to the ctor; `{}` (the default) is the empty
    // built-in, so existing real/complex LLR callers are unchanged.
    using constraint_type = windowed_action_type::constraint_type;
    // The tag names the updater only once the windowed action exists — which is
    // the whole reason the parameter is a tag and not the updater itself.
    using sampler_type      = Sampler::template type<windowed_action_type, Rng>;
    using sampler_spec_type = Sampler::spec_type;

    static_assert(updater::Updater<sampler_type>, "the LLR sampler must model updater::Updater");

    static constexpr std::string_view log_tag = "repl";

    struct Spec {
        std::string id;
        SizeVec shape;
        scalar_t e_n{};
        scalar_t delta{};
        scalar_t a_init = scalar_t{0};
    };

    Replica(Action const& base,
            Rng rng_init,
            Spec spec,
            sampler_spec_type const& sampler_spec,
            constraint_type constraint = {})
        : id_{std::move(spec.id)}, field_{std::move(spec.shape)},
          windowed_{.base       = base,
                    .a          = spec.a_init,
                    .E_n        = spec.e_n,
                    .delta      = spec.delta,
                    .constraint = std::move(constraint)},
          sampler_{make_sampler_(windowed_, field_, std::move(rng_init), sampler_spec)} {}

    // Announce the nested sampler (integrator / τ / n_md, or σ). The LLR driver
    // calls this once, on a representative replica, so the ensemble's shared
    // sampler config shows exactly once instead of N× — replica construction is
    // otherwise wrapped in `log::quiet()` by the app.
    void announce_sampler() const { log::algo(sampler_); }

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
        auto _           = log::scope(id_);
        double local_acc = 0.0;
        for (int i = 0; i < n; ++i) {
            double const acc = sampler_.step(log::Mode::silent).acceptance();
            ++stats_.n_traj;
            stats_.sum_acceptance += acc;
            local_acc += acc;
        }
        if (log_mode == log::Mode::normal) {
            log::info("repl",
                      "thermalize n={}  acc={:.3f}",
                      n,
                      n > 0 ? local_acc / static_cast<double>(n) : 0.0);
        }
    }

    // Running mean of `Q(phi) - E_n` over `n` trajectories, measured at the
    // end of each trajectory (accepted or not). Q is the window constraint
    // observable (`windowed_.last_constraint()`): the base action for real LLR,
    // its imaginary part for complex, or a custom observable. Reads the
    // post-trajectory cache HMC already populated (rolled back on reject),
    // not a fresh O(V) sweep.
    [[nodiscard]] scalar_t sample(int n, log::Mode log_mode = log::Mode::normal) {
        auto _       = log::scope(id_);
        scalar_t sum = scalar_t{0};
        for (int i = 0; i < n; ++i) {
            stats_.sum_acceptance += sampler_.step(log::Mode::silent).acceptance();
            ++stats_.n_traj;
            sum += windowed_.last_constraint() - windowed_.E_n;
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
        return static_cast<scalar_t>(windowed_.last_constraint());
    }

    // After an accepted exchange swaps the two fields, each caches must follow
    // the config it now describes: the base action cache always, and the
    // separate constraint-value cache when there is one (mode B / custom Q).
    void swap_energy_cache(Replica& other) noexcept {
        auto const mine    = windowed_.base.last_s_full();
        auto const other_s = other.windowed_.base.last_s_full();
        windowed_.base.restore_last_s_full(other_s);
        other.windowed_.base.restore_last_s_full(mine);
        if constexpr (Windowed::k_complex) {
            auto const mine_i  = windowed_.last_s_imag();
            auto const other_i = other.windowed_.last_s_imag();
            windowed_.restore_last_s_imag(other_i);
            other.windowed_.restore_last_s_imag(mine_i);
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

    [[nodiscard]] field_type& field() noexcept { return field_; }
    [[nodiscard]] field_type const& field() const noexcept { return field_; }
    // The replica's randomness is owned by its sampler (one site stream per slab —
    // one stream inside the replica team — plus a driver for serial draws).
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return sampler_.rng(); }
    [[nodiscard]] windowed_action_type const& windowed_action() const noexcept { return windowed_; }

    // Per-replica checkpoint payload beyond field + rng: the adapted tilt `a`
    // (orch::HasCheckpointExtra). Templated on the writer/reader so replica.hpp
    // stays free of the io headers.
    template <class Writer>
    void save_extra(Writer& w, std::string const& group) const {
        w.template attr<double>(group + "@a", static_cast<double>(a()));
    }
    template <class Reader>
    void load_extra(Reader& r, std::string const& group) {
        set_a(static_cast<scalar_t>(r.template attr<double>(group + "@a")));
    }

    // Random Gaussian-shift seed of the field, sigma per real component.
    // Complex fields get independent N(0, sigma²) on Re and Im.
    void hot_start(scalar_t sigma) noexcept {
        value_type* const data = field_.data();
        std::size_t const n    = field_.nsites();
        Rng& drv               = sampler_.rng().driver();  // serial draw → driver stream
        if constexpr (std::is_same_v<value_type, std::complex<double>> ||
                      std::is_same_v<value_type, std::complex<float>>) {
            using R = value_type::value_type;
            for (std::size_t i = 0; i < n; ++i) {
                data[i] = value_type{static_cast<R>(static_cast<scalar_t>(drv.normal()) * sigma),
                                     static_cast<R>(static_cast<scalar_t>(drv.normal()) * sigma)};
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                data[i] = static_cast<value_type>(static_cast<scalar_t>(drv.normal()) * sigma);
            }
        }
    }

    // Two-stage warm-up for this replica, returning the total trajectories used.
    //
    // Stage 1 — thermalize on the BASE action to statistical equilibrium. The
    // window is neutralised (δ inflated so (S−E_n)²/2δ² ≈ 0, tilt held at 0) so
    // the HMC samples the plain base action; `n_therm` trajectories take the
    // field from its cold/hot start to a representative equilibrium config.
    //
    // Stage 2 — seat the constraint (S_base in mode A, S_I in mode B, or a custom
    // observable) into its E_n window by coarse Newton adaptation of the tilt
    // (a ← a + ⟨dE⟩/δ²). A deep window sits far from the base equilibrium reached
    // in stage 1, and the fixed-δ quadratic penalty is stiff, so seating it in one
    // step overflows the integrator (NaN ΔH). Stage 2 therefore RAMPS: it walks the
    // window centre from the post-thermalization constraint value out to E_n in
    // steps of ≤ ½δ, adapting `a` each step so the equilibrium follows and the
    // field never lags the centre by more than a window width — the force stays in
    // the integrator's stable range. Once the centre reaches E_n a short Newton
    // seat loop settles the tilt. The warmed `a` is left for NR. `max_traj` bounds
    // stage 2 (a warning if hit).
    int warm_into_window(int n_therm,
                         int max_traj            = 2000,
                         double threshold_sigmas = 1.0,
                         log::Mode log_mode      = log::Mode::normal) {
        // ---- Stage 1: base-action thermalization (window off) ----
        scalar_t const saved_delta = windowed_.delta;
        scalar_t const saved_a     = windowed_.a;
        windowed_.a                = scalar_t{0};
        windowed_.delta            = saved_delta * static_cast<scalar_t>(1e8);
        for (int i = 0; i < n_therm; ++i) {
            (void)sampler_.step(log::Mode::silent);
            ++stats_.n_traj;
        }
        windowed_.delta = saved_delta;
        windowed_.a     = saved_a;

        // ---- Stage 2: seat into the window ----
        auto _                = log::scope(id_);
        constexpr int k_batch = 20;  // trajectories per a-update / ⟨dE⟩ estimate
        int traj              = 0;

        // Ramp phase: the fixed-δ window is stiff, so slamming it onto a field
        // that thermalized far from E_n (deep windows: |E_n − ⟨Q⟩_base| ≫ δ)
        // produces a force ∝ |Q − E_n|/δ² that overflows the integrator's stable
        // step and NaNs the trajectory. Instead walk the window CENTRE from the
        // current constraint value out to E_n in steps of ≤ ½δ, adapting the
        // tilt each step so the equilibrium follows — the field stays within a
        // window width of the centre throughout, so the force never spikes.
        scalar_t const true_e_n = windowed_.E_n;
        scalar_t const q0       = windowed_.constraint_value(field_);
        scalar_t const gap      = true_e_n - q0;
        int const n_ramp        = ramp_steps(gap, windowed_.delta);
        for (int k = 1; k <= n_ramp && traj < max_traj; ++k) {
            windowed_.E_n = ramp_centre(q0, gap, k, n_ramp);
            scalar_t sum  = scalar_t{0};
            int cnt       = 0;
            for (; cnt < k_batch && traj < max_traj; ++cnt) {
                (void)sampler_.step(log::Mode::silent);
                ++traj;
                ++stats_.n_traj;
                sum += windowed_.last_constraint() - windowed_.E_n;
            }
            windowed_.a = nr_update(windowed_.a, sum / static_cast<scalar_t>(cnt), windowed_.delta);
        }
        windowed_.E_n = true_e_n;  // restore the real window centre

        // Seat phase: Newton a-adaptation at the true centre. The field is now
        // within a few δ (the ramp brought it in), so this converges without the
        // integrator ever seeing a spike. `last_constraint()` is the windowed
        // observable Q — the action for real LLR, S_I for complex, or a custom
        // observable — matching sample()/energy().
        double ratio = 0.0;
        bool reached = false;
        while (traj < max_traj) {
            scalar_t sum = scalar_t{0};
            int cnt      = 0;
            for (; cnt < k_batch && traj < max_traj; ++cnt) {
                (void)sampler_.step(log::Mode::silent);
                ++traj;
                ++stats_.n_traj;
                sum += windowed_.last_constraint() - windowed_.E_n;
            }
            scalar_t const mean_dE = sum / static_cast<scalar_t>(cnt);
            ratio                  = static_cast<double>((windowed_.delta != scalar_t{0})
                                                             ? (std::abs(mean_dE) / windowed_.delta)
                                                             : scalar_t{0});
            if (ratio < threshold_sigmas) {
                reached = true;
                break;
            }
            windowed_.a = nr_update(windowed_.a, mean_dE, windowed_.delta);
        }
        if (log_mode == log::Mode::normal) {
            if (reached) {
                log::info("repl",
                          "warm  therm {} + seat {} traj  |⟨S⟩-E_n|/δ={:.3f}  a={:+.3f}",
                          n_therm,
                          traj,
                          ratio,
                          static_cast<double>(windowed_.a));
            } else {
                log::warn("repl",
                          "warm  therm {} + {} traj, |⟨S⟩-E_n|/δ={:.3f} (not seated; NR continues)",
                          n_therm,
                          max_traj,
                          ratio);
            }
        }
        return n_therm + traj;
    }

    [[nodiscard]] ReplicaStats const& stats() const noexcept { return stats_; }

private:
    using Windowed = windowed_action_type;

    // The samplers differ only in whether they take an integrator, and the ones
    // that do name it themselves — so the branch reads the sampler's own alias
    // rather than a parameter the replica would otherwise have to carry for a
    // sampler that has no integrator. Guaranteed elision constructs the return
    // object directly into `sampler_`, which matters: an updater holds references
    // into this replica and is not movable.
    [[nodiscard]] static sampler_type
    make_sampler_(Windowed& w, field_type& f, Rng rng, sampler_spec_type const& spec) {
        if constexpr (requires { typename Sampler::integrator; }) {
            return sampler_type{
                w, f, std::move(rng), spec, typename Sampler::integrator{}, log::Mode::silent};
        } else {
            return sampler_type{w, f, std::move(rng), spec, log::Mode::silent};
        }
    }

    std::string id_;
    field_type field_;
    Windowed windowed_;
    sampler_type sampler_;
    ReplicaStats stats_{};
};

}  // namespace reticolo::orch::llr
