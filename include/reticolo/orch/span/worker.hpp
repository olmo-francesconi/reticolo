#pragma once

#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/samplers.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace reticolo::orch::span {

// One parameter-span worker: a self-contained chain pinned to a fixed action
// (its parameters set by the span grid). Any scalar or gauge action can be
// span-swept without new code.
//
// SAME PARAMETER STRUCTURE AS `orch::llr::Replica`, because they are the same
// kind of thing — an `orch::Worker` wrapping one action and one updater:
//
//     Chain  <Action, Rng, Sampler>
//     Replica<Action, Rng, Sampler, Constraint = void>
//
// The sampler has NO default: which updater drives a run is a decision worth
// reading at the call site, not one to inherit silently. The element and field
// types are not parameters at all — the action names both (`value_type` /
// `field_type`). The MD integrator is not one either: it rides on the sampler
// tag (`updater::HmcSampler<updater::integ::Omelyan4>`; `HmcSampler<>` is
// Omelyan2), so it simply does not exist when the sampler is a local updater.
//
// Owns its field + a by-value copy of the action + its updater (which holds
// references into both, so a Chain is non-movable and lives in a
// std::vector<std::unique_ptr<Chain>>, exactly like llr::Replica). Satisfies
// orch::Worker (`id`) and orch::Checkpointable (`field` + `rng`), so the generic
// ensemble checkpoint can snapshot a span run unchanged.

template <class Action, class Rng, class Sampler>
class Chain {
public:
    using value_type        = Action::value_type;
    using field_type        = Action::field_type;
    using action_type       = Action;
    using sampler_type      = Sampler::template type<Action, Rng>;
    using sampler_spec_type = Sampler::spec_type;

    // Whether the sampler's step result carries the HMC diagnostics.
    // `acceptance()` is the cross-updater accessor (`updater::Updater`); dH and
    // accepted are algorithm-specific, so the driver records them only when they
    // exist — the same split the apps make between /stats/dH + /stats/accepted
    // and /stats/acceptance.
    using step_result                 = decltype(std::declval<sampler_type&>().step());
    static constexpr bool k_hmc_stats = requires(step_result const& r) {
        r.dH;
        r.accepted;
    };

    static constexpr std::string_view log_tag = "chain";

    Chain(std::string id,
          field_type::SizeVec const& shape,
          Action action,
          Rng rng,
          sampler_spec_type const& spec)
        : id_{std::move(id)}, field_{shape}, action_{std::move(action)},
          sampler_{make_sampler_(action_, field_, std::move(rng), spec)} {}

    Chain(Chain const&)            = delete;
    Chain& operator=(Chain const&) = delete;
    Chain(Chain&&)                 = delete;
    Chain& operator=(Chain&&)      = delete;
    ~Chain()                       = default;

    // Announce the nested sampler once (τ / n_md / integrator, or σ). The span
    // driver calls this on a representative worker so the shared sampler config
    // prints once instead of per worker.
    void announce_sampler() const { log::algo(sampler_); }

    [[nodiscard]] std::string_view id() const noexcept { return id_; }

    void thermalize(int n) {
        for (int i = 0; i < n; ++i) {
            (void)sampler_.step(log::Mode::silent);
        }
    }

    // One production update; stash the diagnostics for the serial drain. The
    // per-config observables are read straight off `field()` in the drain, so
    // they need no scratch here.
    void advance() {
        auto const st    = sampler_.step(log::Mode::silent);
        last_acceptance_ = st.acceptance();
        if constexpr (k_hmc_stats) {
            last_dh_  = st.dH;
            last_acc_ = st.accepted;
        }
    }

    [[nodiscard]] double last_acceptance() const noexcept { return last_acceptance_; }

    [[nodiscard]] double last_dh() const noexcept
        requires k_hmc_stats
    {
        return last_dh_;
    }
    [[nodiscard]] bool last_accepted() const noexcept
        requires k_hmc_stats
    {
        return last_acc_;
    }

    [[nodiscard]] field_type& field() noexcept { return field_; }
    [[nodiscard]] field_type const& field() const noexcept { return field_; }
    [[nodiscard]] Action const& action() const noexcept { return action_; }
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return sampler_.rng(); }

private:
    static_assert(updater::Updater<sampler_type>, "the span sampler must model updater::Updater");

    // Identical to llr::Replica's: the tag's optional `integrator` alias decides
    // whether the updater ctor takes one. Guaranteed elision constructs in
    // place — an updater holds references into this worker and is not movable.
    [[nodiscard]] static sampler_type
    make_sampler_(Action& a, field_type& f, Rng rng, sampler_spec_type const& spec) {
        if constexpr (requires { typename Sampler::integrator; }) {
            return sampler_type{
                a, f, std::move(rng), spec, typename Sampler::integrator{}, log::Mode::silent};
        } else {
            return sampler_type{a, f, std::move(rng), spec, log::Mode::silent};
        }
    }

    std::string id_;
    field_type field_;
    Action action_;
    sampler_type sampler_;
    double last_acceptance_ = 0.0;
    double last_dh_         = 0.0;
    bool last_acc_          = false;
};

}  // namespace reticolo::orch::span
