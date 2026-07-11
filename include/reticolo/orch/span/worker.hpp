#pragma once

#include <reticolo/algorithm/concepts.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace reticolo::orch::span {

// One parameter-span worker: a self-contained HMC chain pinned to a fixed
// action (its parameters set by the span grid). Generic over the action, RNG,
// integrator, and field type — the whole point is that any scalar or gauge HMC
// action can be span-swept without new code.
//
// Owns its field + a by-value copy of the action + its HMC (which holds
// references into both, so a Chain is non-movable and lives in a
// std::vector<std::unique_ptr<Chain>>, exactly like llr::Replica). Satisfies
// orch::Worker (`id`) and orch::Checkpointable (`field` + `rng`), so the generic
// ensemble checkpoint can snapshot a span run unchanged.

template <class Action,
          class Rng,
          class Integrator = alg::integ::Omelyan2,
          class T          = Action::value_type,
          class Field      = Lattice<T>>
class Chain {
public:
    using value_type  = T;
    using field_type  = Field;
    using action_type = Action;

    static constexpr std::string_view log_tag = "chain";

    Chain(std::string id,
          Field::SizeVec const& shape,
          Action action,
          Rng rng,
          alg::HmcSpec const& spec)
        : id_{std::move(id)}, field_{shape}, action_{std::move(action)},
          hmc_{action_, field_, std::move(rng), spec, {}, log::Mode::silent} {}

    Chain(Chain const&)            = delete;
    Chain& operator=(Chain const&) = delete;
    Chain(Chain&&)                 = delete;
    Chain& operator=(Chain&&)      = delete;
    ~Chain()                       = default;

    // Announce the nested HMC sampler once (τ / n_md / integrator). The span
    // driver calls this on a representative worker so the shared sampler config
    // prints once instead of per worker.
    void announce_sampler() const { log::algo(hmc_); }

    [[nodiscard]] std::string_view id() const noexcept { return id_; }

    void thermalize(int n) {
        for (int i = 0; i < n; ++i) {
            (void)hmc_.step(log::Mode::silent);
        }
    }

    // One production trajectory; stash ΔH / acceptance for the serial drain.
    // The per-config observables are read straight off `field()` in the drain,
    // so they need no scratch here.
    void advance() {
        auto const st = hmc_.step(log::Mode::silent);
        last_dh_      = st.dH;
        last_acc_     = st.accepted;
    }

    [[nodiscard]] double last_dh() const noexcept { return last_dh_; }
    [[nodiscard]] bool last_accepted() const noexcept { return last_acc_; }

    [[nodiscard]] Field& field() noexcept { return field_; }
    [[nodiscard]] Field const& field() const noexcept { return field_; }
    [[nodiscard]] Action const& action() const noexcept { return action_; }
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return hmc_.rng(); }

private:
    static_assert(alg::Updater<alg::Hmc<Action, Rng, Integrator, Field, T>>,
                  "the span sampler must model alg::Updater");

    std::string id_;
    Field field_;
    Action action_;
    alg::Hmc<Action, Rng, Integrator, Field, T> hmc_;
    double last_dh_ = 0.0;
    bool last_acc_  = false;
};

}  // namespace reticolo::orch::span
