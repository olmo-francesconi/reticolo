#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/exec/nn_checkerboard.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/random_fill.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reticolo::updater {

// Local Metropolis — the second implementation of `updater::Updater`, and the
// same class shape as `updater::Hmc`: action by `const&` (rvalue ctor deleted),
// field by `&`, RNG taken BY VALUE with the class owning a `StreamSet<R>` sized
// to the field's canonical partition, and (n_threads, slabs_per_thread) resolved
// once at construction and frozen. One class covers site fields and link fields,
// as Hmc does, and for the same reason: nothing here is field-specific.
//
// The division of labour is the library's, not this algorithm's:
//
//   action leaf     the physics — couplings and per-site kernels
//   action family   how those kernels sweep a lattice: `s_full`, `compute_force`,
//                   `compute_force_and_kick`, and now `metropolis_stencil`
//   updater         the algorithm and the randomness — here, drawing the proposal
//                   noise and the accept thresholds, driving the colours, and
//                   carrying S
//
// So this class contains no traversal, no neighbour arithmetic and no ΔS: it
// fills two fields and calls one action member per colour, exactly as `Hmc`
// fills the momentum field and calls `compute_force_and_kick` per MD step.
// Adding an action costs nothing — `NNAction::metropolis_stencil` is built from the
// leaf's existing `action_kernel`, so any leaf that runs under HMC runs here.
//
// Weight convention is HMC's: ∝ exp(−S), accept a move with min(1, exp(−ΔS)).
//
// STATE. Where Hmc carries momentum + force + rollback, this carries proposal
// noise + accept thresholds — two sibling fields, and no rollback buffer at all,
// because a rejected local move is discarded before it is written. It is one
// lattice lighter than HMC and has no MD loop.
//
// PARALLELISM. The colour sweep runs over the same canonical partition every
// other per-site pass uses. A colour's neighbours are all of the other colour, so
// it threads with no halo and no lock. Determinism is HMC's guarantee, no
// stronger: bit-identical for a FIXED (n_threads, slabs_per_thread), because the
// streams bind to the partition and the partition tracks the team.

struct MetropolisSpec {
    // Gaussian proposal width: φ' = φ + σ·N(0,1) for a site field; for a link
    // field, the spread of the near-identity group element.
    double sigma = 0.5;
    // Frozen at construction, like HmcSpec's — see the class comment.
    int n_threads        = 0;
    int slabs_per_thread = 0;
};

// A sweep's outcome. Counts, not a bool: a sweep is V independent accept tests.
// `acceptance()` is the accessor `updater::Updater` knows about; `n_accepted` /
// `ds` are the algorithm-specific extras, the peers of `HmcResult::dH`.
using MetropolisResult = action::LocalStats;

// Same parameter structure as `updater::Hmc` — the element and field types are
// not parameters, the action names both.
template <class Action, SamplerRng Rng>
    requires action::LocalAction<Action, typename Action::field_type>
class Metropolis {
public:
    using action_type = Action;
    using rng_type    = Rng;
    using value_type  = Action::value_type;
    using field_type  = Action::field_type;
    using spec_type   = MetropolisSpec;  // see Hmc::spec_type

    static constexpr std::string_view log_tag = "metr";

    Metropolis(Action const& action,
               field_type& field,
               Rng rng,
               MetropolisSpec const& spec,
               log::Mode announce = log::Mode::normal)
        : action_{action}, field_{field}, noise_{field.indexing()}, logu_{field.indexing()},
          sigma_{static_cast<real_scalar_t<value_type>>(spec.sigma)},
          n_threads_{resolve_threads_(spec.n_threads)}, n_slabs_{spec.slabs_per_thread},
          streams_{rng.uniform_u64(), slab_count_(field, n_threads_, n_slabs_), announce},
          last_s_full_{action_.s_full(field_)} {
        // The even-extent precondition belongs to the CHECKERBOARD, not to the
        // algorithm: an odd extent wraps a colour onto itself under periodic BCs,
        // so a colour's sweep would read what it writes. An action that declares a
        // single colour (action::WindowedAction, whose global window forces a
        // sequential pass) never makes that assumption and must not inherit the
        // restriction.
        if (action_.n_colors(field_) > 1 && !exec::checkerboard_ok(field_)) {
            throw std::invalid_argument{
                "Metropolis: the checkerboard sweep needs every lattice extent even "
                "(an odd extent wraps a colour onto itself under periodic BCs)"};
        }
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    // See Hmc's equivalent: binding the action to a temporary would dangle for
    // the life of the driver, so make the mistake a compile error.
    Metropolis(Action const&& action,
               field_type& field,
               Rng rng,
               MetropolisSpec const& spec,
               log::Mode announce = log::Mode::normal) = delete;

    void describe(log::Entry& e) const {
        e.line("Metropolis");
        e.param("σ={:.3f}", sigma_);
        if (n_threads_ > 1) {
            e.param("threads={}", n_threads_);
        }
        e.param("streams={}", streams_.n_streams());
    }

    // One full sweep. Compare Hmc::step(): bind the team, draw the randomness,
    // hand the action the fields it needs, fold the result.
    MetropolisResult step(log::Mode log_mode = log::Mode::normal) {
        exec::team_scope const team{n_threads_};
        exec::slab_scope const slabs{n_slabs_};
        check_streams_();
        // `noise` once per sweep: every element of it is consumed exactly once,
        // in whichever colour owns that site (or that link).
        //
        // `logu` carries one real per SITE, so how often it must be refilled is
        // decided by how many times a sweep visits a site. A SITE field visits
        // each site once — one fill covers the sweep, each colour reading its own
        // half. A LINK field visits each site once per DIRECTION, so a single fill
        // would reuse a site's threshold across its d directions and correlate
        // accept decisions that must be independent; it is refilled per colour.
        // The distinction is the field type, known at compile time, so this is a
        // branch the optimiser removes rather than a runtime test.
        constexpr bool per_color_thresholds = MatrixLinkField<field_type>;
        fill_gaussian(noise_, streams_);
        if constexpr (!per_color_thresholds) {
            fill_log_uniform(logu_, streams_);
        }

        MetropolisResult stats{};
        int const n_colors = action_.n_colors(field_);
        for (int color = 0; color < n_colors; ++color) {
            if constexpr (per_color_thresholds) {
                fill_log_uniform(logu_, streams_);
            }
            stats += action_.metropolis_stencil(field_, color, noise_, logu_, sigma_);
        }

        // Incremental S: exact to rounding and free — the sweep already knew every
        // accepted ΔS, so there is no second O(V) pass. `resync_s_full()`
        // re-derives it for a run long enough to care about the accumulation.
        last_s_full_ += stats.ds;
        ++step_count_;
        if (log_mode == log::Mode::normal) {
            log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
        }
        return stats;
    }

    // S of the current config, carried incrementally across sweeps — the same
    // contract as Hmc::last_s_full(), so orchestration reads it identically.
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    double resync_s_full() { return last_s_full_ = action_.s_full(field_); }

    [[nodiscard]] double sigma() const noexcept { return static_cast<double>(sigma_); }
    void set_sigma(double s) noexcept { sigma_ = static_cast<real_scalar_t<value_type>>(s); }

    [[nodiscard]] MetropolisSpec spec() const noexcept {
        return {.sigma            = static_cast<double>(sigma_),
                .n_threads        = n_threads_,
                .slabs_per_thread = n_slabs_};
    }

    // The owned stream set — one site stream per canonical slab + the driver.
    // Exposed for checkpointing, exactly as Hmc::rng() is.
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return streams_; }
    [[nodiscard]] StreamSet<Rng> const& rng() const noexcept { return streams_; }

private:
    [[nodiscard]] static int resolve_threads_(int n) noexcept {
        return n > 0 ? n : exec::traverse_threads(0, 0);
    }

    [[nodiscard]] static std::size_t slab_count_(field_type const& field, int nthr, int slabs) {
        exec::team_scope const team{nthr};
        exec::slab_scope const sl{slabs};
        return exec::partition(field).n_items;
    }

    // The same strict slab↔stream guard Hmc runs before every momentum fill.
    void check_streams_() const {
        std::size_t const items = exec::partition(field_).n_items;
        if (items != streams_.n_streams()) {
            throw std::logic_error{"Metropolis: field partition (" + std::to_string(items) +
                                   " slabs) no longer matches the " +
                                   std::to_string(streams_.n_streams()) +
                                   " RNG streams bound at construction"};
        }
    }

    Action const& action_;
    field_type& field_;
    field_type noise_;                         // proposal noise — the peer of Hmc's mom_
    Lattice<real_scalar_t<value_type>> logu_;  // accept thresholds, one real per site
    real_scalar_t<value_type> sigma_;
    int n_threads_;
    int n_slabs_;
    StreamSet<Rng> streams_;
    double last_s_full_;
    std::size_t step_count_ = 0;
};

}  // namespace reticolo::updater
