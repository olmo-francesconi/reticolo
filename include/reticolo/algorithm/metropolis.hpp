#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace reticolo::alg {

// Update order for the scalar nearest-neighbour fast path.
//   Sequential   — one in-place lexicographic sweep; site i+1 sees i's
//                  accepted update. The default; bit-stable RNG order.
//   Checkerboard — even sublattice then odd, each against the frozen other
//                  colour. A different (still valid) Markov chain: proposals
//                  and the exp(-ds) acceptance weights batch through Sleef,
//                  moving both transcendentals off the scalar critical path.
//                  Only available for actions with an NN ds_local_from_nbrs;
//                  requested on any other action it logs once and falls back
//                  to Sequential.
enum class Sweep : std::uint8_t { Sequential, Checkerboard };

struct MetropolisSpec {
    double sigma = 1.0;
    Sweep sweep  = Sweep::Sequential;
};

struct MetropolisResult {
    // Counts (not a bool) — named distinctly from HmcResult::accepted so the
    // blessed cross-updater accessor stays `acceptance()`, never `.accepted`.
    std::size_t n_accepted = 0;
    std::size_t n_attempts = 0;

    [[nodiscard]] double acceptance() const noexcept {
        return n_attempts == 0 ? 0.0
                               : static_cast<double>(n_accepted) / static_cast<double>(n_attempts);
    }
};

// Metropolis sweep — one class for both site fields (Lattice<F>) and link
// fields (LinkLattice<F>). Three code paths picked at compile time:
//
//   1. HasDsLocalFromNbrs (scalar action with NN-only ds_local):
//      visit_nn fast path with direct-stride neighbour reads; the
//      acceptance body reuses the in-register `phi` instead of re-reading.
//
//   2. HasProposal (action provides its own proposal kernel):
//      proposal comes from action_.propose(field, loc..., rng); fallback
//      is Gaussian. Hooked into both the fast and generic paths.
//
//   3. Generic field.for_each_update visit (works for Lattice via
//      (T&, Site) and LinkLattice via (T&, Site, mu)). The body's
//      parameter pack `loc...` absorbs whichever extra coords the field
//      yields, and `action_.ds_local(field, loc..., new_v)` unpacks the
//      matching arity.

template <class A, class R, class F = typename A::value_type, class Field = Lattice<F>>
    requires(action::LocalAction<A, F> || gauge::LinkLocalAction<A, F>) && Rng<R>
class Metropolis {
public:
    using value_type = F;

    static constexpr std::string_view log_tag = "metr";

    Metropolis(A const& action,
               Field& field,
               R& rng,
               MetropolisSpec const& spec,
               log::Mode announce = log::Mode::normal)
        : action_{action}, field_{field}, rng_{rng}, sigma_{spec.sigma}, sweep_{spec.sweep} {
        if constexpr (!action::HasDsLocalFromNbrs<A, F>) {
            if (sweep_ == Sweep::Checkerboard) {
                log::warn("metr",
                          "checkerboard sweep needs an NN ds_local_from_nbrs action; "
                          "falling back to sequential");
                sweep_ = Sweep::Sequential;
            }
        }
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    void describe(log::Entry& e) const {
        e.line("Metropolis");
        e.param("σ={:.3f}", sigma_);
        e.param("sweep={}", sweep_ == Sweep::Checkerboard ? "checkerboard" : "sequential");
    }

    MetropolisResult step(log::Mode log_mode = log::Mode::normal) {
        MetropolisResult stats{};

        if constexpr (action::HasSweepState<A, F> || gauge::HasLinkSweepState<A, F>) {
            action_.begin_sweep(field_);
        }

        if constexpr (action::HasDsLocalFromNbrs<A, F>) {
            if (sweep_ == Sweep::Checkerboard) {
                checkerboard_color_(field_.even_sites(), stats);
                checkerboard_color_(field_.odd_sites(), stats);
                ++step_count_;
                if (log_mode == log::Mode::normal) {
                    log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
                }
                return stats;
            }
            // Scalar fast path: visit_nn precomputes the NN sum and the body
            // accepts (phi, nbrs) directly. Bit-stable order, vectorised
            // neighbour sum.
            F* const fdata = field_.data();
            action::detail::visit_nn<F>(
                field_, [this, fdata, &stats](std::size_t i, F phi, F nbrs) {
                    F const new_v = propose_local_(phi, i);
                    auto const ds =
                        static_cast<double>(action_.ds_local_from_nbrs(phi, new_v, nbrs));
                    ++stats.n_attempts;
                    if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                        if constexpr (action::HasSweepState<A, F>) {
                            action_.commit_accept(field_, Site{i}, new_v);
                        }
                        fdata[i] = new_v;
                        ++stats.n_accepted;
                    }
                });
            ++step_count_;
            if (log_mode == log::Mode::normal) {
                log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
            }
            return stats;
        }

        // The `auto... loc` pack captures the field's location coords:
        // Lattice yields (ref, Site) → loc=(Site); LinkLattice yields
        // (ref, Site, mu) → loc=(Site, mu). Action's ds_local unpacks the arity.
        field_.for_each_update([&](F& ref, auto... loc) {
            F const old   = ref;
            F const new_v = propose_(old, loc...);
            auto const ds = static_cast<double>(action_.ds_local(field_, loc..., new_v));
            ++stats.n_attempts;
            if (ds <= 0.0 || rng_.uniform() < std::exp(-ds)) {
                if constexpr (action::HasSweepState<A, F> || gauge::HasLinkSweepState<A, F>) {
                    action_.commit_accept(field_, loc..., new_v);
                }
                ref = new_v;
                ++stats.n_accepted;
            }
        });
        ++step_count_;
        if (log_mode == log::Mode::normal) {
            log::info("metr", "sweep {:>6}  acc={:.3f}", step_count_, stats.acceptance());
        }
        return stats;
    }

    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    void set_sigma(double s) noexcept { sigma_ = s; }

private:
    template <class... Loc>
    [[nodiscard]] F propose_(F old, Loc... loc) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, loc..., rng_);
        } else {
            return old + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    // Variant for the visit_nn fast path that already has phi in a register
    // and so can skip re-reading from the lattice. Custom proposals still
    // route through the action.
    [[nodiscard]] F propose_local_(F phi, std::size_t i) {
        if constexpr (action::HasProposal<A, F, R>) {
            return action_.propose(field_, Site{i}, rng_);
        } else {
            return phi + static_cast<F>(sigma_ * rng_.normal());
        }
    }

    // One sublattice of a checkerboard sweep: propose, score, and accept every
    // site of `color` against the frozen opposite colour. Proposals and the
    // exp(-ds) acceptance weights are batched (Sleef) so neither transcendental
    // sits on the scalar accept loop; the NN sum gather and the accept compare
    // are all that remain serial. `mds` holds −ds, so the weight is exp(mds)
    // and the unconditional-accept test is `mds >= 0`.
    void checkerboard_color_(std::span<Site const> color, MetropolisResult& stats) {
        std::size_t const m = color.size();
        if (m == 0) {
            return;
        }
        F* const data                = field_.data();
        Indexing const& idx          = field_.indexing_ref();
        Site::value_type const* next = idx.next_data();
        Site::value_type const* prev = idx.prev_data();
        std::size_t const d          = idx.ndims();

        thread_local std::vector<F> prop;
        thread_local std::vector<double> mds;
        thread_local std::vector<double> wexp;
        thread_local std::vector<double> nrm;
        prop.resize(m);
        mds.resize(m);
        wexp.resize(m);

        if constexpr (action::HasProposal<A, F, R>) {
            for (std::size_t k = 0; k < m; ++k) {
                prop[k] = action_.propose(field_, color[k], rng_);
            }
        } else {
            nrm.resize(m);
            rng_.normal_fill(nrm.data(), m);
            for (std::size_t k = 0; k < m; ++k) {
                prop[k] = data[color[k].value()] + static_cast<F>(sigma_ * nrm[k]);
            }
        }

        for (std::size_t k = 0; k < m; ++k) {
            std::size_t const s    = color[k].value();
            std::size_t const base = s * d;
            F nbrs                 = F{0};
            for (std::size_t mu = 0; mu < d; ++mu) {
                nbrs += data[next[base + mu]] + data[prev[base + mu]];
            }
            mds[k] = -static_cast<double>(action_.ds_local_from_nbrs(data[s], prop[k], nbrs));
        }

        math::exp_batch(wexp.data(), mds.data(), m);

        for (std::size_t k = 0; k < m; ++k) {
            ++stats.n_attempts;
            bool const accepted = (mds[k] >= 0.0) || (rng_.uniform() < wexp[k]);
            if (accepted) {
                Site const s = color[k];
                if constexpr (action::HasSweepState<A, F>) {
                    action_.commit_accept(field_, s, prop[k]);
                }
                data[s.value()] = prop[k];
                ++stats.n_accepted;
            }
        }
    }

    A const& action_;
    Field& field_;
    R& rng_;
    double sigma_;
    Sweep sweep_;

    // Cumulative sweep counter — advances on every call, silent or not.
    std::size_t step_count_ = 0;
};

}  // namespace reticolo::alg
