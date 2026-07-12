#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/math/vec_libm.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace reticolo::updater {

struct HmcSpec {
    double tau = 1.0;
    int n_md   = 20;
    // Team size for this HMC's per-site passes (force, s_full, kick, drift, …).
    // 0 = inherit the OpenMP ambient (OMP_NUM_THREADS). Set it to pin a trajectory
    // to N threads regardless of the global — the basis for later mixing replica-
    // and site-level parallelism (m threads/replica × max/m replicas).
    int n_threads = 0;
    // Slabs handed to each thread — the partition granularity. 0 = the default 1
    // (one slab per thread). Higher = finer slabs (smaller cache footprint). Since
    // the slab count tracks the team, reproducing a chain means holding both
    // n_threads AND slabs_per_thread fixed (the s_full/kinetic reduces re-fold
    // otherwise); the maps stay bit-identical regardless.
    int slabs_per_thread = 0;
};

struct HmcResult {
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention for ΔH = H_final - H_initial
    double dH     = 0.0;
    bool accepted = false;

    [[nodiscard]] double acceptance() const noexcept { return accepted ? 1.0 : 0.0; }
};

// Hybrid Monte Carlo — one class for both scalar (`Lattice<F>`) and link
// (`LinkLattice<F>`) fields. The integrator is a type parameter, the field
// type is a type parameter; the trajectory body is field-agnostic via
// `flat_size(field)` + raw-pointer iteration.
//
// Standard Wilson convention everywhere: weight ∝ exp(-S), H = K + S.
// `compute_force` returns the force F = -dS/dfield (matches what the
// integrator's `mom += k_dt * F` expects).
//
// Kinetic term:
//   - real F:    H_kin = (1/2) sum_i mom_i^2
//   - complex F: H_kin = (1/2) sum_i |mom_i|^2  (= (Re^2 + Im^2)/2 per slot)
//
// Momentum sampling: Hmc OWNS a StreamSet<R> sized from the canonical field
// partition — one site stream per slab, bound one-to-one by the partition
// item index (exec::field_visit_indexed), never by the executing thread or
// the schedule. The count and the (n_threads, slabs_per_thread) config are
// frozen at construction; every fill re-derives the partition and throws
// std::logic_error if it no longer matches the streams. The Metropolis
// accept draws from the set's dedicated driver stream. Field-type mapping:
//   - double:                 per-slab batched normal_fill from stream i
//   - std::complex<double>:   reinterpret-cast to double[2], per-slab fill;
//                             independent N(0,1) on Re and Im — the correct
//                             complex Gaussian (§29.5.4 guarantees layout).
//   - float / complex<float>: per-slab double fill narrowed to float
//   - matrix links:           Group::sample_algebra_slab per (slab, μ)
//   - anything else:          per-element stream_i.normal()

// The field-agnostic action gate `action::HmcAction<A, Field>` (defined in
// action/concepts.hpp) is the single source of truth: any action with
// `s_full(field)` and `compute_force(field, force)` qualifies — scalar, U(1),
// and `Wilson<G>` over `MatrixLinkLattice<G, T>` all satisfy it directly.
using action::HmcAction;

// Optional: action caches its last `s_full` evaluation and exposes a
// restore-from-snapshot hook so HMC can roll the cache back on a rejected
// trajectory. Mirrored separately on `s_imag` for complex-LLR actions whose
// imaginary part is the LLR constraint observable.
template <class A>
concept HasSFullCache = requires(A const& a) {
    { a.last_s_full() } noexcept;
    { a.restore_last_s_full(a.last_s_full()) } noexcept;
};

template <class A>
concept HasSImagCache = requires(A const& a) {
    { a.last_s_imag() } noexcept;
    { a.restore_last_s_imag(a.last_s_imag()) } noexcept;
};

// Compile-time tag: Field is a MatrixLinkLattice (carries a Group type alias).
template <class Field>
concept MatrixLinkField = requires { typename Field::group_type; };

template <class A,
          class R,
          class Integrator = integ::Omelyan2,
          class Field      = Lattice<typename A::value_type>,
          class Scalar     = A::value_type>
    requires HmcAction<A, Field> && Rng<R> && (JumpStream<R> || KeyedStream<R>)
class Hmc {
public:
    using value_type = Scalar;
    using integrator = Integrator;

    static constexpr std::string_view log_tag = "hmc";

    // The defaulted Integrator parameter is a deduction anchor only: it lets
    // CTAD pick the integrator from a tag value (e.g. integ::leapfrog) while
    // A/R/Field/Scalar deduce from the other args. Omit it for Omelyan2.
    //
    // `rng` is taken BY VALUE: the caller hands over a freshly-seeded family
    // generator as the entropy source and Hmc owns the randomness from there —
    // one uniform_u64() draw seeds the StreamSet, sized to the canonical
    // partition of `field` under this Hmc's (frozen) thread/slab config.
    // n_threads = 0 resolves to the ambient team ONCE, here, so a later
    // OMP_NUM_THREADS change cannot silently re-partition a running chain
    // (the per-fill check below would refuse it anyway).
    Hmc(A const& action,
        Field& field,
        R rng,
        HmcSpec const& spec,
        Integrator         = {},
        log::Mode announce = log::Mode::normal)
        : action_{action}, field_{field}, mom_{field.indexing()}, force_{field.indexing()},
          old_field_{field.indexing()}, tau_{spec.tau}, n_md_{spec.n_md},
          n_threads_{resolve_threads_(spec.n_threads)}, n_slabs_{spec.slabs_per_thread},
          streams_{rng.uniform_u64(), slab_count_(field, n_threads_, n_slabs_), announce} {
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    // The action is retained by `const&` — binding it to a temporary would
    // dangle for the life of the driver. Delete construction from an rvalue
    // action so the mistake is a compile error. (field_ is a non-const lvalue
    // reference, so a temporary there already fails to bind; rng is by-value
    // ownership transfer, so an rvalue is exactly right for it.)
    Hmc(A const&& action,
        Field& field,
        R rng,
        HmcSpec const& spec,
        Integrator         = {},
        log::Mode announce = log::Mode::normal) = delete;

    void describe(log::Entry& e) const {
        e.line("HMC<{}>", Integrator::name);
        e.param("τ={:.3f}", tau_);
        e.param("n_md={}", n_md_);
        if (n_threads_ > 1) {
            e.param("threads={}", n_threads_);
        }
        if (n_slabs_ > 0) {
            e.param("slabs/thr={}", n_slabs_);
        }
        e.param("streams={}", streams_.n_streams());
    }

    HmcResult step(log::Mode log_mode = log::Mode::normal) {
        // Bind this HMC's team size + slab granularity for the whole trajectory:
        // every per-site pass below (sample, copy, kinetic, s_full, MD force/drift/
        // kick) slices across n_threads_ (0 = ambient) into n_slabs_ per thread
        // (0 = 1). Restored on return.
        exec::team_scope const team{n_threads_};
        exec::slab_scope const slabs{n_slabs_};
        sample_momenta_();

        // Snapshot for rejection rollback — flat-buffer copy.
        std::size_t const n = flat_size(field_);
        copy_flat_(old_field_.data(), field_.data(), n);
        double const kin0 = kinetic_();
        auto const s0     = static_cast<double>(action_.s_full(field_));
        // Snapshot the action's raw-scalar cache(s) at h0 so a reject can
        // roll them back to match the restored phi0 field. No-op for
        // actions that don't satisfy the cache concepts.
        auto const cache_snap = capture_action_cache_();
        double const h0       = kin0 + s0;

        run_md_(tau_, n_md_);

        double const kin1 = kinetic_();
        auto const s1     = static_cast<double>(action_.s_full(field_));
        double const h1   = kin1 + s1;
        // NOLINTNEXTLINE(readability-identifier-naming) physics convention
        double const dH = h1 - h0;

        if (std::isnan(dH)) {
            log::warn("hmc",
                      "ΔH=NaN on trajectory {} — integrator unstable, forcing reject",
                      step_count_ + 1);
        }

        bool const accepted = (dH <= 0.0) || (streams_.uniform() < std::exp(-dH));
        if (!accepted) {
            copy_flat_(field_.data(), old_field_.data(), n);
            // s_full now reflects the field BEFORE the trajectory — restore to s0.
            last_s_full_ = s0;
            restore_action_cache_(cache_snap);
        } else {
            last_s_full_ = s1;
        }
        ++step_count_;
        if (log_mode == log::Mode::normal) {
            log::info("hmc",
                      "traj {:>6}  ΔH={:+.3e}  {}",
                      step_count_,
                      dH,
                      accepted ? "accept" : "reject");
        }
        return {.dH = dH, .accepted = accepted};
    }

    // The s_full value of the action on the current field, as computed at the
    // end of the most recent trajectory (== s_full before the rejected step,
    // or after the accepted step). Avoids one re-sweep at the call site.
    // NaN if no trajectory has run yet.
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }

    [[nodiscard]] HmcSpec spec() const noexcept {
        return {.tau = tau_, .n_md = n_md_, .n_threads = n_threads_, .slabs_per_thread = n_slabs_};
    }
    // tau / n_md are tunable; the threading config is frozen at construction
    // (the RNG streams are bound to its partition) — changing it would need a
    // new Hmc, so a mismatch here is refused rather than silently re-binding.
    void set_spec(HmcSpec const& s) {
        if (resolve_threads_(s.n_threads) != n_threads_ || s.slabs_per_thread != n_slabs_) {
            throw std::logic_error{"Hmc::set_spec: n_threads/slabs_per_thread are frozen at "
                                   "construction (RNG streams are bound to the partition)"};
        }
        tau_  = s.tau;
        n_md_ = s.n_md;
    }

    // The owned stream set — one site stream per canonical slab + the driver.
    // Exposed for checkpointing: io::save_config(path, field, hmc.rng(), i)
    // and load_config(path, field, hmc.rng()) round-trip the full RNG state.
    [[nodiscard]] StreamSet<R>& rng() noexcept { return streams_; }
    [[nodiscard]] StreamSet<R> const& rng() const noexcept { return streams_; }

    // Exposed so reversibility tests can run a bare integrator without
    // the Metropolis acceptance and momentum sampling.
    [[nodiscard]] Field& momentum() noexcept { return mom_; }
    [[nodiscard]] Field const& momentum() const noexcept { return mom_; }
    [[nodiscard]] Field& force_buffer() noexcept { return force_; }

    void integrate_only(double tau, int n_md) noexcept {
        exec::team_scope const team{n_threads_};
        exec::slab_scope const slabs{n_slabs_};
        run_md_(tau, n_md);
    }

private:
    // No persistent region around the MD loop: each force/drift/kick opens its
    // own fork/join region (see integ_ops / SiteAction). Measured on libgomp,
    // per-op fork/join scales to ~22x on 32 cores while a persistent region with
    // many internal `omp for` barriers collapses to ~2x — libgomp worksharing
    // barriers are the bottleneck, its fork/join is not.
    void run_md_(double tau, int n_md) noexcept {
        Integrator::run(action_, field_, mom_, force_, tau, n_md);
    }

    // Flat elementwise copy (rollback snapshot / restore) — a write-disjoint map
    // over the raw buffer. Unlike the leapfrog ops this is not a producer/consumer
    // in the trajectory chain, so it needs no slab-ownership alignment: a plain
    // chunked flat copy serves both site fields (Scalar per site) and the
    // multi-component link buffer (whose flat index isn't site-major).
    void copy_flat_(Scalar* dst, Scalar const* src, std::size_t n) const noexcept {
        reticolo::exec::parallel_map_ranges(
            n, sizeof(Scalar), 1, [dst, src](std::size_t base, std::size_t cnt) {
                std::size_t const end = base + cnt;
                for (std::size_t i = base; i < end; ++i) {
                    dst[i] = src[i];
                }
            });
    }

    // n_threads = 0 resolves to the ambient team once, at construction time.
    [[nodiscard]] static int resolve_threads_(int n) noexcept {
        return n > 0 ? n : exec::traverse_threads(0, 0);
    }

    // The canonical slab count for `field` under this Hmc's thread/slab config
    // — the stream count the set is built with, and the value every fill is
    // checked against.
    [[nodiscard]] static std::size_t slab_count_(Field const& field, int nthr, int slabs) {
        exec::team_scope const team{nthr};
        exec::slab_scope const sl{slabs};
        return exec::partition(field).n_items;
    }

    // Strict slab↔stream binding guard, run before every momentum fill (under
    // the trajectory's team/slab scopes): the partition must still produce
    // exactly one item per site stream. It cannot drift under normal use —
    // the config is frozen and set_spec refuses threading changes — so a
    // mismatch means the environment changed under us; refuse to sample
    // rather than silently rebind streams to different slabs.
    void check_streams_() const {
        std::size_t const items = exec::partition(mom_).n_items;
        if (items != streams_.n_streams()) {
            throw std::logic_error{"Hmc: field partition (" + std::to_string(items) +
                                   " slabs) no longer matches the " +
                                   std::to_string(streams_.n_streams()) +
                                   " RNG streams bound at construction"};
        }
    }

    void sample_momenta_() {
        check_streams_();
        if constexpr (MatrixLinkField<Field>) {
            // Algebra-aware momentum sampling per direction. The group writes
            // anti-hermitian elements with the right structural zeros; raw
            // normal_fill would put noise on the constrained slots. Slab i of
            // every direction draws from site stream i, in μ order — the draw
            // sequence is a pure function of the frozen partition.
            using Group            = Field::group_type;
            std::size_t const d    = mom_.ndims();
            std::size_t const span = mom_.link_span();  // padded component stride
            for (std::size_t mu = 0; mu < d; ++mu) {
                Scalar* const pblk = mom_.mu_block_data(mu);
                exec::field_visit_indexed(
                    mom_, 1, [this, pblk, span](std::size_t i, std::size_t base, std::size_t cnt) {
                        Group::sample_algebra_slab(pblk + base, streams_.site_stream(i), span, cnt);
                    });
            }
        } else if constexpr (std::is_same_v<Scalar, double>) {
            double* const m = mom_.data();
            exec::field_visit_indexed(
                mom_, 1, [this, m](std::size_t i, std::size_t base, std::size_t cnt) {
                    streams_.site_stream(i).normal_fill(m + base, cnt);
                });
        } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
            // Independent N(0,1) on Re/Im is the correct complex-Gaussian sampling;
            // std guarantees the {Re,Im} layout (§29.5.4). Two doubles per site,
            // so slab i covers doubles [2·base, 2·(base+cnt)).
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            auto* const md = reinterpret_cast<double*>(mom_.data());
            exec::field_visit_indexed(
                mom_, 1, [this, md](std::size_t i, std::size_t base, std::size_t cnt) {
                    streams_.site_stream(i).normal_fill(md + (2 * base), 2 * cnt);
                });
        } else if constexpr (std::is_same_v<Scalar, float>) {
            float* const m = mom_.data();
            exec::field_visit_indexed(
                mom_, 1, [this, m](std::size_t i, std::size_t base, std::size_t cnt) {
                    fill_narrowed_slab_(streams_.site_stream(i), m + base, cnt);
                });
        } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            auto* const mf = reinterpret_cast<float*>(mom_.data());
            exec::field_visit_indexed(
                mom_, 1, [this, mf](std::size_t i, std::size_t base, std::size_t cnt) {
                    fill_narrowed_slab_(streams_.site_stream(i), mf + (2 * base), 2 * cnt);
                });
        } else {
            Scalar* const m = mom_.data();
            exec::field_visit_indexed(
                mom_, 1, [this, m](std::size_t i, std::size_t base, std::size_t cnt) {
                    R& r = streams_.site_stream(i);
                    for (std::size_t j = base; j < base + cnt; ++j) {
                        m[j] = static_cast<Scalar>(r.normal());
                    }
                });
        }
    }

    // Per-slab standard-normal fill narrowed to float. Draws doubles (so the
    // stream advances exactly as the double path would) into a thread_local
    // scratch, then narrows. Momentum precision is non-critical — the kinetic
    // energy is summed in double regardless of the field's scalar type.
    static void fill_narrowed_slab_(R& r, float* out, std::size_t n) {
        thread_local std::vector<double> scratch;
        double* const s = reticolo::exec::thread_scratch(scratch, n);
        r.normal_fill(s, n);
        for (std::size_t j = 0; j < n; ++j) {
            out[j] = static_cast<float>(s[j]);
        }
    }

    [[nodiscard]] double kinetic_() const {
        double kin = 0.0;
        if constexpr (MatrixLinkField<Field>) {
            // K = (1/2)·Tr(P†P) = ‖h‖² for SU(N) with P = i·(h·σ). Hamilton's
            // dU/dt = P·U gives ∂K/∂P = P → K = (1/2)·Tr(P†P) = ‖h‖² (no extra
            // 1/2 here). For exp(-K) detailed balance the algebra coords are
            // sampled with variance 1/2 (see SU2::sample_algebra_slab).
            using Group            = Field::group_type;
            std::size_t const d    = mom_.ndims();
            std::size_t const span = mom_.link_span();  // padded component stride
            // Parallel deterministic reduce per direction; ½ applied once.
            // gran = 8 (kinetic_range's internal batch).
            double raw = 0.0;
            for (std::size_t mu = 0; mu < d; ++mu) {
                Scalar const* const pblk = mom_.mu_block_data(mu);
                raw += reticolo::exec::field_reduce(
                    mom_, 8, [pblk, span](std::size_t base, std::size_t cnt) {
                        return Group::kinetic_range(pblk, span, base, cnt);
                    });
            }
            kin += 0.5 * raw;
        } else {
            Scalar const* const p = mom_.data();
            double const sum =
                reticolo::exec::field_reduce(mom_, 1, [p](std::size_t base, std::size_t cnt) {
                    return reticolo::exec::lane_sum8(base, cnt, [p](std::size_t i) {
                        if constexpr (is_complex_v_) {
                            return static_cast<double>(std::norm(p[i]));
                        } else {
                            auto const pi = static_cast<double>(p[i]);
                            return pi * pi;
                        }
                    });
                });
            kin += 0.5 * sum;
        }
        return kin;
    }

    static constexpr bool is_complex_v_ =
        requires(Scalar f) { std::norm(f); } && !std::is_arithmetic_v<Scalar>;

    struct CacheSnap_ {
        std::optional<double> s_full;
        std::optional<double> s_imag;
    };

    [[nodiscard]] CacheSnap_ capture_action_cache_() const noexcept {
        CacheSnap_ snap;
        if constexpr (HasSFullCache<A>) {
            snap.s_full = action_.last_s_full();
        }
        if constexpr (HasSImagCache<A>) {
            snap.s_imag = action_.last_s_imag();
        }
        return snap;
    }

    void restore_action_cache_(CacheSnap_ const& snap) const noexcept {
        if constexpr (HasSFullCache<A>) {
            action_.restore_last_s_full(*snap.s_full);
        }
        if constexpr (HasSImagCache<A>) {
            action_.restore_last_s_imag(*snap.s_imag);
        }
    }

    A const& action_;
    Field& field_;
    Field mom_;
    Field force_;
    Field old_field_;
    double tau_;
    int n_md_;
    int n_threads_;  // bound via team_scope each trajectory; resolved from
                     // ambient at construction, frozen thereafter
    int n_slabs_;    // bound via slab_scope each trajectory; 0 = 1 slab/thread
    // Owned randomness: one site stream per canonical slab + a driver for the
    // serial draws. Declared after the threading fields — its size derives
    // from them (member init order).
    StreamSet<R> streams_;
    // last s_full of `field_` after the most recent trajectory. NaN until the
    // first trajectory runs. Used by Replica / Exchange so they can read the
    // current action without re-sweeping.
    double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

    // Monotonically advanced on every trajectory (silent or not). Reflects
    // cumulative work, not lines printed — so therm + prod trajectories
    // share one numbering. Used by the in-method log line.
    std::size_t step_count_ = 0;
};

}  // namespace reticolo::updater
