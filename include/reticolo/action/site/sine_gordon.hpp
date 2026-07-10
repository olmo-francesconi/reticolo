#pragma once

#include <reticolo/action/site/formula/sine_gordon_formula.hpp>
#include <reticolo/action/site/site_action.hpp>
#include <reticolo/action/sweep/site.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace reticolo::action {

// Sine-Gordon scalar action: same NN hopping + on-site mass term as Phi4,
// plus a periodic potential -alpha * cos(phi):
//
//   S = sum_x [ -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//               + phi(x)^2
//               - alpha * cos(phi(x)) ]
//
// The per-site transcendental is the only reason this doesn't reduce to the
// plain SiteAction kernels: on the f64 hot path sin(phi) is Sleef-batched into
// the shared scratch by `prep`, and cos(phi) is batched inside the bespoke
// `s_full`. The physics still lives entirely in `detail/sine_gordon_formula.hpp`.

template <class T = double>
struct SineGordon : SiteAction<SineGordon<T>, T> {
    using value_type = T;

    T kappa = T{0};
    T alpha = T{0};

    // Chunk-alignment granularity for the parallel sin/cos transcendental passes.
    // ≥ any SIMD width, so every non-final chunk is a whole number of vectors — the
    // Sleef batch takes the identical vector path it would in a single full sweep,
    // keeping the result bit-identical for any thread count. (The chunk *size* is
    // derived from bytes_per_site by the primitive; this just aligns it.)
    static constexpr std::size_t k_simd_gran = 16;

    void describe(log::Entry& e) const {
        e.line("SineGordon<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("α={:.3f}", alpha);
    }

    // Fill the shared scratch with sin(phi) per site — Sleef-batched on f64,
    // a scalar loop otherwise. Called by SiteAction before each force pass.
    // Worksplit: each chunk sin-batches its own sub-range (elementwise → bit-
    // identical; the chunk is a multiple of the SIMD width so no chunk-boundary
    // tail changes which lanes take the vector path).
    void prep(Lattice<T> const& l) const noexcept {
        std::size_t const n = l.nsites();
        this->ensure_scratch(n);
        T* const sp       = this->scratch_.data();
        T const* const in = l.data();
        reticolo::exec::field_visit(l, k_simd_gran, [sp, in](std::size_t base, std::size_t cnt) {
            if constexpr (std::is_same_v<T, double>) {
                math::sin_batch(sp + base, in + base, cnt);
            } else {
                std::size_t const end = base + cnt;
                for (std::size_t i = base; i < end; ++i) {
                    sp[i] = std::sin(in[i]);
                }
            }
        });
    }

    // force(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - alpha sin(phi(x)).
    // sin(phi) was staged into scratch by `prep`.
    [[nodiscard]] auto force_kernel() const noexcept {
        return [k = kappa, alp = alpha, sp = this->scratch_.data()](std::size_t i, T phi, T nbrs) {
            return formula::sine_gordon_force_site<T>(phi, nbrs, sp[i], k, alp);
        };
    }

    // Per-site math in `T`, volume sum accumulated in (and returned as) `double`.
    // The -alpha*cos term is a Sleef-batched cos_sum on f64; the hopping+mass
    // goes through the shared formula (cos folded to 0 there).
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        T const k   = kappa;
        T const alp = alpha;
        double s{};
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            this->ensure_scratch(n);
            T* const cs       = this->scratch_.data();
            T const* const in = l.data();
            // Fused cos + Σcos in one deterministic reduce: each chunk cos-batches
            // its sub-range into scratch and folds it (fixed partition → thread-
            // invariant; one-time bit re-baseline vs the old single running sum).
            double const cos_sum = reticolo::exec::field_reduce(
                l, k_simd_gran, [cs, in](std::size_t base, std::size_t cnt) {
                    math::cos_batch(cs + base, in + base, cnt);
                    return reticolo::exec::lane_sum8(
                        base, cnt, [cs](std::size_t i) { return static_cast<double>(cs[i]); });
                });
            double const hopping = sweep::reduce_fwd<T, double>(l, [k, alp](T phi, T fwd_sum) {
                return static_cast<double>(
                    formula::sine_gordon_action_site<T>(phi, fwd_sum, T{0}, k, alp));
            });
            s                    = hopping - (static_cast<double>(alp) * cos_sum);
        } else {
            s = sweep::reduce_fwd<T, double>(l, [k, alp](T phi, T fwd_sum) {
                return static_cast<double>(
                    formula::sine_gordon_action_site<T>(phi, fwd_sum, std::cos(phi), k, alp));
            });
        }
        this->last_s_full_ = s;
        return s;
    }
};

}  // namespace reticolo::action
