#pragma once

#include <reticolo/action/site/detail/site_action.hpp>
#include <reticolo/action/site/detail/traversal.hpp>
#include <reticolo/action/site/formula/sine_gordon_formula.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
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
struct SineGordon : detail::SiteAction<SineGordon<T>, T> {
    using value_type = T;

    T kappa = T{0};
    T alpha = T{0};

    void describe(log::Entry& e) const {
        e.line("SineGordon<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("α={:.3f}", alpha);
    }

    // Fill the shared scratch with sin(phi) per site — Sleef-batched on f64,
    // a scalar loop otherwise. Called by SiteAction before each force pass.
    void prep(Lattice<T> const& l) const noexcept {
        std::size_t const n = l.nsites();
        this->ensure_scratch(n);
        T* const sp       = this->scratch_.data();
        T const* const in = l.data();
        if constexpr (std::is_same_v<T, double>) {
            math::sin_batch(sp, in, n);
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                sp[i] = std::sin(in[i]);
            }
        }
    }

    // force(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - alpha sin(phi(x)).
    // sin(phi) was staged into scratch by `prep`.
    [[nodiscard]] auto force_kernel() const noexcept {
        return [k = kappa, alp = alpha, sp = this->scratch_.data()](std::size_t i, T phi, T nbrs) {
            return detail::sine_gordon_force_site<T>(phi, nbrs, sp[i], k, alp);
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
            math::cos_batch(this->scratch_.data(), l.data(), n);
            T const* const cs = this->scratch_.data();
            double cos_sum    = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                cos_sum += cs[i];
            }
            double const hopping = detail::reduce_fwd<T, double>(l, [k, alp](T phi, T fwd_sum) {
                return static_cast<double>(
                    detail::sine_gordon_action_site<T>(phi, fwd_sum, T{0}, k, alp));
            });
            s                    = hopping - (static_cast<double>(alp) * cos_sum);
        } else {
            s = detail::reduce_fwd<T, double>(l, [k, alp](T phi, T fwd_sum) {
                return static_cast<double>(
                    detail::sine_gordon_action_site<T>(phi, fwd_sum, std::cos(phi), k, alp));
            });
        }
        this->last_s_full_ = s;
        return s;
    }
};

}  // namespace reticolo::action
