#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

// Fused, parallel per-site observable reduction — the observer analogue of the
// action sweep engine, and the ONE way observables are measured. An observable is
// a per-site kernel `value_type -> R`; `obs::reduce` folds a whole PACK of them in
// ONE pass over `exec::field_reduce` (the same canonical-partition, deterministic
// item-order fold the action `reduce_stencil` rides on). The kernels are applied
// back-to-back on the same `self` inside the inner loop, so the compiler keeps φ
// in a register and co-schedules the observables instead of re-streaming the field
// once per observable.
//
// Each lane accumulates in its OWN return type, deduced from the kernel: a real
// field with `double` kernels, a complex field with a complex-valued mean kernel
// next to a `double` amplitude kernel, etc. all wire straight in — pass a lambda
// (or a builtin from `obs::kernel`) and it fuses. `reduce` returns a std::tuple of
// the raw (un-normalised) sums; structured-bind it and hand each sum to a `*_of`
// finalizer / divide by the volume to unpack it into an I/O series.
//
// Determinism matches the action engine: for a fixed (team, slabs-per-thread) the
// fold is reproducible; a different team re-folds.

namespace reticolo::obs {

// Heterogeneous N-lane accumulator: lane i holds the running sum in kernel i's
// return type. `field_reduce` seeds `Lanes{}` per item and folds the per-item
// partials in canonical order via element-wise `operator+=`.
template <class... Ts>
struct Lanes {
    std::tuple<Ts...> v{};

    Lanes& operator+=(Lanes const& o) noexcept {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((std::get<I>(v) += std::get<I>(o.v)), ...);
        }(std::index_sequence_for<Ts...>{});
        return *this;
    }
};

// Run every kernel `ks...` over each site in a single parallel sweep; lane i is
// Σ_x ks_i(φ(x)) accumulated in `ks_i`'s result type. Generic in both the field
// value type and the kernels — a kernel is any callable on the field's
// `value_type` whose result is default-constructible and supports `+=`. Indices
// are compile-time constants so the lanes stay independent accumulators.
//
// Vectorisation notes (checked on the emitted asm — matters for custom kernels):
//   * A `double`-returning lane vectorises fully: the sweep runs 8 sites/iter
//     with several NEON accumulators, same as the action reductions. Keep hot
//     lanes double-valued.
//   * A lane whose result is `std::complex<…>` still folds correctly, but the
//     complex ACCUMULATION scalarises (the compiler lays memory out for the
//     double lanes and reduces the complex sum with scalar adds). If a complex
//     mean sits on a hot path, split it into two double lanes — a Σ(re) kernel
//     and a Σ(im) kernel — and reassemble afterward; those vectorise like any
//     real-valued lane.
//   * NEVER call `std::abs(z)` on a complex `z` inside a kernel: it lowers to a
//     `hypot` library CALL that scalarises the whole loop to 1 site/iter. For an
//     amplitude sum use `std::sqrt(re*re + im*im)` (lowers to `fsqrt.2d`), or
//     accumulate |φ|² and take the sqrt once at the end.
template <class T, class... Ks>
[[nodiscard]] inline auto reduce(Lattice<T> const& l, Ks const&... ks) noexcept {
    static_assert(sizeof...(Ks) > 0, "obs::reduce needs at least one kernel");
    using Acc           = Lanes<std::decay_t<std::invoke_result_t<Ks const&, T const&>>...>;
    T const* const data = l.data();
    auto const kt       = std::forward_as_tuple(ks...);
    Acc const acc       = exec::field_reduce<Acc>(l, 1, [&](std::size_t base, std::size_t cnt) {
        RETICOLO_FP_REASSOCIATE
        Acc s{};
        for (std::size_t j = 0; j < cnt; ++j) {
            T const self = data[base + j];
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((std::get<I>(s.v) += std::get<I>(kt)(self)), ...);
            }(std::index_sequence_for<Ks...>{});
        }
        return s;
    });
    return acc.v;
}

// ---------- builtin per-site kernels -----------------------------------------
//
// Stateless functors, `T -> double`, for the field itself and its low powers.
// Templated on the field type so the same kernel serves f32 and f64 lattices —
// every one casts to double, so measurements are double regardless of field
// precision. Need something else? Pass a lambda straight into `obs::reduce`.
namespace kernel {

struct Phi {
    template <class T>
    [[nodiscard]] double operator()(T self) const noexcept {
        return static_cast<double>(self);
    }
};
struct PhiSq {
    template <class T>
    [[nodiscard]] double operator()(T self) const noexcept {
        auto const v = static_cast<double>(self);
        return v * v;
    }
};
struct PhiQuartic {
    template <class T>
    [[nodiscard]] double operator()(T self) const noexcept {
        auto v = static_cast<double>(self);
        v *= v;
        return v * v;
    }
};

inline constexpr Phi phi{};
inline constexpr PhiSq phi_sq{};
inline constexpr PhiQuartic phi_quartic{};

}  // namespace kernel

// ---------- finalizers: raw sum + nsites -> physics --------------------------
//
// A `reduce` lane is Σ_x kernel(φ(x)); these unpack it into the reported quantity.
// `mean_of` is the generic per-site average (works on any lane — Σφ→<φ>, Σφ²→<φ²>,
// …); the others are the non-linear-in-mean channels that consume the Σφ lane.

[[nodiscard]] inline double mean_of(double sum, double nsites) noexcept {
    return sum / nsites;
}
[[nodiscard]] inline double sq_of_mean_of(double sum1, double nsites) noexcept {
    double const m = sum1 / nsites;
    return m * m;
}
[[nodiscard]] inline double mag_abs_of(double sum1, double nsites) noexcept {
    return std::abs(sum1 / nsites);
}

}  // namespace reticolo::obs
