# Action concepts

Actions in reticolo are plain structs that satisfy one or more C++23
concepts defined in
[`<reticolo/action/concepts.hpp>`](../include/reticolo/action/concepts.hpp).

There is no base class, no virtual method, no `register_action` macro. An
action just *has* the right member functions; the updater concept-checks at
the call site and picks the right code path.

## The concept lattice

```
                           LocalAction          (Metropolis baseline)
                                │
            ┌───────────────────┼──────────────────────────┐
            │                   │                          │
         HasSEff             HasForce                 HasProposal
            │                   │
            └─────────┬─────────┘
                      │
                    Hmc (needs both)             WolffEmbeddable
                                                       │
                                                     Wolff
```

Each refinement adds **one member function** to the previous interface.
Updaters take a class template parameter `A` and constrain it with whichever
concepts they need. Adding a new updater is "constrain A by some subset of
the concepts, write the loop" — no plumbing changes.

## `LocalAction` — minimum interface

The baseline an updater needs to do a Metropolis sweep:

```cpp
template <class A, class F>
concept LocalAction = requires(A const& a, Lattice<F> const& l, Site x, F nv) {
    { a.s_local(l, x)        } -> std::convertible_to<double>;
    { a.ds_local(l, x, nv)   } -> std::convertible_to<double>;
};
```

- `s_local(l, x)` — the contribution to S that touches site x.
- `ds_local(l, x, new_v)` — change in `s_local` if `phi(x)` is replaced by
  `new_v`. The Metropolis sweep accepts with `min(1, exp(-ds_local))`.

A `LocalAction` is enough for `alg::Metropolis`.

## `HasSEff` — full action total

```cpp
{ a.s_full(l) } -> std::convertible_to<double>;
```

The total action S. Needed by HMC (for ΔH) and by any diagnostic series
that wants S itself.

## `HasForce` — molecular-dynamics force

```cpp
void compute_force(Lattice<F> const& l, Lattice<F>& force) const noexcept;
```

Write `-dS/dphi` into `force`. The HMC integrator calls this once per MD
step. The force buffer is allocated and owned by the `Hmc` object and reuses
the field's `Indexing` (sibling lattice — no extra neighbour table).

`HasForce + HasSEff + Rng` is what `alg::Hmc` requires.

## `HasProposal` — action-supplied Metropolis proposal

```cpp
template <class R>
F propose(Lattice<F> const&, Site, R& rng) const noexcept;
```

If present, `alg::Metropolis` uses it; otherwise it falls back to a Gaussian
random walk of width `sigma` around the current `phi(x)`. The selection
happens at instantiation via `if constexpr`, no virtual dispatch.

`OnSigma<N>` ships a `propose` that draws a uniform-on-sphere vector
(Marsaglia normalised-Gaussian trick) — for a constrained manifold the
unconstrained Gaussian fallback would be wrong, so `HasProposal` is how the
action carries the manifold knowledge.

## `WolffEmbeddable` — cluster embedding

```cpp
template <class A, class F, class R>
concept WolffEmbeddable = LocalAction<A, F> && Rng<R> &&
    requires(A const& a, F const& v, typename A::axis_type const& axis, R& rng) {
        typename A::axis_type;
        { a.wolff_random_axis(rng)            } -> std::same_as<typename A::axis_type>;
        { a.wolff_reflect(v, axis)            } -> std::same_as<F>;
        { a.wolff_link_p(v, v, axis)          } -> std::convertible_to<double>;
    };
```

Three pieces, all action-specific:

- `axis_type` — the type of one reflection (an angle for XY, a unit vector
  for O(N)).
- `wolff_random_axis(rng)` — draw a random one.
- `wolff_reflect(v, axis)` — the involution R: `v -> v'`.
- `wolff_link_p(v_x, v_y, axis)` — the bond-activation probability between
  two original (pre-flip) field values across that reflection. The cluster
  updater stays totally action-agnostic by deferring this calculation to the
  action — XY uses `1 - exp(min(0, -2β sin(θ_x-r) sin(θ_y-r)))`, OnSigma
  uses `1 - exp(min(0, -2β (r·φ_x)(r·φ_y)))`, etc.

## Worked example: writing a new action

Imagine you want to study `S = -beta sum_<x,y> cos(phi_x - phi_y) +
m^2/2 sum_x phi_x^2` — XY with an explicit symmetry-breaking mass term.
The whole thing satisfies `LocalAction + HasSEff + HasForce`:

```cpp
template <class T = double>
struct XyWithMass {
    using value_type = T;
    T beta = 0;
    T m2   = 0;

    T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const theta = l[x];
        T hop = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) hop += std::cos(theta - l[fwd]);
            if (bwd.is_valid()) hop += std::cos(theta - l[bwd]);
        }
        return -beta * hop + T{0.5} * m2 * theta * theta;
    }

    T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        // ... finite difference, see Phi4 / Xy for the shape ...
    }

    T s_full(Lattice<T> const& l) const noexcept {
        T total = 0;
        for (Site const x : l.sites()) {
            T const theta = l[x];
            total += T{0.5} * m2 * theta * theta;
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                if (fwd.is_valid()) total += -beta * std::cos(theta - l[fwd]);
            }
        }
        return total;
    }

    void compute_force(Lattice<T> const& l, Lattice<T>& f) const noexcept {
        // -dS/dtheta(x); the mass term contributes -m^2 * theta(x).
        // See Xy::compute_force for the hopping piece.
    }
};
```

Drop it in `include/reticolo/action/builtins/xy_with_mass.hpp`, add it to
the umbrella include in `reticolo.hpp`, and it Just Works with both
`alg::Metropolis<XyWithMass<double>, FastRng>` and
`alg::Hmc<XyWithMass<double>, FastRng>`. Nothing else changes — no
registration, no factory, no string switch.

It does **not** satisfy `WolffEmbeddable` (the mass term breaks the
reflection symmetry); calling `alg::Wolff<XyWithMass<double>, FastRng>`
would fail to compile with a concept-check error pointing at the missing
`axis_type` member. The compiler tells you exactly what's wrong.

## Boundary conditions

Every action body guards neighbour accesses with
`if (s.is_valid()) { ... }`. The lattice's `Indexing` returns
`Site{Site::k_invalid_value}` for off-lattice neighbours under open BCs, so
the action stays BC-agnostic — the same `s_local` body works for periodic,
open, or mixed boundaries with no `#ifdef`.

## What about gauge actions?

Out of scope for v3. The concept lattice was deliberately designed so that
*adding* gauge concepts (`HasParallelTransport`, `Gauge`, …) wouldn't
collide with anything; see the [v3 plan
§13–14](../proposals/rewrite_plan_v3.md) for the explicit deferral list.
