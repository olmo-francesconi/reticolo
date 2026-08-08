#pragma once

#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

// Per-slab random field fills — the randomness primitive every updater shares.
//
// Both fills bind slab i of the field's canonical partition to site stream i of
// the caller's StreamSet, one-to-one, BY PARTITION INDEX and never by the
// executing thread or the schedule (exec::field_visit_indexed). That binding is
// the reproducibility contract: for a fixed (n_threads, slabs_per_thread) the
// draw sequence of every slab is a pure function of the lattice shape, so a chain
// replays bit-for-bit. The owner is responsible for checking the partition still
// matches the stream count before calling (see `updater::check_streams`).
//
// `fill_gaussian` is HMC's momentum sampling and Metropolis's proposal noise —
// the same operation, so it lives here once rather than twice.

namespace reticolo::updater {

// Compile-time tag: Field is a MatrixLinkLattice (carries a Group type alias).
template <class Field>
concept MatrixLinkField = requires { typename Field::group_type; };

namespace impl {

// Per-slab standard-normal fill narrowed to float. Draws doubles (so the stream
// advances exactly as the double path would) into a thread_local scratch, then
// narrows: single-precision fields still consume the double draw sequence, so a
// float run and a double run sample the same ensemble.
template <class R>
inline void fill_narrowed_slab(R& r, float* out, std::size_t n) {
    thread_local std::vector<double> scratch;
    double* const s = reticolo::exec::thread_scratch(scratch, n);
    r.normal_fill(s, n);
    for (std::size_t j = 0; j < n; ++j) {
        out[j] = static_cast<float>(s[j]);
    }
}

}  // namespace impl

// Fill `f` with independent standard normals, one per real degree of freedom.
// Field-type mapping:
//   - double:                 per-slab batched normal_fill from stream i
//   - std::complex<double>:   reinterpret-cast to double[2], per-slab fill;
//                             independent N(0,1) on Re and Im — the correct
//                             complex Gaussian (§29.5.4 guarantees layout)
//   - float / complex<float>: per-slab double fill narrowed to float
//   - matrix links:           Group::sample_algebra_slab per (slab, μ), so the
//                             anti-hermitian structural zeros stay zero (a raw
//                             normal_fill would put noise on constrained slots)
//   - anything else:          per-element stream_i.normal()
template <class Field, class R>
void fill_gaussian(Field& f, StreamSet<R>& streams) {
    using Scalar = Field::value_type;
    if constexpr (MatrixLinkField<Field>) {
        // Slab i of every direction draws from site stream i, in μ order — the
        // draw sequence is a pure function of the frozen partition.
        using Group            = Field::group_type;
        std::size_t const d    = f.ndims();
        std::size_t const span = f.link_span();  // padded component stride
        for (std::size_t mu = 0; mu < d; ++mu) {
            Scalar* const pblk = f.mu_block_data(mu);
            exec::field_visit_indexed(
                f, 1, [&streams, pblk, span](std::size_t i, std::size_t base, std::size_t cnt) {
                    Group::sample_algebra_slab(pblk + base, streams.site_stream(i), span, cnt);
                });
        }
    } else if constexpr (std::is_same_v<Scalar, double>) {
        double* const m = f.data();
        exec::field_visit_indexed(
            f, 1, [&streams, m](std::size_t i, std::size_t base, std::size_t cnt) {
                streams.site_stream(i).normal_fill(m + base, cnt);
            });
    } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
        // Two doubles per site, so slab i covers doubles [2·base, 2·(base+cnt)).
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* const md = reinterpret_cast<double*>(f.data());
        exec::field_visit_indexed(
            f, 1, [&streams, md](std::size_t i, std::size_t base, std::size_t cnt) {
                streams.site_stream(i).normal_fill(md + (2 * base), 2 * cnt);
            });
    } else if constexpr (std::is_same_v<Scalar, float>) {
        float* const m = f.data();
        exec::field_visit_indexed(
            f, 1, [&streams, m](std::size_t i, std::size_t base, std::size_t cnt) {
                impl::fill_narrowed_slab(streams.site_stream(i), m + base, cnt);
            });
    } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* const mf = reinterpret_cast<float*>(f.data());
        exec::field_visit_indexed(
            f, 1, [&streams, mf](std::size_t i, std::size_t base, std::size_t cnt) {
                impl::fill_narrowed_slab(streams.site_stream(i), mf + (2 * base), 2 * cnt);
            });
    } else {
        Scalar* const m = f.data();
        exec::field_visit_indexed(
            f, 1, [&streams, m](std::size_t i, std::size_t base, std::size_t cnt) {
                R& r = streams.site_stream(i);
                for (std::size_t j = base; j < base + cnt; ++j) {
                    m[j] = static_cast<Scalar>(r.normal());
                }
            });
    }
}

// Fill `f` with log(u), u ~ U(0,1] — the accept thresholds of a local sweep,
// which tests `−ΔS ≥ log u`. Batching the log through Sleef keeps the
// transcendental off the scalar accept path, exactly as the Box-Muller fill does
// for the momentum draw.
//
// Two passes for a double field: draw the slab's uniforms straight into the
// destination (the family's `uniform_fill` keeps the generator state in registers
// across the loop and clamps away from 0, so `log` never sees −inf), then batch
// the log IN PLACE. A float field pays a third pass to narrow. Note the clamp is
// only there to keep the Sleef contract — a u of 0 would accept unconditionally,
// which is what `u < exp(−ΔS)` does anyway.
template <class Field, class R>
void fill_log_uniform(Field& f, StreamSet<R>& streams) {
    using Scalar = Field::value_type;
    static_assert(std::is_floating_point_v<Scalar>,
                  "fill_log_uniform is for real-valued accept-threshold fields");
    Scalar* const m = f.data();
    exec::field_visit_indexed(
        f, 1, [&streams, m](std::size_t i, std::size_t base, std::size_t cnt) {
            R& r = streams.site_stream(i);
            if constexpr (std::is_same_v<Scalar, double>) {
                r.uniform_fill(m + base, cnt);
                math::log_batch(m + base, m + base, cnt);
            } else {
                thread_local std::vector<double> scratch;
                double* const u = reticolo::exec::thread_scratch(scratch, cnt);
                r.uniform_fill(u, cnt);
                math::log_batch(u, u, cnt);
                for (std::size_t j = 0; j < cnt; ++j) {
                    m[base + j] = static_cast<Scalar>(u[j]);
                }
            }
        });
}

}  // namespace reticolo::updater
