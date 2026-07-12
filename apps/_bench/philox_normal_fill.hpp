#pragma once

#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/rng/philox.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace reticolo::bench {

// Parallel counter-based standard-normal fill over a field's canonical
// partition. out[2p], out[2p+1] are the Box-Muller pair of
// philox_uniform2(key, 0, p) — each pair an independent function of the site
// index, so the fill worksplits and is bit-identical for any thread count.
// Bench infrastructure: this was Hmc's momentum sampler before the owned
// per-slab StreamSet replaced it; it lives on here as the keyed reference
// fill for the component benches and as the CPU mirror of the CUDA momentum
// sampler.
//
// Every stage is SIMD-batched: the Philox counter core through
// philox_uniform2_batch (bit-identical integers, vectorised across pairs), the
// transcendentals through Sleef log/sincos, the clamp/scale/sqrt through plain
// vector loops. Pairs are partitioned by the CANONICAL field partition (scaled
// sites → flat doubles → pairs), so momentum keeps the same thread↔slab
// ownership as every other pass; a pair straddling an item boundary (odd item
// size) belongs to the item holding its first double — a fixed shape-only rule,
// so the split — and therefore every bit — matches between serial and any-thread
// runs on a machine.
//
// `n` is the flat double count (2·nsites for a complex field, nsites for real);
// `fld` supplies the partition and nsites.
template <class Field>
inline void
philox_normal_fill(double* out, std::size_t n, std::uint64_t key, Field const& fld) noexcept {
    std::size_t const npair = n / 2;
    std::size_t const scale = n / fld.nsites();  // flat doubles per site (1 or 2)
    reticolo::exec::field_visit(
        fld, 16, [out, key, npair, scale](std::size_t sbase, std::size_t scnt) {
            std::size_t const dlo  = sbase * scale;
            std::size_t const dhi  = dlo + (scnt * scale);
            std::size_t const plo  = (dlo + 1) / 2;
            std::size_t const phi  = std::min((dhi + 1) / 2, npair);
            std::size_t const base = plo;
            std::size_t const cnt  = phi - plo;
            if (cnt == 0) {
                return;
            }
            // Thread-local staging: lg | th | sn (3·cnt doubles).
            thread_local std::vector<double> stage;
            double* const lg          = reticolo::exec::thread_scratch(stage, 3 * cnt);
            double* const th          = lg + cnt;
            double* const sn          = th + cnt;
            constexpr double k_two_pi = 2.0 * std::numbers::pi;
            philox_uniform2_batch(key, 0, base, cnt, lg, th);
            for (std::size_t k = 0; k < cnt; ++k) {
                lg[k] = lg[k] > 1.0e-300 ? lg[k] : 1.0e-300;
                th[k] = k_two_pi * th[k];
            }
            math::log_batch(lg, lg, cnt);
            math::sincos_batch(sn, th, th, cnt);  // sn = sin, th = cos (in place)
            for (std::size_t k = 0; k < cnt; ++k) {
                lg[k] = std::sqrt(-2.0 * lg[k]);  // r, vector sqrt
            }
            for (std::size_t k = 0; k < cnt; ++k) {
                out[2 * (base + k)]       = lg[k] * th[k];
                out[(2 * (base + k)) + 1] = lg[k] * sn[k];
            }
        });
    if ((n & 1U) != 0U) {  // odd tail
        double a = 0.0;
        double b = 0.0;
        philox_normal2(key, 0, npair, a, b);
        out[n - 1] = a;
    }
}

}  // namespace reticolo::bench
