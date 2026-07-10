#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"
#include "_bench/philox_normal_fill.hpp"
#include "_bench/timing.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// Per-component HMC breakdown: time each pass of ONE trajectory in isolation
// (momentum refresh, rollback snapshot, kinetic, s_full, fused force+kick,
// drift) for one action + shape + ambient OMP_NUM_THREADS, and report an
// effective memory throughput per pass. Each pass is reconstructed from the
// SAME public primitives alg::Hmc uses, so the numbers reflect the real hot
// loop. A STREAM-triad probe gives the machine's achievable bandwidth ceiling.
//
// Effective bandwidth = (compulsory bytes) / time, counting each array the pass
// streams once (neighbour re-reads hit cache). coeff = arrays streamed:
//   refresh 1 (write p) · snapshot 2 (read φ, write φ₀) · kinetic 1 (read p)
//   s_full 1 (read φ) · kick 3 (read φ, rmw p) · drift 3 (rmw φ, read p).
// GB/s = coeff·V·bytes_per_site / t / 1e9. Phi4 lands near the STREAM ceiling
// (bandwidth-bound); SU3 reads far below it (compute-bound) — one metric splits
// the regimes. Output is CSV to stdout (driver-parsed) + a table to stderr.

using namespace reticolo;
using reticolo::bench::consume;
using reticolo::bench::time_per_call;

template <class F>
concept GaugeField = requires { typename F::group_type; };

// argv[1]: cube edge ("16" → 16⁴) or explicit "AxBxCxD".
std::vector<std::size_t> parse_shape(char const* s) {
    std::vector<std::size_t> dims;
    std::size_t v = 0;
    bool any      = false;
    for (char const* p = s;; ++p) {
        if (*p >= '0' && *p <= '9') {
            v   = v * 10 + static_cast<std::size_t>(*p - '0');
            any = true;
        } else {
            if (any) {
                dims.push_back(v);
            }
            v   = 0;
            any = false;
            if (*p == 0) {
                break;
            }
        }
    }
    if (dims.size() == 1) {
        return std::vector<std::size_t>(4, dims[0]);
    }
    return dims;
}

template <class Field>
void refresh_mom(Field& mom, FastRng& rng) {
    if constexpr (GaugeField<Field>) {
        using G                 = typename Field::group_type;
        std::size_t const d     = mom.ndims();
        std::size_t const ns    = mom.nsites();
        std::uint64_t const key = rng.uniform_u64();
        for (std::size_t mu = 0; mu < d; ++mu) {
            auto* const pblk = mom.mu_block_data(mu);
            exec::field_visit(mom, 1, [pblk, key, mu, ns](std::size_t base, std::size_t cnt) {
                G::sample_algebra_philox_range(pblk, key, mu, ns, base, cnt);
            });
        }
    } else {
        bench::philox_normal_fill(mom.data(), flat_size(mom), rng.uniform_u64(), mom);
    }
}

template <class Field>
double kinetic_e(Field const& mom) {
    if constexpr (GaugeField<Field>) {
        using G              = typename Field::group_type;
        std::size_t const d  = mom.ndims();
        std::size_t const ns = mom.nsites();
        double raw           = 0.0;
        for (std::size_t mu = 0; mu < d; ++mu) {
            auto const* const pblk = mom.mu_block_data(mu);
            raw += exec::field_reduce(mom, 8, [pblk, ns](std::size_t base, std::size_t cnt) {
                return G::kinetic_range(pblk, ns, base, cnt);
            });
        }
        return 0.5 * raw;
    } else {
        auto const* const p = mom.data();
        double const sum    = exec::field_reduce(mom, 1, [p](std::size_t base, std::size_t cnt) {
            return exec::lane_sum8(base, cnt, [p](std::size_t i) {
                double const pi = p[i];
                return pi * pi;
            });
        });
        return 0.5 * sum;
    }
}

// STREAM triad a=b+s·c over a cache-busting buffer, threaded through the SAME
// parallel path as the field passes → the achievable aggregate BW at this thread
// count. 3 arrays moved (2 read, 1 write). The working set MUST exceed aggregate
// L3 or this measures cache, not DRAM, bandwidth — a cache-resident STREAM scales
// linearly with cores past any real DRAM limit (>1 TB/s), which is meaningless as
// a ceiling. Default 1 GB/array (3 GB set) > any current server L3; shrink via
// RETICOLO_STREAM_MB on constrained hosts.
double stream_triad_gbps() {
    char const* const env = std::getenv("RETICOLO_STREAM_MB");
    std::size_t const mb  = env != nullptr ? static_cast<std::size_t>(std::atoi(env)) : 1024;
    std::size_t const n   = (mb * 1024UL * 1024UL) / sizeof(double);
    static std::vector<double> a(n, 1.0);
    static std::vector<double> b(n, 2.0);
    static std::vector<double> c(n, 3.0);
    double const s         = 0.42;
    double* const ap       = a.data();
    double const* const bp = b.data();
    double const* const cp = c.data();
    double const t         = time_per_call([&] {
        exec::parallel_map_ranges(
            n, sizeof(double), 8, [ap, bp, cp, s](std::size_t base, std::size_t cnt) {
                std::size_t const end = base + cnt;
                for (std::size_t i = base; i < end; ++i) {
                    ap[i] = bp[i] + s * cp[i];
                }
            });
        consume(a[0]);
    });
    return 3.0 * static_cast<double>(n) * sizeof(double) / t / 1e9;
}

template <class Action, class Field>
void run_action(char const* name,
                Action const& act,
                Field& field,
                FastRng& rng,
                int n_md,
                char const* shape,
                char const* th) {
    using T = typename Field::value_type;
    Field mom{field.indexing()};
    Field snap{field.indexing()};
    reticolo::bench::hot_init(mom, rng);
    double const V   = static_cast<double>(field.nsites());
    double const b   = static_cast<double>(field.bytes_per_site());
    double const dt  = 1.0 / static_cast<double>(n_md);
    double const hdt = 0.5 * dt;

    auto emit = [&](char const* pass, double t, double coeff, int calls) {
        double const ms   = t * 1e3;
        double const gbps = coeff * V * b / t / 1e9;
        std::printf("%s %s %s %-8s %.6f %.3f %d\n", name, shape, th, pass, ms, gbps, calls);
        std::fprintf(stderr, "  %-8s %10.5f ms  %8.2f GB/s  ×%d\n", pass, ms, gbps, calls);
    };

    std::fprintf(stderr, "== %s  %s  threads=%s  n_md=%d ==\n", name, shape, th, n_md);
    {  // refresh momenta
        double const t = time_per_call([&] {
            refresh_mom(mom, rng);
            consume(mom.data()[0]);
        });
        emit("refresh", t, 1.0, 1);
    }
    {  // rollback snapshot (flat copy)
        std::size_t const nf = flat_size(field);
        T* const dst         = snap.data();
        T const* const src   = field.data();
        double const t       = time_per_call([&] {
            exec::parallel_map_ranges(
                nf, sizeof(T), 1, [dst, src](std::size_t base, std::size_t cnt) {
                    std::size_t const end = base + cnt;
                    for (std::size_t i = base; i < end; ++i) {
                        dst[i] = src[i];
                    }
                });
            consume(dst[0]);
        });
        emit("snapshot", t, 2.0, 1);
    }
    {  // kinetic reduce
        double const t = time_per_call([&] { consume(kinetic_e(mom)); });
        emit("kinetic", t, 1.0, 2);
    }
    {  // action
        double const t = time_per_call([&] { consume(act.s_full(field)); });
        emit("s_full", t, 1.0, 2);
    }
    {  // fused force+kick (the real MD kick; force never materialised)
        double const t = time_per_call([&] {
            act.compute_force_and_kick(field, mom, static_cast<T>(hdt));
            consume(mom.data()[0]);
        });
        emit("kick", t, 3.0, n_md + 1);
    }
    {  // drift
        double const t = time_per_call([&] {
            alg::integ::drift_field(field, mom, dt);
            consume(field.data()[0]);
        });
        emit("drift", t, 3.0, n_md);
    }
}

// Usage: bench_hmc_breakdown <shape> [action=both|phi4|su3] [n_md].
int main(int argc, char** argv) {
    reticolo::log::off();
    char const* const shape_arg = argc > 1 ? argv[1] : "16";
    std::string_view const act  = argc > 2 ? argv[2] : "both";
    int const n_md              = argc > 3 ? std::atoi(argv[3]) : 8;
    char const* th              = std::getenv("OMP_NUM_THREADS");
    if (th == nullptr) {
        th = "1";
    }
    std::vector<std::size_t> const shape = parse_shape(shape_arg);
    FastRng rng{42};

    double const ceil_gbps = stream_triad_gbps();
    std::printf("machine %s %s stream    0.000000 %.3f 0\n", shape_arg, th, ceil_gbps);
    std::fprintf(stderr, "  STREAM triad ceiling: %.2f GB/s\n", ceil_gbps);

    bool const phi = (act == "phi4" || act == "both");
    bool const su3 = (act == "su3" || act == "both");
    if (phi) {
        Lattice<double> f{shape};
        reticolo::bench::hot_init(f, rng);
        run_action(
            "Phi4", act::Phi4<double>{.kappa = 0.18, .lambda = 1.0}, f, rng, n_md, shape_arg, th);
    }
    if (su3) {
        using F = MatrixLinkLattice<math::group::SU3, double>;
        F u{shape};
        reticolo::bench::hot_init(u, rng);
        run_action("Wilson<SU3>",
                   act::Wilson<math::group::SU3, double>{.beta = 6.0},
                   u,
                   rng,
                   n_md,
                   shape_arg,
                   th);
    }
    return 0;
}
