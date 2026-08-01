#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _OPENMP
    #include <omp.h>
#endif

using reticolo::Lattice;
using reticolo::exec::better_slabs;
using reticolo::exec::partition;
using reticolo::exec::Partition;
using reticolo::exec::slab_scope;

namespace {

// The canonical shapes every per-site pass slices: one per dimensionality.
std::vector<Lattice<double>::SizeVec> shapes() {
    return {{64, 64}, {24, 24, 24}, {8, 8, 8, 8}, {12, 6, 4, 2}};
}

// Item i covers [i·item_sites, …) — replay field_visit_indexed's own mapping so
// the test pins the contract the consumers actually see, not just the struct.
struct Span {
    std::size_t base;
    std::size_t cnt;
};

std::vector<Span> spans(Partition const& p) {
    std::vector<Span> out;
    out.reserve(p.n_items);
    for (std::size_t i = 0; i < p.n_items; ++i) {
        std::size_t const base = i * p.item_sites;
        out.push_back({base, base < p.n_sites ? std::min(p.item_sites, p.n_sites - base) : 0});
    }
    return out;
}

}  // namespace

// The partition is the shared decomposition every op rides, so its geometry must
// close exactly: for d > 1 the tiling is exact (item_sites = n / n_items), dim 0
// is never split, and the item count is the product of the blocked split dim and
// every dim above it.
TEST_CASE("partition tiles a multi-dim field exactly", "[exec][partition]") {
    for (auto const& shape : shapes()) {
        for (int q : {1, 2, 4}) {
            slab_scope const sl{q};
            Lattice<double> const f{shape};
            Partition const p = partition(f);
            INFO("ndims=" << shape.size() << " slabs_per_thread=" << q);

            REQUIRE(p.n_sites == f.nsites());
            REQUIRE(p.n_items >= 1);
            REQUIRE(p.n_items * p.item_sites == p.n_sites);
            REQUIRE(p.split_dim >= 1);
            REQUIRE(p.split_dim < shape.size());
            REQUIRE(p.block >= 1);
            REQUIRE(shape[p.split_dim] % p.block == 0);

            std::size_t above = 1;
            for (std::size_t k = p.split_dim + 1; k < shape.size(); ++k) {
                above *= shape[k];
            }
            REQUIRE(p.n_items == (shape[p.split_dim] / p.block) * above);
        }
    }
}

// Every site belongs to exactly one item, items are contiguous and in ascending
// order, and none is empty — the property that makes a write-disjoint map
// bit-identical for any slab count.
TEST_CASE("partition items cover the site range once, contiguously", "[exec][partition]") {
    for (auto const& shape : shapes()) {
        for (int q : {1, 2, 3, 4}) {
            slab_scope const sl{q};
            Lattice<double> const f{shape};
            Partition const p = partition(f);
            auto const sp     = spans(p);
            INFO("ndims=" << shape.size() << " slabs_per_thread=" << q);

            REQUIRE(sp.front().base == 0);
            std::size_t total = 0;
            for (std::size_t i = 0; i < sp.size(); ++i) {
                INFO("item=" << i);
                REQUIRE(sp[i].cnt > 0);
                REQUIRE(sp[i].base + sp[i].cnt <= p.n_sites);
                if (i > 0) {
                    REQUIRE(sp[i].base == sp[i - 1].base + sp[i - 1].cnt);
                }
                total += sp[i].cnt;
            }
            REQUIRE(total == p.n_sites);
        }
    }
}

// 1D falls back to flat chunks — the ONE case where the final item is allowed to
// be ragged. It must still cover the range without an empty or out-of-range item.
TEST_CASE("1D partition covers the range with flat chunks", "[exec][partition]") {
    for (std::size_t n : {std::size_t{1024}, std::size_t{1000}, std::size_t{10}}) {
        for (int q : {1, 2, 3, 4, 7}) {
            slab_scope const sl{q};
            Lattice<double> const f{{n}};
            Partition const p = partition(f);
            auto const sp     = spans(p);
            INFO("n=" << n << " slabs_per_thread=" << q << " n_items=" << p.n_items);

            REQUIRE(p.n_sites == n);
            REQUIRE(p.n_items >= 1);
            std::size_t total = 0;
            for (std::size_t i = 0; i < sp.size(); ++i) {
                INFO("item=" << i << " base=" << sp[i].base);
                REQUIRE(sp[i].base <= p.n_sites);
                REQUIRE(sp[i].base + sp[i].cnt <= p.n_sites);
                total += sp[i].cnt;
            }
            REQUIRE(total == p.n_sites);
        }
    }
}

// Raising slabs_per_thread splits finer: the slab count never drops when the
// target rises. (Determinism of a reduce is tied to the count, so a
// non-monotone count would make the knob unpredictable.)
TEST_CASE("slab count is monotone in slabs_per_thread", "[exec][partition]") {
    for (auto const& shape : shapes()) {
        Lattice<double> const f{shape};
        std::size_t prev = 0;
        for (int q : {1, 2, 4, 8}) {
            slab_scope const sl{q};
            std::size_t const items = partition(f).n_items;
            INFO("ndims=" << shape.size() << " slabs_per_thread=" << q << " items=" << items);
            REQUIRE(items >= prev);
            prev = items;
        }
    }
}

// The occupancy heuristic: a count that divides the team evenly wins outright;
// otherwise the nearest to target, ties rounding up.
TEST_CASE("better_slabs prefers even team occupancy, then nearest target", "[exec][partition]") {
    REQUIRE(better_slabs(3, 0, 8, 4));  // any candidate beats "none yet"

    REQUIRE(better_slabs(8, 7, 7, 4));       // 8 divides the team, 7 does not
    REQUIRE_FALSE(better_slabs(7, 8, 7, 4));  // exact target loses to even occupancy

    REQUIRE(better_slabs(6, 10, 5, 3));        // both multiples of 3? no — 6 is, 10 is not
    REQUIRE(better_slabs(6, 4, 5, 2));         // both even: |6-5| == |4-5|, tie rounds up
    REQUIRE_FALSE(better_slabs(4, 6, 5, 2));   // the same tie, seen from the other side
    REQUIRE(better_slabs(4, 8, 5, 2));         // both even, 4 is nearer to 5 than 8
}

#ifdef _OPENMP
// The search only accepts counts ≥ team so every thread gets work; when the
// lattice cannot supply that many slabs it falls back to the finest outer
// tiling (one item per dim-0 row) rather than under-splitting.
TEST_CASE("slab count meets the team, or falls back to the finest tiling",
          "[exec][partition]") {
    int const saved = omp_get_max_threads();

    for (int team : {1, 2, 4, 8}) {
        omp_set_num_threads(team);
        slab_scope const sl{1};
        for (auto const& shape : shapes()) {
            Lattice<double> const f{shape};
            Partition const p = partition(f);
            INFO("team=" << team << " ndims=" << shape.size() << " items=" << p.n_items);
            REQUIRE(p.n_items >= static_cast<std::size_t>(team));
        }
    }

    // 8x2: the only achievable outer splits are 1 and 2 items, so a team of 4
    // cannot be met — the fallback is n / shape[0] = 2.
    omp_set_num_threads(4);
    {
        slab_scope const sl{1};
        Lattice<double> const small{{8, 2}};
        Partition const p = partition(small);
        REQUIRE(p.n_items == small.nsites() / small.shape()[0]);
        REQUIRE(p.n_items * p.item_sites == p.n_sites);
    }

    omp_set_num_threads(saved);
}
#endif
