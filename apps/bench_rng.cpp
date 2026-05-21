// bench_rng — momentum-sampling throughput per field type × RNG.
//
// For each (field type, RNG) tuple, measure the cost of filling a buffer
// the size of one trajectory's worth of momenta. Three RNGs (FastRng,
// RanluxRng, Mt19937Rng) all satisfy the same `Rng` concept and plug in
// at every existing call site.

#include <reticolo/reticolo.hpp>

#include "_bench/timing.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using reticolo::bench::time_per_call;

constexpr int k_sigma_n = 3;

void print_header() {
    std::printf(
        "%-30s %-12s %-12s %-12s %-14s\n", "field", "rng", "doubles", "wall [s]", "doubles/s");
    std::printf(
        "%-30s %-12s %-12s %-12s %-14s\n", "-----", "---", "-------", "--------", "---------");
}

void print_row(char const* field, char const* rng, std::size_t doubles, double wall_s) {
    double const per_s = static_cast<double>(doubles) / wall_s;
    std::printf("%-30s %-12s %-12zu %-12.3e %-12.3e\n", field, rng, doubles, wall_s, per_s);
}

// Field-agnostic batched fill: same path the HMC sample_momenta_ takes for
// the field type. Each lambda calls the *right* batched API for the field:
//  Lattice<double>           → normal_fill
//  Lattice<complex<double>>  → normal_fill on reinterpret_cast<double*>
//  Lattice<array<double,N>>  → per-element rng.normal()
//  LinkLattice<double>       → normal_fill
//  MatrixLinkLattice<G,T>    → G::sample_algebra_slab per direction

template <class Rng>
void bench_rng_for(char const* rng_name) {
    using namespace reticolo;

    Rng rng{2024};

    // Lattice<double>  at 3D L=24 and 4D L=8
    for (auto const& shape :
         std::array<std::vector<std::size_t>, 2>{{{24, 24, 24}, {8, 8, 8, 8}}}) {
        std::size_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        std::vector<double> buf(n);
        char field_name[64];
        std::snprintf(field_name,
                      sizeof(field_name),
                      "Lattice<double> %dD L=%zu",
                      static_cast<int>(shape.size()),
                      shape[0]);
        double const t = time_per_call([&] { rng.normal_fill(buf.data(), buf.size()); });
        print_row(field_name, rng_name, buf.size(), t);
    }

    // Lattice<complex<double>>  at 3D L=24 and 4D L=8
    for (auto const& shape :
         std::array<std::vector<std::size_t>, 2>{{{24, 24, 24}, {8, 8, 8, 8}}}) {
        std::size_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        std::vector<std::complex<double>> buf(n);
        char field_name[64];
        std::snprintf(field_name,
                      sizeof(field_name),
                      "Lattice<complex> %dD L=%zu",
                      static_cast<int>(shape.size()),
                      shape[0]);
        double const t = time_per_call(
            [&] { rng.normal_fill(reinterpret_cast<double*>(buf.data()), 2 * buf.size()); });
        print_row(field_name, rng_name, 2 * buf.size(), t);
    }

    // Lattice<array<double,3>>  at 3D L=16 (OnSigma path)
    {
        std::vector<std::array<double, k_sigma_n>> buf(16ULL * 16ULL * 16ULL);
        char field_name[64];
        std::snprintf(field_name, sizeof(field_name), "Lattice<array<3>> 3D L=16");
        double const t = time_per_call([&] {
            for (auto& v : buf) {
                for (auto& x : v) {
                    x = rng.normal();
                }
            }
        });
        print_row(field_name, rng_name, k_sigma_n * buf.size(), t);
    }

    // LinkLattice<double> at 3D L=24 and 4D L=8
    for (auto const& shape :
         std::array<std::vector<std::size_t>, 2>{{{24, 24, 24}, {8, 8, 8, 8}}}) {
        std::size_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        std::size_t const nlinks = shape.size() * n;
        std::vector<double> buf(nlinks);
        char field_name[64];
        std::snprintf(field_name,
                      sizeof(field_name),
                      "LinkLattice<double> %dD L=%zu",
                      static_cast<int>(shape.size()),
                      shape[0]);
        double const t = time_per_call([&] { rng.normal_fill(buf.data(), buf.size()); });
        print_row(field_name, rng_name, buf.size(), t);
    }

    // MatrixLinkLattice<SU2, double> 4D L=8: per-direction sample_algebra_slab
    {
        MatrixLinkLattice<gauge_group::SU2, double>::SizeVec shape{8, 8, 8, 8};
        MatrixLinkLattice<gauge_group::SU2, double> mom{shape};
        std::size_t const ns      = mom.nsites();
        std::size_t const doubles = mom.ndims() * gauge_group::SU2::n_real_components * ns;
        double const t            = time_per_call([&] {
            for (std::size_t mu = 0; mu < mom.ndims(); ++mu) {
                math::su2::sample_algebra_slab(mom.mu_block_data(mu), rng, ns);
            }
        });
        print_row("MatrixLinkLattice<SU2> 4D L=8", rng_name, doubles, t);
    }

    // MatrixLinkLattice<SU3, double> 4D L=8
    {
        MatrixLinkLattice<gauge_group::SU3, double>::SizeVec shape{8, 8, 8, 8};
        MatrixLinkLattice<gauge_group::SU3, double> mom{shape};
        std::size_t const ns      = mom.nsites();
        std::size_t const doubles = mom.ndims() * gauge_group::SU3::n_real_components * ns;
        double const t            = time_per_call([&] {
            for (std::size_t mu = 0; mu < mom.ndims(); ++mu) {
                math::su3::sample_algebra_slab(mom.mu_block_data(mu), rng, ns);
            }
        });
        print_row("MatrixLinkLattice<SU3> 4D L=8", rng_name, doubles, t);
    }
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();
    std::printf("RNG — momentum-sampling throughput\n\n");
    print_header();
    bench_rng_for<FastRng>("FastRng");
    bench_rng_for<RanluxRng>("Ranlux48");
    bench_rng_for<Mt19937Rng>("Mt19937_64");
}
