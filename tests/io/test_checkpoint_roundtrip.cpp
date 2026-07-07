// Round-trip and resume tests for the config-snapshot surface
// (io::Writer::field / io::Writer::rng_state / io::Reader / io::checkpoint).

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/io/checkpoint.hpp>
#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/math/group/u1.hpp>

#include "../test_helpers.hpp"

#include <complex>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

namespace {

template <class L, class F>
void fill_lattice(L& lat, F&& f) {
    auto* d = lat.data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(lat.end() - lat.begin()); ++i) {
        d[i] = f(i);
    }
}

}  // namespace

TEST_CASE("scalar Lattice<double> round-trips through Writer::field / Reader::field",
          "[io][checkpoint][scalar]") {
    test::ScratchH5 scratch{"ckpt_scalar"};

    Lattice<double>::SizeVec const shape{4, 4, 4};
    Lattice<double> phi{shape};
    fill_lattice(phi, [](std::size_t i) { return 0.125 * static_cast<double>(i) - 7.0; });

    {
        io::Writer w{scratch.path()};
        w.field("/field", phi);
    }

    io::Reader r{scratch.path()};
    REQUIRE(r.field_shape("/field") == shape);

    Lattice<double> rt{shape};
    r.field("/field", rt);

    auto const* a = phi.data();
    auto const* b = rt.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        REQUIRE(a[i] == b[i]);
    }
}

TEST_CASE("scalar Lattice<complex<double>> round-trips bit-exact", "[io][checkpoint][complex]") {
    test::ScratchH5 scratch{"ckpt_complex"};

    Lattice<std::complex<double>>::SizeVec const shape{3, 5};
    Lattice<std::complex<double>> phi{shape};
    fill_lattice(phi, [](std::size_t i) {
        return std::complex<double>{0.5 * static_cast<double>(i), -1.0 - static_cast<double>(i)};
    });

    {
        io::Writer w{scratch.path()};
        w.field("/cfg/phi", phi);
    }

    io::Reader r{scratch.path()};
    Lattice<std::complex<double>> rt{shape};
    r.field("/cfg/phi", rt);

    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        REQUIRE(phi.data()[i] == rt.data()[i]);
    }
}

TEST_CASE("FastRng state round-trips through Writer::rng_state / Reader::rng_state",
          "[io][checkpoint][rng]") {
    test::ScratchH5 scratch{"ckpt_rng"};

    FastRng rng{12345ULL};
    for (int k = 0; k < 47; ++k) {
        (void)rng.uniform_u64();
    }
    (void)rng.normal();  // leaves a cached normal in the spare slot.

    auto const state_before  = rng.state();
    auto const cached_before = rng.cached_normal();
    auto const flag_before   = rng.has_cached_normal();

    {
        io::Writer w{scratch.path()};
        w.rng_state("/rng", rng);
    }

    io::Reader r{scratch.path()};
    FastRng restored = r.rng_state("/rng");

    REQUIRE(restored.state() == state_before);
    REQUIRE(restored.cached_normal() == cached_before);
    REQUIRE(restored.has_cached_normal() == flag_before);

    // Draw the same sequence from both — bit-exact divergence-free.
    for (int k = 0; k < 64; ++k) {
        REQUIRE(rng.uniform_u64() == restored.uniform_u64());
    }
}

TEST_CASE("save_config + load_config preserve field, RNG, and trajectory counter",
          "[io][checkpoint][bundle]") {
    test::ScratchH5 scratch{"ckpt_bundle"};

    Lattice<double>::SizeVec const shape{6, 6};
    Lattice<double> phi{shape};
    fill_lattice(phi, [](std::size_t i) { return std::sin(0.31 * static_cast<double>(i)); });
    FastRng rng{99ULL};
    for (int k = 0; k < 100; ++k) {
        (void)rng.uniform();
    }
    long long const traj_in = 4321;

    io::save_config(scratch.path(), phi, rng, traj_in);

    REQUIRE(io::load_field_shape(scratch.path()) == shape);

    Lattice<double> phi_out{shape};
    FastRng rng_out{0ULL};
    long long const traj_out = io::load_config(scratch.path(), phi_out, rng_out);

    REQUIRE(traj_out == traj_in);
    REQUIRE(rng_out.state() == rng.state());
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        REQUIRE(phi.data()[i] == phi_out.data()[i]);
    }
}

TEST_CASE("Reader::field rejects shape, scalar-type, and kind mismatches",
          "[io][checkpoint][mismatch]") {
    test::ScratchH5 scratch{"ckpt_mismatch"};

    Lattice<double>::SizeVec const shape{4, 4};
    Lattice<double> phi{shape};
    fill_lattice(phi, [](std::size_t i) { return static_cast<double>(i); });

    {
        io::Writer w{scratch.path()};
        w.field("/field", phi);
    }

    io::Reader r{scratch.path()};

    SECTION("wrong shape throws") {
        Lattice<double> bad{Lattice<double>::SizeVec{4, 5}};
        REQUIRE_THROWS_AS(r.field("/field", bad), std::runtime_error);
    }

    SECTION("wrong scalar type throws") {
        Lattice<float> bad{Lattice<float>::SizeVec{4, 4}};
        REQUIRE_THROWS_AS(r.field("/field", bad), std::runtime_error);
    }

    SECTION("wrong field kind throws") {
        MatrixLinkLattice<math::group::U1, double> bad{
            MatrixLinkLattice<math::group::U1, double>::SizeVec{4, 4}};
        REQUIRE_THROWS_AS(r.field("/field", bad), std::runtime_error);
    }
}
