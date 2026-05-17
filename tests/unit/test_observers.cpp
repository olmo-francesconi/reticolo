#include <reticolo/core/bc.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/obs/catalog.hpp>
#include <reticolo/obs/concepts.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

using reticolo::Bc;
using reticolo::BcMask;
using reticolo::Lattice;
using reticolo::Site;

namespace obs = reticolo::obs;

TEST_CASE("obs::mean and obs::sq on a constant configuration", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, BcMask{}, 0.7};
    REQUIRE(std::abs(obs::mean(phi) - 0.7) < 1e-14);
    REQUIRE(std::abs(obs::sq(phi) - 0.49) < 1e-14);
}

TEST_CASE("obs::magnetization is |<phi>| (sign-independent)", "[obs]") {
    Lattice<double> pos{{4, 4}, BcMask{}, 0.3};
    Lattice<double> neg{{4, 4}, BcMask{}, -0.3};
    REQUIRE(std::abs(obs::magnetization(pos) - 0.3) < 1e-14);
    REQUIRE(std::abs(obs::magnetization(neg) - 0.3) < 1e-14);
}

TEST_CASE("obs::mean of antiparity-flipped field is zero", "[obs]") {
    Lattice<double> phi{{4, 4}};
    for (Site const x : phi.sites()) {
        phi[x] = (x.value() % 2 == 0) ? 1.0 : -1.0;
    }
    REQUIRE(std::abs(obs::mean(phi)) < 1e-15);
    REQUIRE(obs::sq(phi) == 1.0);
}

TEST_CASE("obs::m2 equals obs::sq", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, BcMask{}, 0.5};
    REQUIRE(obs::m2(phi) == obs::sq(phi));
}

TEST_CASE("obs::m4 on a uniform field is phi^4", "[obs]") {
    Lattice<double> phi{{4, 4}, BcMask{}, 2.0};
    REQUIRE(obs::m4(phi) == 16.0);
}

TEST_CASE("obs::two_point at r=0 reduces to obs::sq", "[obs][two_point]") {
    Lattice<double> phi{{6, 6, 6}};
    for (Site const x : phi.sites()) {
        phi[x] = 0.1 * static_cast<double>(x.value());
    }
    for (std::size_t mu = 0; mu < phi.ndims(); ++mu) {
        REQUIRE(std::abs(obs::two_point(phi, 0, mu) - obs::sq(phi)) < 1e-12);
    }
}

TEST_CASE("obs::two_point recovers the squared mean on a constant field", "[obs][two_point]") {
    Lattice<double> phi{{4, 4}, BcMask{}, 1.5};
    for (std::size_t mu = 0; mu < 2; ++mu) {
        for (std::size_t r = 0; r <= 3; ++r) {
            REQUIRE(std::abs(obs::two_point(phi, r, mu) - 1.5 * 1.5) < 1e-12);
        }
    }
}

TEST_CASE("obs::two_point: plane wave along mu=0", "[obs][two_point]") {
    // phi(x, y) = cos(2π x / L_0); L_0 = 4.
    // G(r, mu=0) = (1/N) Σ cos(2π x / L) cos(2π (x + r) / L) = 0.5 cos(2π r / L).
    constexpr std::size_t k_l = 4;
    Lattice<double> phi{{k_l, k_l}};
    for (Site const s : phi.sites()) {
        auto const i   = s.value() % k_l;
        double const v = std::cos(2.0 * M_PI * static_cast<double>(i) / k_l);
        phi[s]         = v;
    }
    for (std::size_t r = 0; r <= 3; ++r) {
        double const expected = 0.5 * std::cos(2.0 * M_PI * static_cast<double>(r) / k_l);
        REQUIRE(std::abs(obs::two_point(phi, r, 0) - expected) < 1e-12);
    }
}

TEST_CASE("obs::two_point rejects mu out of range", "[obs][two_point]") {
    Lattice<double> phi{{4, 4}};
    REQUIRE_THROWS_AS(obs::two_point(phi, 1, 2), std::out_of_range);
}

TEST_CASE("obs::two_point rejects open BC in direction mu for r>0", "[obs][two_point]") {
    Lattice<double> phi{{4, 4}, BcMask{Bc::Open, Bc::Periodic}};
    REQUIRE_THROWS_AS(obs::two_point(phi, 1, 0), std::invalid_argument);
    REQUIRE_NOTHROW(obs::two_point(phi, 1, 1));
    REQUIRE_NOTHROW(obs::two_point(phi, 0, 0));  // r=0 is unaffected by BC
}

// Observer concept satisfied by free function templates wrapped in a lambda.
TEST_CASE("obs::Observer concept is satisfied by lambda wrappers", "[obs][concept]") {
    auto wrap_mean = [](Lattice<double> const& l) { return obs::mean(l); };
    static_assert(obs::Observer<decltype(wrap_mean), double>);
}
