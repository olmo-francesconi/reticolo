#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/obs/catalog.hpp>
#include <reticolo/obs/concepts.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::Lattice;
using reticolo::Site;

namespace obs = reticolo::obs;

TEST_CASE("obs::mean and obs::sq on a constant configuration", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, 0.7};
    REQUIRE(obs::mean(phi) == Catch::Approx(0.7).margin(1e-14));
    REQUIRE(obs::sq(phi) == Catch::Approx(0.49).margin(1e-14));
}

TEST_CASE("obs::mag::abs is |<phi>| (sign-independent)", "[obs]") {
    Lattice<double> pos{{4, 4}, 0.3};
    Lattice<double> neg{{4, 4}, -0.3};
    REQUIRE(obs::mag::abs(pos) == Catch::Approx(0.3).margin(1e-14));
    REQUIRE(obs::mag::abs(neg) == Catch::Approx(0.3).margin(1e-14));
}

TEST_CASE("obs::mean of antiparity-flipped field is zero", "[obs]") {
    Lattice<double> phi{{4, 4}};
    for (Site const x : phi.sites()) {
        phi[x] = (x.value() % 2 == 0) ? 1.0 : -1.0;
    }
    REQUIRE(obs::mean(phi) == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(obs::sq(phi) == 1.0);
}

TEST_CASE("obs::quartic on a uniform field is phi^4", "[obs]") {
    Lattice<double> phi{{4, 4}, 2.0};
    REQUIRE(obs::quartic(phi) == 16.0);
}

TEST_CASE("obs::sq_of_mean is <phi>^2", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, 0.5};
    REQUIRE(obs::sq_of_mean(phi) == Catch::Approx(0.25).margin(1e-14));
}

TEST_CASE("obs::two_point at r=0 reduces to obs::sq", "[obs][two_point]") {
    Lattice<double> phi{{6, 6, 6}};
    for (Site const x : phi.sites()) {
        phi[x] = 0.1 * static_cast<double>(x.value());
    }
    for (std::size_t mu = 0; mu < phi.ndims(); ++mu) {
        REQUIRE(obs::two_point(phi, 0, mu) == Catch::Approx(obs::sq(phi)).margin(1e-12));
    }
}

TEST_CASE("obs::two_point recovers the squared mean on a constant field", "[obs][two_point]") {
    Lattice<double> phi{{4, 4}, 1.5};
    for (std::size_t mu = 0; mu < 2; ++mu) {
        for (std::size_t r = 0; r <= 3; ++r) {
            REQUIRE(obs::two_point(phi, r, mu) == Catch::Approx(1.5 * 1.5).margin(1e-12));
        }
    }
}

TEST_CASE("obs::two_point: plane wave along mu=0", "[obs][two_point]") {
    constexpr std::size_t k_l = 4;
    Lattice<double> phi{{k_l, k_l}};
    for (Site const s : phi.sites()) {
        auto const i   = s.value() % k_l;
        double const v = std::cos(2.0 * std::numbers::pi * static_cast<double>(i) / k_l);
        phi[s]         = v;
    }
    for (std::size_t r = 0; r <= 3; ++r) {
        double const expected =
            0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(r) / k_l);
        REQUIRE(obs::two_point(phi, r, 0) == Catch::Approx(expected).margin(1e-12));
    }
}

TEST_CASE("obs::two_point rejects mu out of range", "[obs][two_point]") {
    Lattice<double> phi{{4, 4}};
    REQUIRE_THROWS_AS(obs::two_point(phi, 1, 2), std::out_of_range);
}

TEST_CASE("obs::Observer concept is satisfied by lambda wrappers", "[obs][concept]") {
    auto wrap_mean = [](Lattice<double> const& l) { return obs::mean(l); };
    static_assert(obs::Observer<decltype(wrap_mean), double>);
}
