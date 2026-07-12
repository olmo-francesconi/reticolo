#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/obs/catalog.hpp>
#include <reticolo/obs/concepts.hpp>
#include <reticolo/obs/reduce.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <tuple>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::Lattice;
using reticolo::Site;

namespace obs = reticolo::obs;

namespace {

// The standard unpack idiom: reduce one kernel over the field, average the sum.
template <class T, class K>
[[nodiscard]] double avg(Lattice<T> const& l, K const& k) {
    return obs::mean_of(std::get<0>(obs::reduce(l, k)), static_cast<double>(l.nsites()));
}

}  // namespace

TEST_CASE("kernel::phi and kernel::phi_sq on a constant configuration", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, 0.7};
    REQUIRE(avg(phi, obs::kernel::phi) == Catch::Approx(0.7).margin(1e-14));      // <phi>
    REQUIRE(avg(phi, obs::kernel::phi_sq) == Catch::Approx(0.49).margin(1e-14));  // <phi^2>
}

TEST_CASE("mag_abs_of is |<phi>| (sign-independent)", "[obs]") {
    Lattice<double> pos{{4, 4}, 0.3};
    Lattice<double> neg{{4, 4}, -0.3};
    auto const abs_m = [](Lattice<double> const& l) {
        return obs::mag_abs_of(std::get<0>(obs::reduce(l, obs::kernel::phi)),
                               static_cast<double>(l.nsites()));
    };
    REQUIRE(abs_m(pos) == Catch::Approx(0.3).margin(1e-14));
    REQUIRE(abs_m(neg) == Catch::Approx(0.3).margin(1e-14));
}

TEST_CASE("kernel::phi mean of an antiparity-flipped field is zero", "[obs]") {
    Lattice<double> phi{{4, 4}};
    for (Site const x : phi.sites()) {
        phi[x] = (x.value() % 2 == 0) ? 1.0 : -1.0;
    }
    REQUIRE(avg(phi, obs::kernel::phi) == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(avg(phi, obs::kernel::phi_sq) == 1.0);
}

TEST_CASE("kernel::phi_quartic on a uniform field is phi^4", "[obs]") {
    Lattice<double> phi{{4, 4}, 2.0};
    REQUIRE(avg(phi, obs::kernel::phi_quartic) == 16.0);
}

TEST_CASE("sq_of_mean_of is <phi>^2", "[obs]") {
    Lattice<double> phi{{4, 4, 4}, 0.5};
    REQUIRE(obs::sq_of_mean_of(std::get<0>(obs::reduce(phi, obs::kernel::phi)),
                               static_cast<double>(phi.nsites())) ==
            Catch::Approx(0.25).margin(1e-14));
}

TEST_CASE("obs::reduce fuses several kernels into one sweep", "[obs][reduce]") {
    Lattice<double> phi{{4, 4}, 2.0};
    auto const [s1, s2, s4] =
        obs::reduce(phi, obs::kernel::phi, obs::kernel::phi_sq, obs::kernel::phi_quartic);
    auto const v = static_cast<double>(phi.nsites());
    REQUIRE(obs::mean_of(s1, v) == Catch::Approx(2.0).margin(1e-14));
    REQUIRE(obs::mean_of(s2, v) == Catch::Approx(4.0).margin(1e-14));
    REQUIRE(obs::mean_of(s4, v) == Catch::Approx(16.0).margin(1e-14));
}

TEST_CASE("obs::kernel casts a float field to double accumulation", "[obs][reduce]") {
    Lattice<float> phi{{4, 4}, 0.5F};
    using ret_t = decltype(obs::reduce(phi, obs::kernel::phi_sq));
    static_assert(std::is_same_v<std::tuple_element_t<0, ret_t>, double>);
    auto const s = obs::reduce(phi, obs::kernel::phi_sq);
    REQUIRE(obs::mean_of(std::get<0>(s), static_cast<double>(phi.nsites())) ==
            Catch::Approx(0.25).margin(1e-7));
}

TEST_CASE("obs::two_point at r=0 reduces to <phi^2>", "[obs][two_point]") {
    Lattice<double> phi{{6, 6, 6}};
    for (Site const x : phi.sites()) {
        phi[x] = 0.1 * static_cast<double>(x.value());
    }
    for (std::size_t mu = 0; mu < phi.ndims(); ++mu) {
        REQUIRE(obs::two_point(phi, 0, mu) ==
                Catch::Approx(avg(phi, obs::kernel::phi_sq)).margin(1e-12));
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

TEST_CASE("obs concepts: builtin kernels and observer lambdas model them", "[obs][concept]") {
    static_assert(obs::SiteKernel<decltype(obs::kernel::phi), double>);
    static_assert(obs::SiteKernel<decltype(obs::kernel::phi_sq), float>);
    auto wrap_mean = [](Lattice<double> const& l) { return avg(l, obs::kernel::phi); };
    static_assert(obs::Observer<decltype(wrap_mean), double>);
}
