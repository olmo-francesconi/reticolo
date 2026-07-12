#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/obs/reduce.hpp>

#include <cmath>
#include <complex>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Wire CUSTOM lambdas straight into the fused `obs::reduce` on a complex field.
// The complex mean is split into two double lanes (Σre, Σim) and the amplitude
// uses sqrt(re²+im²) rather than std::abs — both per the vectorisation notes in
// reduce.hpp, so all three lanes stay SIMD-friendly. One parallel sweep folds
// them into a std::tuple<double, double, double> of raw sums.

using reticolo::Lattice;
using reticolo::Site;
namespace obs = reticolo::obs;

namespace {

using cd = std::complex<double>;

// The custom observable: three double lanes, one sweep, reassembled after.
template <class T>
[[nodiscard]] std::pair<cd, double> mean_and_amplitude(Lattice<std::complex<T>> const& l) {
    auto const [sum_re, sum_im, sum_abs] = obs::reduce(
        l,
        [](std::complex<T> z) { return static_cast<double>(z.real()); },  // Σre
        [](std::complex<T> z) { return static_cast<double>(z.imag()); },  // Σim
        [](std::complex<T> z) {                                           // Σ|φ|
            auto const re = static_cast<double>(z.real());
            auto const im = static_cast<double>(z.imag());
            return std::sqrt((re * re) + (im * im));
        });
    auto const inv_v = 1.0 / static_cast<double>(l.nsites());
    return {cd{sum_re * inv_v, sum_im * inv_v}, sum_abs * inv_v};
}

}  // namespace

TEST_CASE("obs::reduce with custom lambdas: complex mean + amplitude on a constant field",
          "[obs][reduce][complex]") {
    Lattice<cd> phi{{4, 4}, cd{3.0, 4.0}};  // |3+4i| = 5
    auto const [mean, amp] = mean_and_amplitude(phi);
    REQUIRE(mean.real() == Catch::Approx(3.0).margin(1e-14));
    REQUIRE(mean.imag() == Catch::Approx(4.0).margin(1e-14));
    REQUIRE(amp == Catch::Approx(5.0).margin(1e-14));
}

TEST_CASE("obs::reduce with custom lambdas: cancelling phases -> zero mean, unit amplitude",
          "[obs][reduce][complex]") {
    Lattice<cd> phi{{2, 2}};
    phi[Site{0}]           = {1.0, 0.0};
    phi[Site{1}]           = {0.0, 1.0};
    phi[Site{2}]           = {-1.0, 0.0};
    phi[Site{3}]           = {0.0, -1.0};
    auto const [mean, amp] = mean_and_amplitude(phi);
    REQUIRE(mean.real() == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(mean.imag() == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(amp == Catch::Approx(1.0).margin(1e-15));
}

TEST_CASE("obs::reduce still fuses homogeneous double moments into a tuple", "[obs][reduce]") {
    Lattice<double> phi{{4, 4}, 2.0};
    auto const [s1, s2] = obs::reduce(phi, obs::kernel::phi, obs::kernel::phi_sq);
    REQUIRE(s1 == Catch::Approx(2.0 * 16).margin(1e-13));  // Σφ  = 2·16
    REQUIRE(s2 == Catch::Approx(4.0 * 16).margin(1e-13));  // Σφ² = 4·16
}
