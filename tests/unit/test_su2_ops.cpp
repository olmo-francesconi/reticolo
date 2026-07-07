// Group-axiom and closed-form-exp tests for the SU(2) slab kernels in
// `math::su2`. Verified per-site (slab over n=1) so each test exercises the
// stack-array helpers AND the slab driver in one pass.

#include <reticolo/core/rng/rng.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;

namespace su2 = reticolo::math::su2;

namespace {

// Helpers to keep test bodies short.
constexpr std::size_t k_n_real = 8;

void fill_random_anti_hermitian(std::array<double, k_n_real>& p, FastRng& rng) {
    double const h1 = rng.normal();
    double const h2 = rng.normal();
    double const h3 = rng.normal();
    p[0]            = 0.0;
    p[1]            = h3;
    p[2]            = h2;
    p[3]            = h1;
    p[4]            = -h2;
    p[5]            = h1;
    p[6]            = 0.0;
    p[7]            = -h3;
}

double frob_diff(std::array<double, k_n_real> const& a,
                 std::array<double, k_n_real> const& b) noexcept {
    double s = 0.0;
    for (std::size_t k = 0; k < k_n_real; ++k) {
        s += (a[k] - b[k]) * (a[k] - b[k]);
    }
    return std::sqrt(s);
}

// A · A† for SU(2) using the per-site kernels. Returns 8 reals.
std::array<double, k_n_real> a_times_a_dag(std::array<double, k_n_real> const& u) {
    std::array<double, k_n_real> out{};
    su2::mul_adj_2x2(out.data(), u.data(), u.data());
    return out;
}

}  // namespace

TEST_CASE("SU(2) project_su2: exp(P) is unitary up to ~1e-15", "[unit][gauge][su2]") {
    FastRng rng{42};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n_real> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n_real> u{};
        su2::exp_su2(u.data(), p.data(), 1.0);
        auto const uu = a_times_a_dag(u);
        // U·U† should be the 2×2 identity.
        std::array<double, k_n_real> expected{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        REQUIRE(frob_diff(uu, expected) < 1.0e-12);
    }
}

TEST_CASE("SU(2) exp(P) · exp(-P) = I", "[unit][gauge][su2]") {
    FastRng rng{31415};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n_real> p{};
        fill_random_anti_hermitian(p, rng);

        std::array<double, k_n_real> u_pos{};
        std::array<double, k_n_real> u_neg{};
        su2::exp_su2(u_pos.data(), p.data(), +1.0);
        su2::exp_su2(u_neg.data(), p.data(), -1.0);

        std::array<double, k_n_real> prod{};
        su2::mul_2x2(prod.data(), u_pos.data(), u_neg.data());

        std::array<double, k_n_real> expected{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        REQUIRE(frob_diff(prod, expected) < 1.0e-13);
    }
}

TEST_CASE("SU(2) exp(0) = identity (small-||h|| branch)", "[unit][gauge][su2]") {
    std::array<double, k_n_real> const p_zero{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, k_n_real> u{};
    su2::exp_su2(u.data(), p_zero.data(), 0.5);
    std::array<double, k_n_real> const expected{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    REQUIRE(frob_diff(u, expected) < 1.0e-15);
}

TEST_CASE("SU(2) project_su2 is idempotent on unitary matrices", "[unit][gauge][su2]") {
    FastRng rng{27182};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n_real> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n_real> u{};
        su2::exp_su2(u.data(), p.data(), 1.0);

        std::array<double, k_n_real> u_proj = u;
        su2::project_su2(u_proj.data());
        REQUIRE(frob_diff(u, u_proj) < 1.0e-12);

        std::array<double, k_n_real> u_proj2 = u_proj;
        su2::project_su2(u_proj2.data());
        REQUIRE(frob_diff(u_proj, u_proj2) < 1.0e-15);
    }
}

TEST_CASE("SU(2) project_su2 cleans a perturbed SU(2) matrix", "[unit][gauge][su2]") {
    FastRng rng{16180};
    std::array<double, k_n_real> p{};
    fill_random_anti_hermitian(p, rng);
    std::array<double, k_n_real> u{};
    su2::exp_su2(u.data(), p.data(), 1.0);
    // Add ~1e-6 noise: outside-SU(2) drift.
    for (std::size_t k = 0; k < k_n_real; ++k) {
        u[k] += 1.0e-6 * rng.normal();
    }
    std::array<double, k_n_real> u_proj = u;
    su2::project_su2(u_proj.data());
    auto const uu = a_times_a_dag(u_proj);
    std::array<double, k_n_real> expected{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    REQUIRE(frob_diff(uu, expected) < 1.0e-12);
}

TEST_CASE("SU(2) (A·B)·C = A·(B·C)", "[unit][gauge][su2]") {
    FastRng rng{11235};
    for (int trial = 0; trial < 16; ++trial) {
        std::array<double, k_n_real> pa{};
        std::array<double, k_n_real> pb{};
        std::array<double, k_n_real> pc{};
        fill_random_anti_hermitian(pa, rng);
        fill_random_anti_hermitian(pb, rng);
        fill_random_anti_hermitian(pc, rng);
        std::array<double, k_n_real> ua{};
        std::array<double, k_n_real> ub{};
        std::array<double, k_n_real> uc{};
        su2::exp_su2(ua.data(), pa.data(), 1.0);
        su2::exp_su2(ub.data(), pb.data(), 1.0);
        su2::exp_su2(uc.data(), pc.data(), 1.0);

        std::array<double, k_n_real> ab{};
        std::array<double, k_n_real> bc{};
        std::array<double, k_n_real> left{};
        std::array<double, k_n_real> right{};
        su2::mul_2x2(ab.data(), ua.data(), ub.data());
        su2::mul_2x2(bc.data(), ub.data(), uc.data());
        su2::mul_2x2(left.data(), ab.data(), uc.data());
        su2::mul_2x2(right.data(), ua.data(), bc.data());
        REQUIRE(frob_diff(left, right) < 1.0e-13);
    }
}

TEST_CASE("SU(2) mul_adj_2x2(A, A) is identity for SU(2) inputs", "[unit][gauge][su2]") {
    FastRng rng{271828};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n_real> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n_real> u{};
        su2::exp_su2(u.data(), p.data(), 1.0);
        std::array<double, k_n_real> out{};
        su2::mul_adj_2x2(out.data(), u.data(), u.data());
        std::array<double, k_n_real> expected{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        REQUIRE(frob_diff(out, expected) < 1.0e-13);
    }
}

TEST_CASE("SU(2) expi_lmul_slab: U ← exp(0)·U leaves U unchanged", "[unit][gauge][su2]") {
    FastRng rng{99173};
    constexpr std::size_t n = 7;
    std::array<double, k_n_real * n> u{};
    // Fill with non-trivial SU(2) matrices.
    for (std::size_t s = 0; s < n; ++s) {
        std::array<double, k_n_real> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n_real> u_s{};
        su2::exp_su2(u_s.data(), p.data(), 1.0);
        for (std::size_t k = 0; k < k_n_real; ++k) {
            u[(k * n) + s] = u_s[k];
        }
    }
    std::array<double, k_n_real * n> u_orig = u;
    std::array<double, k_n_real * n> p_zero{};
    su2::expi_lmul_slab(u.data(), p_zero.data(), 1.0, n);
    for (std::size_t i = 0; i < k_n_real * n; ++i) {
        REQUIRE(u[i] == Catch::Approx(u_orig[i]).margin(1.0e-15));
    }
}
