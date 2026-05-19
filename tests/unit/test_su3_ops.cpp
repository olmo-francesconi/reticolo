// Group-axiom and Cayley-Hamilton-exp tests for the SU(3) slab kernels in
// `math::su3`. Per-site stack-array kernels exercised directly + slab driver
// smoke test. Numerical tolerances ~1e-10 reflect the Cayley-Hamilton
// formula's conditioning (looser than SU(2)'s closed form).

#include <reticolo/core/rng.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;

namespace su3 = reticolo::math::su3;

namespace {

constexpr std::size_t k_n = 18;

void fill_random_anti_hermitian(std::array<double, k_n>& p, FastRng& rng) {
    // Use the production sampler at n=1.
    su3::sample_algebra_slab(p.data(), rng, 1);
}

double frob_diff(std::array<double, k_n> const& a, std::array<double, k_n> const& b) noexcept {
    double s = 0.0;
    for (std::size_t k = 0; k < k_n; ++k) {
        s += (a[k] - b[k]) * (a[k] - b[k]);
    }
    return std::sqrt(s);
}

std::array<double, k_n> identity() {
    std::array<double, k_n> i{};
    i[su3::idx_re(0, 0)] = 1.0;
    i[su3::idx_re(1, 1)] = 1.0;
    i[su3::idx_re(2, 2)] = 1.0;
    return i;
}

}  // namespace

TEST_CASE("SU(3) exp(P) · exp(P)† = I", "[unit][gauge][su3]") {
    FastRng rng{42};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n> u{};
        su3::exp_su3(u.data(), p.data(), 1.0);
        std::array<double, k_n> uu{};
        su3::mul_adj_3x3(uu.data(), u.data(), u.data());
        REQUIRE(frob_diff(uu, identity()) < 1.0e-11);
    }
}

TEST_CASE("SU(3) exp(P) · exp(-P) = I", "[unit][gauge][su3]") {
    FastRng rng{31415};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n> u_pos{};
        std::array<double, k_n> u_neg{};
        su3::exp_su3(u_pos.data(), p.data(), +1.0);
        su3::exp_su3(u_neg.data(), p.data(), -1.0);
        std::array<double, k_n> prod{};
        su3::mul_3x3(prod.data(), u_pos.data(), u_neg.data());
        REQUIRE(frob_diff(prod, identity()) < 1.0e-11);
    }
}

TEST_CASE("SU(3) exp(0) = identity (Taylor branch)", "[unit][gauge][su3]") {
    std::array<double, k_n> const p_zero{};
    std::array<double, k_n> u{};
    su3::exp_su3(u.data(), p_zero.data(), 0.5);
    REQUIRE(frob_diff(u, identity()) < 1.0e-15);
}

TEST_CASE("SU(3) exp on tiny P matches Taylor expansion to 1e-12",
          "[unit][gauge][su3]") {
    FastRng rng{27182};
    std::array<double, k_n> p{};
    fill_random_anti_hermitian(p, rng);
    // Tiny dt → Cayley-Hamilton with small c1; both branches should agree.
    std::array<double, k_n> u_small{};
    su3::exp_su3(u_small.data(), p.data(), 1.0e-5);
    // Linear approximation: exp(dt P) ≈ I + dt P.
    std::array<double, k_n> u_linear = identity();
    for (std::size_t k = 0; k < k_n; ++k) {
        u_linear[k] += 1.0e-5 * p[k];
    }
    REQUIRE(frob_diff(u_small, u_linear) < 1.0e-9);
}

TEST_CASE("SU(3) project_su3 is idempotent on unitary matrices",
          "[unit][gauge][su3]") {
    FastRng rng{16180};
    for (int trial = 0; trial < 32; ++trial) {
        std::array<double, k_n> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n> u{};
        su3::exp_su3(u.data(), p.data(), 1.0);
        std::array<double, k_n> u_proj = u;
        su3::project_su3(u_proj.data());
        REQUIRE(frob_diff(u, u_proj) < 1.0e-11);
        std::array<double, k_n> u_proj2 = u_proj;
        su3::project_su3(u_proj2.data());
        REQUIRE(frob_diff(u_proj, u_proj2) < 1.0e-15);
    }
}

TEST_CASE("SU(3) project_su3 cleans a perturbed SU(3) matrix",
          "[unit][gauge][su3]") {
    FastRng rng{11235};
    std::array<double, k_n> p{};
    fill_random_anti_hermitian(p, rng);
    std::array<double, k_n> u{};
    su3::exp_su3(u.data(), p.data(), 1.0);
    for (std::size_t k = 0; k < k_n; ++k) {
        u[k] += 1.0e-6 * rng.normal();
    }
    std::array<double, k_n> u_proj = u;
    su3::project_su3(u_proj.data());
    std::array<double, k_n> uu{};
    su3::mul_adj_3x3(uu.data(), u_proj.data(), u_proj.data());
    REQUIRE(frob_diff(uu, identity()) < 1.0e-11);
}

TEST_CASE("SU(3) (A·B)·C = A·(B·C) (3×3 matmul associativity)",
          "[unit][gauge][su3]") {
    FastRng rng{271828};
    for (int trial = 0; trial < 16; ++trial) {
        std::array<double, k_n> pa{};
        std::array<double, k_n> pb{};
        std::array<double, k_n> pc{};
        fill_random_anti_hermitian(pa, rng);
        fill_random_anti_hermitian(pb, rng);
        fill_random_anti_hermitian(pc, rng);
        std::array<double, k_n> ua{};
        std::array<double, k_n> ub{};
        std::array<double, k_n> uc{};
        su3::exp_su3(ua.data(), pa.data(), 1.0);
        su3::exp_su3(ub.data(), pb.data(), 1.0);
        su3::exp_su3(uc.data(), pc.data(), 1.0);

        std::array<double, k_n> ab{};
        std::array<double, k_n> bc{};
        std::array<double, k_n> left{};
        std::array<double, k_n> right{};
        su3::mul_3x3(ab.data(), ua.data(), ub.data());
        su3::mul_3x3(bc.data(), ub.data(), uc.data());
        su3::mul_3x3(left.data(), ab.data(), uc.data());
        su3::mul_3x3(right.data(), ua.data(), bc.data());
        REQUIRE(frob_diff(left, right) < 1.0e-11);
    }
}

TEST_CASE("SU(3) traceless_antiherm_3x3 on already-AH input is identity",
          "[unit][gauge][su3]") {
    FastRng rng{99173};
    std::array<double, k_n> p{};
    fill_random_anti_hermitian(p, rng);
    std::array<double, k_n> ta{};
    su3::traceless_antiherm_3x3(ta.data(), p.data());
    REQUIRE(frob_diff(p, ta) < 1.0e-15);
}

TEST_CASE("SU(3) expi_lmul_slab: U ← exp(0)·U leaves U unchanged",
          "[unit][gauge][su3]") {
    FastRng rng{55555};
    constexpr std::size_t n = 5;
    std::array<double, k_n * n> u{};
    for (std::size_t s = 0; s < n; ++s) {
        std::array<double, k_n> p{};
        fill_random_anti_hermitian(p, rng);
        std::array<double, k_n> u_s{};
        su3::exp_su3(u_s.data(), p.data(), 1.0);
        for (std::size_t k = 0; k < k_n; ++k) {
            u[(k * n) + s] = u_s[k];
        }
    }
    std::array<double, k_n * n> const u_orig = u;
    std::array<double, k_n * n> p_zero{};
    su3::expi_lmul_slab(u.data(), p_zero.data(), 1.0, n);
    for (std::size_t i = 0; i < k_n * n; ++i) {
        REQUIRE(std::abs(u[i] - u_orig[i]) < 1.0e-15);
    }
}
