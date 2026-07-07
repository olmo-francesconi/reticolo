// Batched-vs-scalar consistency for the gauge fast paths added in the
// vectorization pass: `action::detail::wilson_kernels<G>::template s_full_plane_re_tr_sum` (Wilson
// s_full plane sums) against the per-plaquette `plaq_re_tr` reference, and the AoSoA
// `su3::expi_lmul_slab` against the per-site Cayley-Hamilton `exp_su3`.
// Shape 5×6×7 is deliberately awkward: ns % 8 ≠ 0 exercises the scalar
// tails, every row crosses batch boundaries (gather fallback), and the
// SU(2) misaligned-L0 visit_plane guard is taken.

#include <reticolo/action/gauge/detail/wilson_su2.hpp>
#include <reticolo/action/gauge/detail/wilson_su3.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/su2_ops.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

namespace {

template <class G, class T>
double plane_sum_reference(MatrixLinkLattice<G, T> const& u) {
    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();
    double accum         = 0.0;
    for (std::size_t mu = 0; mu < d; ++mu) {
        T const* mb = u.mu_block_data(mu);
        for (std::size_t nu = mu + 1; nu < d; ++nu) {
            T const* nb = u.mu_block_data(nu);
            for (std::size_t s = 0; s < ns; ++s) {
                Site const x{s};
                accum += action::detail::wilson_kernels<G>::plaq_re_tr(
                    mb, nb, s, u.next(x, mu).value(), u.next(x, nu).value(), ns);
            }
        }
    }
    return accum;
}

template <class G, class T>
double plane_sum_batched(MatrixLinkLattice<G, T> const& u) {
    std::size_t const d = u.ndims();
    double accum        = 0.0;
    for (std::size_t mu = 0; mu < d; ++mu) {
        for (std::size_t nu = mu + 1; nu < d; ++nu) {
            accum +=
                action::detail::wilson_kernels<G>::template s_full_plane_re_tr_sum<T>(u, mu, nu);
        }
    }
    return accum;
}

template <class G, class T>
void fill_random(MatrixLinkLattice<G, T>& u, FastRng& rng) {
    std::vector<double> buf(u.ncomponents());
    rng.normal_fill(buf.data(), buf.size());
    std::size_t i = 0;
    for (auto& x : u) {
        x = static_cast<T>(0.5 * buf[i++]);
    }
}

}  // namespace

TEST_CASE("batched plane Re Tr sum matches per-plaquette reference", "[gauge][s_full]") {
    FastRng rng{20260610ULL};

    SECTION("SU3 double") {
        MatrixLinkLattice<math::group::SU3, double> u{{5, 6, 7}};
        fill_random(u, rng);
        double const ref = plane_sum_reference(u);
        CHECK(std::abs(plane_sum_batched(u) - ref) <= 1e-10 * std::abs(ref));
    }
    SECTION("SU3 float") {
        MatrixLinkLattice<math::group::SU3, float> u{{5, 6, 7}};
        fill_random(u, rng);
        double const ref = plane_sum_reference(u);
        CHECK(std::abs(plane_sum_batched(u) - ref) <= 1e-5 * std::abs(ref));
    }
    SECTION("SU2 double, misaligned L0 (visit_plane guard)") {
        MatrixLinkLattice<math::group::SU2, double> u{{5, 6, 7}};
        fill_random(u, rng);
        double const ref = plane_sum_reference(u);
        CHECK(std::abs(plane_sum_batched(u) - ref) <= 1e-10 * std::abs(ref));
    }
    SECTION("SU2 double, aligned L0 (batched path)") {
        MatrixLinkLattice<math::group::SU2, double> u{{16, 6, 5}};
        fill_random(u, rng);
        double const ref = plane_sum_reference(u);
        CHECK(std::abs(plane_sum_batched(u) - ref) <= 1e-10 * std::abs(ref));
    }
}

TEST_CASE("batched expi_lmul_slab matches per-site exp_su3", "[gauge][expi]") {
    namespace su3 = math::su3;
    FastRng rng{99ULL};
    std::size_t const n = 1003;  // n % 8 ≠ 0 → scalar tail covered
    double const dt     = 0.137;

    std::vector<double> p(18 * n);
    su3::sample_algebra_slab(p.data(), rng, n);
    // Force a few sites through the small-c1 Taylor blend.
    for (std::size_t s : {std::size_t{0}, std::size_t{500}, n - 1}) {
        for (std::size_t k = 0; k < 18; ++k) {
            p[(k * n) + s] *= 1e-6;
        }
    }

    std::vector<double> u_ref(18 * n);
    std::vector<double> u_new(18 * n);
    for (std::size_t s = 0; s < n; ++s) {
        double m[18];
        for (std::size_t k = 0; k < 18; ++k) {
            m[k] = 0.3 * p[(k * n) + ((s + 1) % n)];  // arbitrary start matrix
        }
        m[su3::idx_re(0, 0)] += 1.0;
        m[su3::idx_re(1, 1)] += 1.0;
        m[su3::idx_re(2, 2)] += 1.0;
        su3::project_su3(m);
        for (std::size_t k = 0; k < 18; ++k) {
            u_ref[(k * n) + s] = m[k];
            u_new[(k * n) + s] = m[k];
        }
    }

    for (std::size_t s = 0; s < n; ++s) {
        double ps[18];
        double us[18];
        double v[18];
        double o[18];
        for (std::size_t k = 0; k < 18; ++k) {
            ps[k] = p[(k * n) + s];
            us[k] = u_ref[(k * n) + s];
        }
        su3::exp_su3(v, ps, dt);
        su3::mul_3x3(o, v, us);
        for (std::size_t k = 0; k < 18; ++k) {
            u_ref[(k * n) + s] = o[k];
        }
    }
    su3::expi_lmul_slab(u_new.data(), p.data(), dt, n);

    double max_err = 0.0;
    for (std::size_t i = 0; i < 18 * n; ++i) {
        max_err = std::max(max_err, std::abs(u_new[i] - u_ref[i]));
    }
    CHECK(max_err < 1e-12);
}
