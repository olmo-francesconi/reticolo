#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using namespace reticolo;
using clk = std::chrono::steady_clock;

template <class F>
double time_ms(F&& f, int reps) {
    for (int i = 0; i < 3; ++i) {
        f();
    }
    auto const t0 = clk::now();
    for (int i = 0; i < reps; ++i) {
        f();
    }
    return std::chrono::duration<double>(clk::now() - t0).count() / reps * 1e3;
}

int main(int argc, char** argv) {
    reticolo::log::off();
    using Group = gauge_group::SU3;
    using Field = MatrixLinkLattice<Group, double>;
    int const L = argc > 1 ? std::atoi(argv[1]) : 16;

    char const* th = std::getenv("OMP_NUM_THREADS");
    if (th == nullptr) {
        th = "1";
    }
    Field::SizeVec sh(4, static_cast<std::size_t>(L));
    Field links{sh};
    FastRng rng{42};
    reticolo::bench::hot_init(links, rng);
    act::Wilson<Group, double> const a{.beta = 6.0};
    Field force{links.indexing()};

    double const fms = time_ms([&] { a.compute_force(links, force); }, 20);
    double sink      = 0.0;
    double const sms = time_ms([&] { sink += a.s_full(links); }, 20);

    alg::Hmc hmc{a, links, rng, {.tau = 1.0, .n_md = 8}, alg::integ::leapfrog, log::Mode::silent};
    double const mdms = time_ms([&] { hmc.integrate_only(1.0, 8); }, 8);
    double const trms = time_ms([&] { (void)hmc.step(log::Mode::silent); }, 8);

    std::printf("FORCE  L=%-3d threads=%-3s val=%10.4f\n", L, th, fms);
    std::printf("SFULL  L=%-3d threads=%-3s val=%10.4f\n", L, th, sms);
    std::printf("MDLOOP L=%-3d threads=%-3s val=%10.4f\n", L, th, mdms);
    std::printf("TRAJ   L=%-3d threads=%-3s val=%10.4f\n", L, th, trms);

    // Bit-invariance probe: recompute force/s_full/drift on a fresh config and
    // dump full-precision checksums so different thread counts can be diffed.
    Field u2{sh};
    FastRng rng2{7};
    reticolo::bench::hot_init(u2, rng2);
    Field f2{u2.indexing()};
    a.compute_force(u2, f2);
    double fsum = 0.0;
    for (std::size_t mu = 0; mu < f2.ndims(); ++mu) {
        double const* blk    = f2.mu_block_data(mu);
        std::size_t const nn = f2.nsites() * gauge_group::SU3::n_real_components;
        for (std::size_t i = 0; i < nn; ++i) {
            fsum += blk[i] * blk[i];
        }
    }
    double const sful = a.s_full(u2);
    alg::integ::drift_field(u2, f2, 0.037);
    double usum = 0.0;
    for (std::size_t mu = 0; mu < u2.ndims(); ++mu) {
        double const* blk    = u2.mu_block_data(mu);
        std::size_t const nn = u2.nsites() * gauge_group::SU3::n_real_components;
        for (std::size_t i = 0; i < nn; ++i) {
            usum += blk[i] * blk[i];
        }
    }
    std::printf(
        "CHECK  threads=%-3s s_full=%.17g force_sq=%.17g drift_sq=%.17g\n", th, sful, fsum, usum);

    // Full-trajectory invariance: fixed seed → the ΔH of a short chain must be
    // bit-identical across thread counts (covers momentum sampling + MD + h0/h1).
    Field u3{sh};
    FastRng rng3{99};
    reticolo::bench::hot_init(u3, rng3);
    alg::Hmc hmc3{a, u3, rng3, {.tau = 1.0, .n_md = 6}, alg::integ::omelyan2, log::Mode::silent};
    double dh_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        dh_sum += hmc3.step(log::Mode::silent).dH;
    }
    std::printf("TRAJCHK threads=%-3s dH_sum=%.17g\n", th, dh_sum);
    (void)sink;
    return 0;
}
