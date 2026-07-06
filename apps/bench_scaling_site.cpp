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

template <class Action>
void run_bench(
    char const* name, Action const& a, Lattice<double>& phi, FastRng& rng, char const* th) {
    Lattice<double> force{phi.indexing()};
    double const fms = time_ms([&] { a.compute_force(phi, force); }, 30);
    double sink      = 0.0;
    double const sms = time_ms([&] { sink += a.s_full(phi); }, 30);
    alg::Hmc hmc{a, phi, rng, {.tau = 1.0, .n_md = 8}, alg::integ::leapfrog, log::Mode::silent};
    double const trms = time_ms([&] { (void)hmc.step(log::Mode::silent); }, 10);
    std::printf("%s %s %.4f %.4f %.4f\n", name, th, fms, sms, trms);
    (void)sink;
}

int main(int argc, char** argv) {
    reticolo::log::off();
    int const L    = argc > 1 ? std::atoi(argv[1]) : 64;
    char const* th = std::getenv("OMP_NUM_THREADS");
    if (th == nullptr) {
        th = "1";
    }
    Lattice<double>::SizeVec sh(4, static_cast<std::size_t>(L));
    Lattice<double> phi{sh};
    FastRng rng{42};
    reticolo::bench::hot_init(phi, rng);

    act::Phi4<double> const p4{.kappa = 0.18, .lambda = 1.0};
    act::Phi6<double> const p6{.kappa = 0.18, .lambda = 1.0, .g6 = 0.1};
    act::SineGordon<double> const sg{.kappa = 0.18, .alpha = 0.7};

    run_bench("Phi4", p4, phi, rng, th);
    run_bench("Phi6", p6, phi, rng, th);
    run_bench("SineGordon", sg, phi, rng, th);
    return 0;
}
