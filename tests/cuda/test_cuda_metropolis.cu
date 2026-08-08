#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/cuda/actions/nn/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/metropolis.cuh>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

using Action = reticolo::action::Phi4<double>;
using Field  = reticolo::cuda::DeviceField<double>;
using DAct   = reticolo::cuda::DeviceAction<Action, Field>;
using Metro  = reticolo::cuda::Metropolis<DAct>;
// Spell the shape type out: `Field f{{4, 4, 4}}` is ambiguous, because a braced
// literal initialises both DeviceField(std::vector<std::size_t>) and the
// DeviceTopology aggregate (int ndim; long nsites; long shape[4]; ...).
using Shape = std::vector<std::size_t>;

constexpr std::uint64_t k_seed = 417;

void zero(Field& field) {
    std::vector<double> host(field.size(), 0.0);
    field.copy_from_host(host.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

TEST_CASE("cuda Metropolis reports an exact per-sweep action delta", "[cuda][metropolis]") {
    Action const action{.kappa = 0.18, .lambda = 1.0};
    Field field{Shape{4, 4, 4}};
    zero(field);
    DAct measure{action, field.topology()};
    Metro metro{DAct{action, field.topology()}, field, 0.3, k_seed};

    double const before = metro.last_s_full();
    auto const result   = metro.step();
    REQUIRE(result.n_attempts == static_cast<std::size_t>(field.nsites()));
    REQUIRE(result.ds == Catch::Approx(metro.last_s_full() - before).margin(1e-10));
    REQUIRE(metro.last_s_full() == Catch::Approx(measure.s_full(field)).margin(1e-9));
}

TEST_CASE("cuda Metropolis rebuilds a captured graph when sigma changes", "[cuda][metropolis]") {
    Action const action{.kappa = 0.18, .lambda = 1.0};
    Field changed{Shape{4, 4, 4}};
    Field reference{Shape{4, 4, 4}};
    zero(changed);
    zero(reference);

    Metro after_capture{DAct{action, changed.topology()}, changed, 0.0, k_seed};
    after_capture.run(1);  // captures sigma=0 and advances the Philox counter
    after_capture.set_sigma(0.3);
    after_capture.step();

    Metro fresh{DAct{action, reference.topology()}, reference, 0.3, k_seed};
    fresh.set_rng_counter(1);  // same counter and field as `after_capture` above
    fresh.step();

    std::vector<double> lhs(changed.size());
    std::vector<double> rhs(reference.size());
    changed.copy_to_host(lhs.data());
    reference.copy_to_host(rhs.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(lhs == rhs);
}

TEST_CASE("cuda Metropolis resync preserves acceptance counters", "[cuda][metropolis]") {
    Action const action{.kappa = 0.18, .lambda = 1.0};
    Field field{Shape{4, 4, 4}};
    zero(field);
    DAct measure{action, field.topology()};
    Metro metro{DAct{action, field.topology()}, field, 0.3, k_seed};

    metro.run(3);
    double const acceptance = metro.acceptance();
    double const s          = metro.resync_s_full();
    REQUIRE(metro.acceptance() == Catch::Approx(acceptance));
    REQUIRE(s == Catch::Approx(measure.s_full(field)).margin(1e-9));

    metro.reset_acceptance_stats();
    REQUIRE(metro.acceptance() == 0.0);
}

TEST_CASE("cuda Metropolis rejects non-bipartite periodic lattices", "[cuda][metropolis]") {
    Action const action{.kappa = 0.18, .lambda = 1.0};
    Field odd{Shape{3, 4}};
    REQUIRE_THROWS_AS((Metro{DAct{action, odd.topology()}, odd, 0.3, k_seed}),
                      std::invalid_argument);
}
