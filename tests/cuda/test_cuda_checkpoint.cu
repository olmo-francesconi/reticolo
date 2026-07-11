#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/actions/nn/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/io/checkpoint.hpp>

#include "../test_helpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

// A checkpointed-and-resumed CUDA run reproduces an uninterrupted one
// bit-for-bit. The device RNG is counter-based Philox, so
// restoring (seed, trajectory counter) + the field fully determines the
// continuation. This .cu orchestrates cuda::Hmc and io::checkpoint in one TU —
// io::Writer PIMPLs HDF5, so nvcc never sees <hdf5.h>; it links the prebuilt
// reticolo::io archive. Requires a GPU at run time (RETICOLO_ENABLE_CUDA).

using reticolo::test::scratch_path;

namespace {

// NB: no namespace-scope `using namespace reticolo;` — it pulls the reticolo::log
// namespace into global lookup, which collides with CUDA's ::log math function in
// the math_functions.hpp inline bodies ("reference to 'log' is ambiguous"). The
// apps dodge this by scoping the using-directive inside main(); we do the same in
// the TEST_CASE body below.
using DField = reticolo::cuda::DeviceField<double>;
using DAct   = reticolo::cuda::DeviceAction<reticolo::action::Phi4<double>, DField>;

constexpr int k_n_first  = 7;
constexpr int k_n_second = 5;
std::vector<std::size_t> const k_shape{4, 4, 4, 4};

DField make_cold_field() {
    DField field{k_shape};
    std::vector<double> const zero(field.size(), 0.0);
    field.copy_from_host(zero.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
    return field;
}

void snapshot(DField const& field, std::vector<double>& dst) {
    dst.resize(field.size());
    field.copy_to_host(dst.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

TEST_CASE("cuda checkpoint round-trip reproduces an uninterrupted run", "[cuda][checkpoint]") {
    using namespace reticolo;
    constexpr std::uint64_t k_seed = 12345;
    action::Phi4<double> const phi4{.kappa = 0.18, .lambda = 1.0};

    // (1) Straight run of k_n_first + k_n_second trajectories.
    std::vector<double> straight;
    {
        DField field = make_cold_field();
        cuda::Hmc<DAct, updater::integ::Leapfrog, DField> hmc{
            DAct{phi4, field.topology()}, field, 1.0, 10, k_seed};
        hmc.run(k_n_first);
        hmc.run(k_n_second);
        hmc.sync();
        snapshot(field, straight);
    }

    // (2) Run k_n_first, checkpoint (field + seed + Philox counter) to HDF5.
    auto const ckpt = scratch_path("cuda_ckpt");
    std::error_code ec;
    std::filesystem::remove(ckpt, ec);
    {
        DField field = make_cold_field();
        cuda::Hmc<DAct, updater::integ::Leapfrog, DField> hmc{
            DAct{phi4, field.topology()}, field, 1.0, 10, k_seed};
        hmc.run(k_n_first);
        hmc.sync();
        Lattice<double> host{Lattice<double>::SizeVec(k_shape.begin(), k_shape.end())};
        field.copy_to_host(host.data());
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        io::save_config_counter(ckpt, host, hmc.seed(), hmc.rng_counter(), k_n_first);
    }

    // (3) Resume from the checkpoint and run the remaining k_n_second.
    std::vector<double> resumed;
    {
        auto const shape = io::load_field_shape(ckpt);
        Lattice<double> host{Lattice<double>::SizeVec(shape.begin(), shape.end())};
        std::uint64_t seed    = 0;
        std::uint64_t counter = 0;
        (void)io::load_config_counter(ckpt, host, seed, counter);

        DField field{k_shape};
        field.copy_from_host(host.data());
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        cuda::Hmc<DAct, updater::integ::Leapfrog, DField> hmc{
            DAct{phi4, field.topology()}, field, 1.0, 10, seed};
        hmc.set_rng_counter(counter);
        hmc.run(k_n_second);
        hmc.sync();
        snapshot(field, resumed);
    }
    std::filesystem::remove(ckpt, ec);

    REQUIRE(straight.size() == resumed.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < straight.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(straight[i] - resumed[i]));
    }
    REQUIRE(max_diff == 0.0);  // counter-based RNG ⇒ exact resume
}
