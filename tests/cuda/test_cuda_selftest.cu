#include <reticolo/core/indexing.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/pinned.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace reticolo::cuda {

namespace {

__global__ void scale_kernel(double* d, std::size_t n, double a) {
    std::size_t const i = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n) {
        d[i] *= a;
    }
}

}  // namespace

bool selftest() {
    constexpr std::size_t n = 1024;
    constexpr double scale  = 2.0;

    PinnedBuffer<double> host{n};
    PinnedBuffer<double> back{n};
    for (std::size_t i = 0; i < n; ++i) {
        host.data()[i] = static_cast<double>(i);
    }

    DeviceBuffer<double> dev{n};
    dev.copy_from_host(host.data());

    constexpr unsigned block = 256;
    unsigned const grid      = static_cast<unsigned>((n + block - 1) / block);
    scale_kernel<<<grid, block>>>(dev.data(), n, scale);
    RETICOLO_CUDA_CHECK_LAUNCH();

    dev.copy_to_host(back.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < n; ++i) {
        if (back.data()[i] != scale * static_cast<double>(i)) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda

// The toolchain builds, a kernel launches, and a
// host->device->host round-trip preserves data.
TEST_CASE("cuda backend self-test round-trips through the device", "[cuda]") {
    REQUIRE(reticolo::cuda::selftest());
}

// The device's closed-form periodic indexing must reproduce the CPU
// reference neighbour table exactly — checked on the host (next/prev are
// __host__ __device__). Covers a 4D cube and a non-cube, non-power-of-two shape.
TEST_CASE("cuda DeviceTopology matches the reference Indexing", "[cuda]") {
    for (std::vector<std::size_t> shape :
         {std::vector<std::size_t>{4, 4, 4, 4}, std::vector<std::size_t>{6, 4, 5}}) {
        auto const idx  = reticolo::Indexing::acquire(shape);
        auto const topo = reticolo::cuda::make_device_topology(shape);
        REQUIRE(topo.nsites == static_cast<long>(idx->nsites()));

        for (std::size_t s = 0; s < idx->nsites(); ++s) {
            for (std::size_t mu = 0; mu < shape.size(); ++mu) {
                REQUIRE(topo.next(static_cast<long>(s), static_cast<int>(mu)) ==
                        static_cast<long>(idx->next(reticolo::Site{s}, mu).value()));
                REQUIRE(topo.prev(static_cast<long>(s), static_cast<int>(mu)) ==
                        static_cast<long>(idx->prev(reticolo::Site{s}, mu).value()));
            }
        }
    }
}

// Deterministic f64 reduction + axpy on the device.
TEST_CASE("cuda reduce_sum_f64 and axpy_f64 are correct", "[cuda]") {
    using namespace reticolo::cuda;
    constexpr long n = 10000;

    std::vector<double> hx(n);
    std::vector<double> hy(n);
    double host_sum = 0.0;
    for (long i = 0; i < n; ++i) {
        hx[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i) - 3.0;
        hy[static_cast<std::size_t>(i)] = 1.0;
        host_sum += hx[static_cast<std::size_t>(i)];
    }

    DeviceBuffer<double> dx{static_cast<std::size_t>(n)};
    DeviceBuffer<double> dy{static_cast<std::size_t>(n)};
    dx.copy_from_host(hx.data());
    dy.copy_from_host(hy.data());

    double const dev_sum = reduce_sum_f64(dx.data(), n);
    REQUIRE(dev_sum == Catch::Approx(host_sum).epsilon(1e-12));

    REQUIRE(reduce_sum_f64(dx.data(), n) == dev_sum);

    axpy_f64(2.0, dx.data(), dy.data(), n);
    double const dev_y_sum = reduce_sum_f64(dy.data(), n);
    REQUIRE(dev_y_sum == Catch::Approx(static_cast<double>(n) + 2.0 * host_sum).epsilon(1e-12));

    double host_sumsq = 0.0;
    for (long i = 0; i < n; ++i) {
        host_sumsq += hx[static_cast<std::size_t>(i)] * hx[static_cast<std::size_t>(i)];
    }
    DeviceBuffer<double> partials{static_cast<std::size_t>(k_reduce_max_grid)};
    DeviceBuffer<double> out{2};
    reduce_sum_into(out.data() + 0, dx.data(), n, partials.data());
    reduce_sumsq_into(out.data() + 1, dx.data(), n, partials.data());
    std::array<double, 2> h_out{};
    out.copy_to_host(h_out.data());
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(h_out[0] == Catch::Approx(host_sum).epsilon(1e-12));
    REQUIRE(h_out[1] == Catch::Approx(host_sumsq).epsilon(1e-12));
}
