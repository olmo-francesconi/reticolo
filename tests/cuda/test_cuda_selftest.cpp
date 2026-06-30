#include <reticolo/core/indexing.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/probes/bose_gas_probe.hpp>
#include <reticolo/cuda/probes/f32_probe.hpp>
#include <reticolo/cuda/probes/gauge_probe.hpp>
#include <reticolo/cuda/probes/hmc_probe.hpp>
#include <reticolo/cuda/probes/phi4_probe.hpp>
#include <reticolo/cuda/probes/rng_probe.hpp>
#include <reticolo/cuda/probes/scalar_probe.hpp>
#include <reticolo/cuda/probes/selftest.hpp>
#include <reticolo/cuda/probes/stencil_probe.hpp>
#include <reticolo/cuda/probes/su2_probe.hpp>
#include <reticolo/cuda/probes/su3_probe.hpp>
#include <reticolo/cuda/probes/u1_probe.hpp>
#include <reticolo/cuda/probes/wilson_u1_probe.hpp>
#include <reticolo/cuda/reduce.hpp>

#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// CUDA tests. Only registered when RETICOLO_ENABLE_CUDA is on (see
// tests/CMakeLists.txt); all require a GPU at run time.

// Phase 0 exit gate: the toolchain builds, a kernel launches, and a
// host→device→host round-trip preserves data.
TEST_CASE("cuda backend self-test round-trips through the device", "[cuda]") {
    REQUIRE(reticolo::cuda::selftest());
}

// Phase 1: the device's closed-form periodic indexing must reproduce the CPU
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

// Phase 1: deterministic f64 reduction + axpy on the device.
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

    // reduce: Σ x
    double const dev_sum = reduce_sum_f64(dx.data(), n);
    REQUIRE(dev_sum == Catch::Approx(host_sum).epsilon(1e-12));

    // determinism: same inputs → identical bits across calls
    REQUIRE(reduce_sum_f64(dx.data(), n) == dev_sum);

    // axpy: y += 2*x, then reduce y == Σ(1 + 2x) = n + 2·Σx
    axpy_f64(2.0, dx.data(), dy.data(), n);
    double const dev_y_sum = reduce_sum_f64(dy.data(), n);
    REQUIRE(dev_y_sum == Catch::Approx(static_cast<double>(n) + 2.0 * host_sum).epsilon(1e-12));

    // Device-scalar reductions (the HMC hot-loop path): match the host values
    // and are deterministic. partials must hold k_reduce_max_grid doubles.
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

// Phase 1 (M1): the gauge action/drift headers compile under nvcc (Sleef now
// guarded out of vec_libm.hpp) and produce finite output. That this links at
// all means src/cuda/gauge_probe.cu compiled the transcendental paths.
TEST_CASE("cuda gauge headers compile and run under nvcc", "[cuda]") {
    REQUIRE(reticolo::cuda::gauge_headers_compile());
}

// Phase 1 exit gate: the generic stencil + reduce_fwd device skeletons, driven
// by a dummy Phi4-shaped functor pair, satisfy force == -dS/dphi by central
// finite differences. Validates the scalar device action protocol end-to-end.
TEST_CASE("cuda stencil force matches finite-difference of reduce_fwd action", "[cuda]") {
    REQUIRE(reticolo::cuda::stencil_force_matches_fd());
}

// Phase 2a: a scalar field survives the host↔device round-trip unchanged.
TEST_CASE("cuda DeviceField round-trips a scalar field", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_roundtrip_ok());
}

// Phase 2a: the generic DeviceAction over the real Phi4 functor pair
// reproduces the CPU action::Phi4 s_full and force to roundoff — the shared
// HD formula is genuinely one source of truth across CPU and device.
TEST_CASE("cuda DeviceAction<Phi4> matches CPU action::Phi4 to roundoff", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_cpu_matches_device());
}

// Phase 2b: Leapfrog MD on DeviceField is time-reversible to roundoff.
TEST_CASE("cuda HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_reversibility_ok());
}

// Phase 2b: the reused alg::integ tags give |ΔH| ~ dt^p with p = 2/2/4 on the
// device — the integrator-genericity proof.
TEST_CASE("cuda integrator order is 2/2/4", "[cuda]") {
    REQUIRE(reticolo::cuda::integrator_order_ok());
}

// Phase 2b: cuda::Hmc::step() (sample → MD → ΔH → host MH) runs and stays finite.
TEST_CASE("cuda Hmc step runs end-to-end", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_step_runs());
}

// Phase 2c: device Philox uniforms are bit-identical to the host primitive.
TEST_CASE("cuda Philox device matches host bit-for-bit", "[cuda]") {
    REQUIRE(reticolo::cuda::philox_host_matches_device());
}

// Phase 2c: advancing the trajectory counter changes momenta; same counter repeats.
TEST_CASE("cuda Philox trajectory counter advances the stream", "[cuda]") {
    REQUIRE(reticolo::cuda::philox_traj_distinct());
}

// Phase 2c: a large device fill is ~N(0,1).
TEST_CASE("cuda Philox normals have N(0,1) moments", "[cuda]") {
    REQUIRE(reticolo::cuda::philox_moments_ok());
}

// Phase 2d: a graph-captured MD trajectory + its replay reproduce eager MD
// bit-for-bit from the same (q0, p0).
TEST_CASE("cuda graph replay matches eager MD", "[cuda]") {
    REQUIRE(reticolo::cuda::graph_replay_matches_eager());
}

// Phase 2e: host-free trajectory streaming (device-side Metropolis accept, no
// per-step sync) is deterministic and produces a sane chain.
TEST_CASE("cuda host-free HMC run is deterministic", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_device_run_deterministic());
}

// Phase 3: each new scalar action's device path reproduces the CPU action's
// s_full + force via the shared HD formula (one source of truth).
TEST_CASE("cuda DeviceAction<Phi6> matches CPU action::Phi6", "[cuda]") {
    REQUIRE(reticolo::cuda::phi6_cpu_matches_device());
}

TEST_CASE("cuda DeviceAction<SineGordon> matches CPU action::SineGordon", "[cuda]") {
    REQUIRE(reticolo::cuda::sine_gordon_cpu_matches_device());
}

TEST_CASE("cuda DeviceAction<Xy> matches CPU action::Xy", "[cuda]") {
    REQUIRE(reticolo::cuda::xy_cpu_matches_device());
}

// Phase 3d: the device HMC stack in single precision — DeviceAction<Phi4<float>>
// matches the CPU f32 action, and f32 Leapfrog MD is reversible (bounded tol).
TEST_CASE("cuda DeviceAction<Phi4<float>> matches CPU to f32 tolerance", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_f32_cpu_matches_device());
}

TEST_CASE("cuda f32 HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::hmc_f32_reversibility_ok());
}

// Phase 4: compact U(1) gauge on the device through the SAME unified
// DeviceAction — the gauge access pattern (per-link gather force, per-site
// plaquette action) lives only in the device_functors<CompactU1> trait.
TEST_CASE("cuda DeviceAction<CompactU1> matches CPU action::CompactU1", "[cuda]") {
    REQUIRE(reticolo::cuda::u1_cpu_matches_device());
}

TEST_CASE("cuda U(1) gather force matches finite-difference of s_full", "[cuda]") {
    REQUIRE(reticolo::cuda::u1_force_matches_fd());
}

TEST_CASE("cuda U(1) HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::u1_hmc_reversibility_ok());
}

TEST_CASE("cuda U(1) host-free HMC runs on the link field", "[cuda]") {
    REQUIRE(reticolo::cuda::u1_hmc_runs());
}

// Phase 5: SU(2) Wilson gauge on the device through the SAME unified
// DeviceAction — the matrix access pattern (staple-gather force, plaquette
// action, group-exp drift, Gell-Mann momentum) lives only in the
// device_functors<Wilson<SU2>> trait, the MatrixLayout drift atom, and the
// SU2Device register-local ops.
TEST_CASE("cuda SU2Device matrix ops match math::su2", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_device_ops_match_cpu());
}

TEST_CASE("cuda DeviceAction<Wilson<SU2>> matches CPU action::Wilson<SU2>", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_cpu_matches_device());
}

TEST_CASE("cuda SU2 Leapfrog MD conserves energy (2nd order)", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_energy_conserved_ok());
}

TEST_CASE("cuda SU2 HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_hmc_reversibility_ok());
}

TEST_CASE("cuda SU2 Gell-Mann momenta have the right moments", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_momentum_moments_ok());
}

TEST_CASE("cuda SU2 host-free HMC runs on the matrix link field", "[cuda]") {
    REQUIRE(reticolo::cuda::su2_hmc_runs());
}

// Phase 5: SU(3) Wilson gauge — the same generic SU(N) kernels as SU(2) with
// GD = SU3Device (nc=18, n_gen=8, Morningstar-Peardon 3×3 group exp). Nothing in
// DeviceAction / Hmc / the kernels changed — only the device-traits struct.
TEST_CASE("cuda SU3Device matrix ops match math::su3", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_device_ops_match_cpu());
}

TEST_CASE("cuda DeviceAction<Wilson<SU3>> matches CPU action::Wilson<SU3>", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_cpu_matches_device());
}

TEST_CASE("cuda SU3 Leapfrog MD conserves energy (2nd order)", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_energy_conserved_ok());
}

TEST_CASE("cuda SU3 HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_hmc_reversibility_ok());
}

TEST_CASE("cuda SU3 Gell-Mann momenta have the right moments", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_momentum_moments_ok());
}

TEST_CASE("cuda SU3 host-free HMC runs on the matrix link field", "[cuda]") {
    REQUIRE(reticolo::cuda::su3_hmc_runs());
}

// Wilson<U(1)> on the device via the SPECIALIZED abelian path: it reuses the
// CompactU1 angle kernels (gauge_u1.cuh) on a 1-angle LinkLayout field — NOT the
// generic SU(N) matrix kernels. Wilson<U1> and CompactU1 are bit-identical
// (n_color=1 ⇒ β/N = β), so this is the same device path with the action type
// swapped, demonstrating the M8 "Wilson<U1> subsumes CompactU1" unification.
TEST_CASE("cuda DeviceAction<Wilson<U1>> matches CPU action::Wilson<U1>", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_cpu_matches_device());
}

TEST_CASE("cuda Wilson<U1> HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_hmc_reversibility_ok());
}

TEST_CASE("cuda Wilson<U1> host-free HMC runs on the link field", "[cuda]") {
    REQUIRE(reticolo::cuda::wilson_u1_hmc_runs());
}

// BoseGas — complex scalar (relativistic Bose gas at finite μ). HMC samples the
// phase-quenched S_R; the device field element is cplx<T> (AoS, flat-copy with
// the host Lattice<std::complex<T>>), the per-site S_R/F_R formula shared with
// the CPU action. Validated in f64 and f32.
TEST_CASE("cuda DeviceAction<BoseGas<double>> matches CPU action::BoseGas", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_cpu_matches_device_f64());
}

TEST_CASE("cuda DeviceAction<BoseGas<float>> matches CPU to f32 tolerance", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_cpu_matches_device_f32());
}

TEST_CASE("cuda BoseGas HMC trajectory is reversible", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_hmc_reversibility_ok());
}

TEST_CASE("cuda BoseGas host-free HMC runs (f64)", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_hmc_runs_f64());
}

TEST_CASE("cuda BoseGas host-free HMC runs (f32)", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_hmc_runs_f32());
}
