#include <reticolo/action/complex/bose_gas.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/cplx.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/cuda/actions/complex/bose_gas.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/llr/windowed_action.cuh>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/llr/windowed_action.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// BoseGas complex-scalar HMC on the device. The field element is cplx<T> (AoS,
// 2 reals/site), so the host Lattice<std::complex<T>> flat-copies in via a
// reinterpret (layout-compatible). device_functors<BoseGas<T>> shares the S_R/F_R
// per-site formula with the CPU action; the complex HMC atoms (drift/kick/kinetic
// over 2·n reals) live in cuda/actions/complex/bose_gas.hpp. (Excluded from the
// no-integrator-kernels lint gate: it names alg::integ::Leapfrog.)

namespace reticolo::cuda {

namespace {

std::vector<std::size_t> const kShape{4, 4, 4, 4};

template <class T>
struct Params {
    static constexpr T mass   = T{1};
    static constexpr T lambda = T{0.5};
    static constexpr T mu     = T{0.3};  // nonzero ⇒ exercises the cosh(μ) time weight
};

// Deterministic non-trivial complex config in the host AoS layout.
template <class T>
Lattice<std::complex<T>> make_field() {
    Lattice<std::complex<T>> l{Indexing::SizeVec(kShape.begin(), kShape.end())};
    std::complex<T>* const d = l.data();
    std::size_t const ns     = l.nsites();
    for (std::size_t i = 0; i < ns; ++i) {
        auto const x = static_cast<T>(i);
        d[i] = std::complex<T>(static_cast<T>(0.5) * std::sin((static_cast<T>(0.3) * x) + T{1}),
                               static_cast<T>(0.4) * std::cos(static_cast<T>(0.2) * x));
    }
    return l;
}

template <class T>
action::BoseGas<T> make_action() {
    return action::BoseGas<T>{
        .mass = Params<T>::mass, .lambda = Params<T>::lambda, .mu = Params<T>::mu};
}

template <class T>
cplx<T> const* as_dev(std::complex<T> const* p) {
    return reinterpret_cast<cplx<T> const*>(p);
}
template <class T>
cplx<T>* as_dev(std::complex<T>* p) {
    return reinterpret_cast<cplx<T>*>(p);
}

template <class T>
bool cpu_matches_device_impl(double s_tol, double f_tol) {
    using DField = DeviceField<cplx<T>>;
    using DAct   = DeviceAction<action::BoseGas<T>, DField>;

    Lattice<std::complex<T>> const host = make_field<T>();
    action::BoseGas<T> const a          = make_action<T>();

    double const s_cpu = a.s_full(host);
    Lattice<std::complex<T>> f_cpu{host.indexing()};
    a.compute_force(host, f_cpu);

    DField dfield{kShape};
    DField dforce{dfield.topology()};
    dfield.copy_from_host(as_dev(host.data()));

    DAct const act{a, dfield.topology()};
    double const s_dev = act.s_full(dfield);
    act.compute_force(dfield, dforce);

    std::vector<std::complex<T>> f_dev(dforce.size());
    dforce.copy_to_host(as_dev(f_dev.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > s_tol * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < f_dev.size(); ++i) {
        std::complex<T> const a_cpu = f_cpu.data()[i];
        if (std::abs(static_cast<double>(a_cpu.real() - f_dev[i].real())) >
                f_tol * (1.0 + std::abs(static_cast<double>(a_cpu.real()))) ||
            std::abs(static_cast<double>(a_cpu.imag() - f_dev[i].imag())) >
                f_tol * (1.0 + std::abs(static_cast<double>(a_cpu.imag())))) {
            return false;
        }
    }
    return true;
}

template <class T>
bool hmc_runs_impl() {
    using DField = DeviceField<cplx<T>>;
    using DAct   = DeviceAction<action::BoseGas<T>, DField>;

    Lattice<std::complex<T>> const host = make_field<T>();
    DField field{kShape};
    field.copy_from_host(as_dev(host.data()));
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    DAct dact{make_action<T>(), field.topology()};
    Hmc<DAct, alg::integ::Leapfrog, DField> hmc{std::move(dact), field, 0.5, 10};
    hmc.run(8);
    double const acc = hmc.acceptance();
    hmc.sync();

    std::vector<std::complex<T>> out(field.size());
    field.copy_to_host(as_dev(out.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (acc <= 0.0 || acc > 1.0) {
        return false;
    }
    for (auto const& z : out) {
        if (!std::isfinite(static_cast<double>(z.real())) ||
            !std::isfinite(static_cast<double>(z.imag()))) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool bose_gas_cpu_matches_device_f64() {
    return cpu_matches_device_impl<double>(1e-9, 1e-9);
}
bool bose_gas_cpu_matches_device_f32() {
    return cpu_matches_device_impl<float>(2e-4, 2e-3);
}

bool bose_gas_hmc_reversibility_ok() {
    using DField = DeviceField<cplx<double>>;
    using DAct   = DeviceAction<action::BoseGas<double>, DField>;

    Lattice<std::complex<double>> const host = make_field<double>();

    DField field{kShape};
    DField mom{field.topology()};
    DField force{field.topology()};
    DAct const act{make_action<double>(), field.topology()};

    field.copy_from_host(as_dev(host.data()));

    DeviceBuffer<std::uint64_t> traj{1};
    std::uint64_t const zero = 0;
    traj.copy_from_host(&zero, nullptr);
    act.sample_momenta(mom.data(), static_cast<long>(mom.size()), 0xB05EULL, traj.data(), nullptr);
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    std::vector<std::complex<double>> f0(field.size());
    field.copy_to_host(as_dev(f0.data()));
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));

    using Integ          = alg::integ::Leapfrog;
    constexpr double tau = 0.5;
    constexpr int n_md   = 10;

    Integ::run(act, field, mom, force, tau, n_md);
    // p -> -p over the 2·size underlying reals.
    axpy_f64(-2.0,
             reinterpret_cast<double const*>(mom.data()),
             reinterpret_cast<double*>(mom.data()),
             2 * static_cast<long>(field.size()));
    Integ::run(act, field, mom, force, tau, n_md);

    std::vector<std::complex<double>> f1(field.size());
    field.copy_to_host(as_dev(f1.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < f1.size(); ++i) {
        if (std::abs(f1[i].real() - f0[i].real()) > 1e-9 ||
            std::abs(f1[i].imag() - f0[i].imag()) > 1e-9) {
            return false;
        }
    }
    return true;
}

bool bose_gas_hmc_runs_f64() {
    return hmc_runs_impl<double>();
}
bool bose_gas_hmc_runs_f32() {
    return hmc_runs_impl<float>();
}

// Device S_I vs CPU action::BoseGas::s_imag — both call the shared imag formula.
bool bose_gas_s_imag_matches_cpu() {
    Lattice<std::complex<double>> const host = make_field<double>();
    action::BoseGas<double> const a          = make_action<double>();

    double const s_cpu = a.s_imag(host);

    DeviceField<cplx<double>> dfield{kShape};
    dfield.copy_from_host(as_dev(host.data()));
    DeviceAction<action::BoseGas<double>, DeviceField<cplx<double>>> const act{a, dfield.topology()};
    double const s_dev = act.s_imag(dfield);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    return std::abs(s_cpu - s_dev) <= 1e-9 * (1.0 + std::abs(s_cpu));
}

// Device F_I vs CPU compute_force_imag.
bool bose_gas_force_imag_matches_cpu() {
    Lattice<std::complex<double>> const host = make_field<double>();
    action::BoseGas<double> const a          = make_action<double>();

    Lattice<std::complex<double>> f_cpu{host.indexing()};
    a.compute_force_imag(host, f_cpu);

    DeviceField<cplx<double>> dfield{kShape};
    DeviceField<cplx<double>> dforce{dfield.topology()};
    dfield.copy_from_host(as_dev(host.data()));
    DeviceAction<action::BoseGas<double>, DeviceField<cplx<double>>> const act{a, dfield.topology()};
    act.compute_force_imag(dfield, dforce);

    std::vector<std::complex<double>> f_dev(dforce.size());
    dforce.copy_to_host(as_dev(f_dev.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < f_dev.size(); ++i) {
        std::complex<double> const c = f_cpu.data()[i];
        if (std::abs(c.real() - f_dev[i].real()) > 1e-9 * (1.0 + std::abs(c.real())) ||
            std::abs(c.imag() - f_dev[i].imag()) > 1e-9 * (1.0 + std::abs(c.imag()))) {
            return false;
        }
    }
    return true;
}

// The whole mode-B windowed path: device cuda::llr::WindowedAction<BoseGas> must
// reproduce the CPU llr::WindowedAction<BoseGas> for both the windowed action
// S_LLR = S_R + a·S_I + (S_I−E_n)²/2δ² and the combined force F_R + scale·F_I.
// Templated so the same check runs in f64 (roundoff) and f32 (device sin/cos +
// the f32 field precision, looser tolerance).
template <class T>
bool bose_gas_windowed_modeB_matches_cpu_impl(double s_tol, double f_tol) {
    Lattice<std::complex<T>> const host = make_field<T>();
    action::BoseGas<T> const a          = make_action<T>();
    constexpr double a_slope = 0.3;
    constexpr double e_n     = 1.5;
    constexpr double delta   = 4.0;

    // CPU windowed (mode B via HasImagPart).
    reticolo::llr::WindowedAction<action::BoseGas<T>> w{
        a, static_cast<T>(a_slope), static_cast<T>(e_n), static_cast<T>(delta)};
    static_assert(decltype(w)::k_complex, "BoseGas CPU WindowedAction must be mode B");
    double const s_cpu = w.s_full(host);
    Lattice<std::complex<T>> f_cpu{host.indexing()};
    w.compute_force(host, f_cpu);

    // Device windowed.
    using DField = DeviceField<cplx<T>>;
    DField dfield{kShape};
    dfield.copy_from_host(as_dev(host.data()));
    llr::WindowedAction<action::BoseGas<T>, DField> dw{a, dfield.topology(), a_slope, e_n, delta};
    static_assert(decltype(dw)::k_complex, "BoseGas device WindowedAction must be mode B");

    DeviceBuffer<double> s_dev_buf{1};
    DeviceBuffer<double> partials{static_cast<std::size_t>(k_reduce_max_grid)};
    dw.s_full_into(s_dev_buf.data(), dfield, partials.data(), nullptr);
    DField dforce{dfield.topology()};
    dw.compute_force(dfield, dforce);

    double s_dev = 0.0;
    s_dev_buf.copy_to_host(&s_dev);
    std::vector<std::complex<T>> f_dev(dforce.size());
    dforce.copy_to_host(as_dev(f_dev.data()));
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > s_tol * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < f_dev.size(); ++i) {
        double const cr = static_cast<double>(f_cpu.data()[i].real());
        double const ci = static_cast<double>(f_cpu.data()[i].imag());
        if (std::abs(cr - static_cast<double>(f_dev[i].real())) > f_tol * (1.0 + std::abs(cr)) ||
            std::abs(ci - static_cast<double>(f_dev[i].imag())) > f_tol * (1.0 + std::abs(ci))) {
            return false;
        }
    }
    return true;
}

bool bose_gas_windowed_modeB_matches_cpu() {
    return bose_gas_windowed_modeB_matches_cpu_impl<double>(1e-8, 1e-8);
}
bool bose_gas_windowed_modeB_matches_cpu_f32() {
    return bose_gas_windowed_modeB_matches_cpu_impl<float>(2e-4, 2e-3);
}

}  // namespace reticolo::cuda

// BoseGas — complex scalar (relativistic Bose gas at finite mu). Validated in
// f64 and f32; the per-site S_R/F_R formula is shared with the CPU action.
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

// Imaginary part: device S_I / F_I share the formula header with the CPU action.
TEST_CASE("cuda BoseGas S_I matches CPU s_imag", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_s_imag_matches_cpu());
}

TEST_CASE("cuda BoseGas F_I matches CPU compute_force_imag", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_force_imag_matches_cpu());
}

// The complex-LLR (mode B) windowed action + combined force, device vs CPU. The
// f32 case also exercises the fused F_I+S_I path in field precision.
TEST_CASE("cuda BoseGas mode-B WindowedAction matches CPU", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_windowed_modeB_matches_cpu());
}

TEST_CASE("cuda BoseGas mode-B WindowedAction matches CPU (f32)", "[cuda]") {
    REQUIRE(reticolo::cuda::bose_gas_windowed_modeB_matches_cpu_f32());
}
