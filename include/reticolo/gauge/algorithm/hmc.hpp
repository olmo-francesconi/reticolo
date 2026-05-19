#pragma once

// Transitional shim: the gauge HMC and integrators have been unified into
// `reticolo::alg::Hmc<A, R, Integrator, LinkLattice<F>, F>` and
// `reticolo::alg::integ::*`. This header re-exports them under the
// `reticolo::gauge::alg` namespace so existing apps (`u1_hmc`, `u1_llr`,
// `bench_gauge_vs_scalar`) and tests keep compiling unchanged. To be
// deleted once those call sites switch to the unqualified `alg::` names.

#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/link_lattice.hpp>

namespace reticolo::gauge::alg {

namespace integ = reticolo::alg::integ;

using HmcSpec = reticolo::alg::HmcSpec;
using HmcStep = reticolo::alg::HmcStep;

template <class A,
          class R,
          class Integrator = reticolo::alg::integ::Leapfrog,
          class F          = typename A::value_type>
using Hmc = reticolo::alg::Hmc<A, R, Integrator, LinkLattice<F>, F>;

}  // namespace reticolo::gauge::alg
