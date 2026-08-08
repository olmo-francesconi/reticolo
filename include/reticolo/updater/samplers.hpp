#pragma once

#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>
#include <reticolo/updater/metropolis/metropolis.hpp>

// Sampler tags — "which updater drives this worker", as one template parameter.
//
// Orchestration cannot take the concrete updater as a type argument: the updater
// is parameterised on the ACTION the worker builds internally (an
// `action::WindowedAction<Action, Constraint>` for LLR), which the caller has no
// business spelling. So the parameter is a TAG that names the updater once the
// worker supplies the action:
//
//     orch::llr::Orchestrator<Action, FastRng, updater::HmcSampler<>>     HMC, Omelyan2
//     orch::llr::Orchestrator<Action, FastRng, updater::MetropolisSampler>
//     orch::llr::Orchestrator<Action, FastRng, updater::HmcSampler<integ::Omelyan4>>
//
// The MD integrator rides on the HMC tag rather than being a parameter of its
// own, because it is meaningless for every other sampler — that is exactly the
// knob a Metropolis-driven run would have had to carry and ignore.
//
// `spec_type` is the updater's construction parameters, so a driver can accept
// `Sampler::spec_type` without knowing which updater it named. The optional
// `integrator` alias is how the driver decides whether the ctor takes one; a tag
// without it is constructed without.

namespace reticolo::updater {

template <class Integ = integ::Omelyan2>
struct HmcSampler {
    using spec_type  = HmcSpec;
    using integrator = Integ;

    template <class Action, class Rng>
    using type = Hmc<Action, Rng, Integ>;
};

struct MetropolisSampler {
    using spec_type = MetropolisSpec;

    template <class Action, class Rng>
    using type = Metropolis<Action, Rng>;
};

}  // namespace reticolo::updater
