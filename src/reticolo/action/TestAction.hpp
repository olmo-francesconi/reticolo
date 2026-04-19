#pragma once

#include <complex>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>

#include "reticolo/action/actionBase.hpp"
#include "reticolo/core/types/coord.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::action {

template <RealValue TImpl>
class TestAction : public ActionBase<std::complex<TImpl>, TImpl, TImpl> {
  public:
    using base_type = ActionBase<std::complex<TImpl>, TImpl, TImpl>;
    using action_type = typename base_type::action_type;
    using field_type = typename base_type::field_type;
    using size_type = typename base_type::size_type;
    using impl_type = TImpl;

    static constexpr int Dims = 4;
    static constexpr int Stencil = 2;

    struct Params {
        // IMPLEMENTATION REQUIRED:
        // Replace these placeholder parameters with the real couplings for your action.
        impl_type coupling = static_cast<impl_type>(1.0);
        impl_type mass = static_cast<impl_type>(1.0);
    } p;

    struct Observables {
        impl_type a;
        impl_type b;

        void reset() {
            a = 0;
            b = 0;
        }

        auto operator+=(const Observables& rhs) -> Observables& {
            a += rhs.a;
            b += rhs.b;
            return *this;
        }

        auto operator/=(const impl_type& rhs) -> Observables& {
            a /= rhs;
            b /= rhs;
            return *this;
        }
    };

    TestAction() = default;

    void setup(const YAML::Node& ActionParams) override;
    void lattice_sync(Lattice<field_type>& field) override;

    auto compute_S(Lattice<field_type>& field) -> action_type override;
    auto compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type override;
    auto compute_dS_loc(Lattice<field_type>& field, const field_type& dphi, size_type site) -> action_type override;

    void compute_Forces(Lattice<field_type>& field, Lattice<field_type>& forces) override;
    void compute_LLRForces(Lattice<field_type>& field, Lattice<field_type>& forces, TImpl Sk, TImpl width,
                           TImpl ak) override;

    static auto Measure(Lattice<field_type>& field) -> Observables;

    auto GetName() -> std::string override;
    auto GetParameters() -> std::string override;
};

template <RealValue TImpl>
inline void TestAction<TImpl>::setup(const YAML::Node& ActionParams) {
    // IMPLEMENTATION REQUIRED:
    // Parse your action-specific YAML fields here.
    if (ActionParams["coupling"]) {
        p.coupling = ActionParams["coupling"].as<impl_type>();
    }
    if (ActionParams["mass"]) {
        p.mass = ActionParams["mass"].as<impl_type>();
    }
}

template <RealValue TImpl>
inline void TestAction<TImpl>::lattice_sync(Lattice<field_type>& /*field*/) {
    // IMPLEMENTATION REQUIRED:
    // Precompute any lattice-dependent caches or consistency checks here.
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::compute_S(Lattice<field_type>& /*field*/) -> action_type {
    // IMPLEMENTATION REQUIRED:
    // Return the total action for the current lattice configuration.
    return action_type{0};
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::compute_S_loc(Lattice<field_type>& /*field*/, size_type /*site*/) -> action_type {
    // IMPLEMENTATION REQUIRED:
    // Return the contribution local to a lattice site if your update algorithm needs it.
    return action_type{0};
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::compute_dS_loc(Lattice<field_type>& /*field*/, const field_type& /*dphi*/, size_type /*site*/)
    -> action_type {
    // IMPLEMENTATION REQUIRED:
    // Return the local action variation associated with a proposed field update.
    return action_type{0};
}

template <RealValue TImpl>
inline void TestAction<TImpl>::compute_Forces(Lattice<field_type>& /*field*/, Lattice<field_type>& /*forces*/) {
    // IMPLEMENTATION REQUIRED:
    // Fill the force lattice if the action supports the generic HMC implementation.
    throw std::runtime_error("TestAction::compute_Forces is not implemented");
}

template <RealValue TImpl>
inline void TestAction<TImpl>::compute_LLRForces(Lattice<field_type>& /*field*/, Lattice<field_type>& /*forces*/,
                                             TImpl /*Sk*/, TImpl /*width*/, TImpl /*ak*/) {
    // IMPLEMENTATION REQUIRED:
    // Fill the force lattice for LLR/HMC-style updates if you plan to support them.
    throw std::runtime_error("TestAction::compute_LLRForces is not implemented");
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::Measure(Lattice<field_type>& /*field*/) -> Observables {
    // IMPLEMENTATION REQUIRED:
    // Populate the observable values that will be written to HDF5.
    Observables Result{};
    Result.a = 0;
    Result.b = 0;
    return Result;
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::GetName() -> std::string {
    std::string Result = "TestAction";
    Result += std::same_as<impl_type, RealF> ? " [float]" : " [double]";
    return Result;
}

template <RealValue TImpl>
inline auto TestAction<TImpl>::GetParameters() -> std::string {
    // IMPLEMENTATION REQUIRED:
    // Replace this with a concise summary of the action parameters.
    return std::format("[ coupling : {:.3f}, mass : {:.3f} ]", p.coupling, p.mass);
}

}  // namespace reticolo::action
