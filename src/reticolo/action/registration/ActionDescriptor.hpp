/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/registration/ActionDescriptor.hpp

*******************************************************************************/

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/TestAction.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"

namespace reticolo::registration {

enum class ActionPrecisionBinding : std::uint8_t {
    float_precision,
    double_precision,
};

struct ActionManifestView {
    std::string_view                  module_name;
    std::string_view                  default_name;
    std::string_view                  float_name;
    std::string_view                  double_name;
    std::span<const std::string_view> aliases;
    std::span<const std::string_view> algorithms;
    bool                              has_float_precision;
    bool                              has_double_precision;
    bool                              supports_metropolis;
    bool                              supports_hmc;
    bool                              supports_llr;
    ActionPrecisionBinding            canonical_precision;
    ActionPrecisionBinding            alias_precision;
    std::string_view                  storage_schema;

    [[nodiscard]] constexpr auto action_names() const -> std::array<std::string_view, 3> {
        return {default_name, float_name, double_name};
    }

    [[nodiscard]] constexpr auto matches_action_name(std::string_view action_name) const -> bool {
        if (action_name == default_name) {
            return true;
        }

        if (has_float_precision && action_name == float_name) {
            return true;
        }

        if (has_double_precision && action_name == double_name) {
            return true;
        }

        for (const auto alias : aliases) {
            if (action_name == alias) {
                return true;
            }
        }

        return false;
    }
};

struct RelativisticBoseGasDescriptor {
    static constexpr std::string_view module_name = "MonteCarlo";
    static constexpr std::string_view default_name = "RelativisticBoseGas";
    static constexpr std::string_view float_name = "RelativisticBoseGas_F";
    static constexpr std::string_view double_name = "RelativisticBoseGas_D";
    static constexpr auto             aliases = std::array<std::string_view, 0>{};
    static constexpr bool             supports_metropolis = true;
    static constexpr bool             supports_hmc = true;
    static constexpr bool             supports_llr = true;
    static constexpr bool             has_float_precision = true;
    static constexpr bool             has_double_precision = true;
    static constexpr auto             canonical_precision = ActionPrecisionBinding::double_precision;
    static constexpr auto             alias_precision = canonical_precision;
    static constexpr std::string_view storage_schema = "relativistic_bose_gas";
    static constexpr auto algorithms = std::array<std::string_view, 3>{"Metropolis", "HMC", "LLRMetropolis"};
};

struct WeakFieldEuclideanGRDescriptor {
    static constexpr std::string_view module_name = "MonteCarlo";
    static constexpr std::string_view default_name = "WeakFieldEuclideanGR";
    static constexpr std::string_view float_name = "WeakFieldEuclideanGR_F";
    static constexpr std::string_view double_name = "WeakFieldEuclideanGR_D";
    static constexpr auto             aliases = std::array<std::string_view, 0>{};
    static constexpr bool             supports_metropolis = true;
    static constexpr bool             supports_hmc = false;
    static constexpr bool             supports_llr = false;
    static constexpr bool             has_float_precision = true;
    static constexpr bool             has_double_precision = true;
    static constexpr auto             canonical_precision = ActionPrecisionBinding::double_precision;
    static constexpr auto             alias_precision = canonical_precision;
    static constexpr std::string_view storage_schema = "weak_field_euclidean_gr";
    static constexpr auto             algorithms = std::array<std::string_view, 1>{"Metropolis"};
};

struct TestActionDescriptor {
    static constexpr std::string_view module_name = "MonteCarlo";
    static constexpr std::string_view default_name = "TestAction";
    static constexpr std::string_view float_name = "TestAction_F";
    static constexpr std::string_view double_name = "TestAction_D";
    static constexpr auto             aliases = std::array<std::string_view, 0>{};
    // Add aliases above if you want backward-compatible alternate names.
    static constexpr bool             supports_metropolis = true;
    static constexpr bool             supports_hmc = true;
    static constexpr bool             supports_llr = false;
    static constexpr bool             has_float_precision = true;
    static constexpr bool             has_double_precision = true;
    static constexpr auto             canonical_precision = ActionPrecisionBinding::double_precision;
    static constexpr auto             alias_precision = canonical_precision;
    static constexpr std::string_view storage_schema = "test_action";
    static constexpr auto algorithms = std::array<std::string_view, 2>{"Metropolis", "HMC"};
};
template <typename Action>
struct ActionDescriptorFor;

template <>
struct ActionDescriptorFor<action::RelativisticBoseGas<RealF>> {
    using type = RelativisticBoseGasDescriptor;
};

template <>
struct ActionDescriptorFor<action::RelativisticBoseGas<RealD>> {
    using type = RelativisticBoseGasDescriptor;
};

template <>
struct ActionDescriptorFor<action::WeakFieldEuclideanGR<RealF>> {
    using type = WeakFieldEuclideanGRDescriptor;
};

template <>
struct ActionDescriptorFor<action::WeakFieldEuclideanGR<RealD>> {
    using type = WeakFieldEuclideanGRDescriptor;
};

template <>
struct ActionDescriptorFor<action::TestAction<RealF>> {
    using type = TestActionDescriptor;
};

template <>
struct ActionDescriptorFor<action::TestAction<RealD>> {
    using type = TestActionDescriptor;
};
template <typename Action>
using action_descriptor_t = ActionDescriptorFor<Action>::type;

template <typename Descriptor>
[[nodiscard]] constexpr auto descriptor_manifest() -> ActionManifestView {
    return ActionManifestView{
        .module_name = Descriptor::module_name,
        .default_name = Descriptor::default_name,
        .float_name = Descriptor::float_name,
        .double_name = Descriptor::double_name,
        .aliases = std::span<const std::string_view>(Descriptor::aliases),
        .algorithms = std::span<const std::string_view>(Descriptor::algorithms),
        .has_float_precision = Descriptor::has_float_precision,
        .has_double_precision = Descriptor::has_double_precision,
        .supports_metropolis = Descriptor::supports_metropolis,
        .supports_hmc = Descriptor::supports_hmc,
        .supports_llr = Descriptor::supports_llr,
        .canonical_precision = Descriptor::canonical_precision,
        .alias_precision = Descriptor::alias_precision,
        .storage_schema = Descriptor::storage_schema,
    };
}

template <typename Action>
[[nodiscard]] constexpr auto action_manifest() -> ActionManifestView {
    return descriptor_manifest<action_descriptor_t<Action>>();
}

template <typename Action>
[[nodiscard]] constexpr auto available_algorithms() {
    return action_manifest<Action>().algorithms;
}

template <typename Action>
[[nodiscard]] inline auto available_algorithm_names() -> std::vector<std::string_view> {
    const auto Algorithms = available_algorithms<Action>();
    return std::vector<std::string_view>(Algorithms.begin(), Algorithms.end());
}

}  // namespace reticolo::registration
