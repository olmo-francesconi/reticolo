#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/TestAction.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/modules/factory/MCAlgorithmFactory.hpp"
#include "reticolo/modules/factory/ModuleFactory.hpp"
#include "reticolo/modules/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/runtime/BuiltinMetadata.hpp"

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "TEST FAILED: %s\n", msg);
        std::abort();
    }
}

template <typename T>
bool contains(const std::vector<T>& values, const T& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

void require_contains(const std::vector<std::string_view>& values, std::string_view needle, const char* msg) {
    require(contains(values, needle), msg);
}

void require_message_contains(const std::runtime_error& err, std::string_view text, const char* msg) {
    require(std::string_view(err.what()).find(text) != std::string_view::npos, msg);
}

}  // namespace

auto main() -> int {
    using Rbg = reticolo::action::RelativisticBoseGas<reticolo::RealD>;
    using RbgHandler = reticolo::MMonteCarlo::MonteCarloHandler<Rbg>;
    using Wfgr = reticolo::action::WeakFieldEuclideanGR<reticolo::RealD>;
    using WfgrHandler = reticolo::MMonteCarlo::MonteCarloHandler<Wfgr>;


    using TestActionAction = reticolo::action::TestAction<reticolo::RealD>;
    using TestActionHandler = reticolo::MMonteCarlo::MonteCarloHandler<TestActionAction>;
    try {
        const auto modules = reticolo::runtime::metadata::available_modules();
        require_contains(modules, "MonteCarlo", "missing MonteCarlo module");

        const auto mc_actions = reticolo::runtime::metadata::actions_for_module("MonteCarlo");
        require_contains(mc_actions, "RelativisticBoseGas", "missing RelativisticBoseGas action");
        require_contains(mc_actions, "RelativisticBoseGas_F", "missing RelativisticBoseGas_F action");
        require_contains(mc_actions, "RelativisticBoseGas_D", "missing RelativisticBoseGas_D action");
        require_contains(mc_actions, "WeakFieldEuclideanGR", "missing WeakFieldEuclideanGR action");

        const auto rbg_info = reticolo::runtime::metadata::describe_action("RelativisticBoseGas");
        require(rbg_info.has_value(), "missing RelativisticBoseGas metadata");
        require(rbg_info->canonical_name == "RelativisticBoseGas", "wrong canonical name for RelativisticBoseGas");
        require(rbg_info->canonical_precision == reticolo::registration::ActionPrecisionBinding::double_precision,
                "RelativisticBoseGas canonical precision should be explicit and double");
        require(rbg_info->alias_precision == reticolo::registration::ActionPrecisionBinding::double_precision,
                "RelativisticBoseGas alias precision should be explicit and double");
        require(rbg_info->supports_hmc, "RelativisticBoseGas should support HMC");
        require_contains(rbg_info->algorithms, "Metropolis", "missing Metropolis for RelativisticBoseGas");
        require_contains(rbg_info->algorithms, "HMC", "missing HMC for RelativisticBoseGas");
        require_contains(rbg_info->algorithms, "LLRMetropolis", "missing LLRMetropolis for RelativisticBoseGas");


        require_contains(mc_actions, "TestAction", "missing TestAction action");
        require_contains(mc_actions, "TestAction_F", "missing TestAction_F action");
        require_contains(mc_actions, "TestAction_D", "missing TestAction_D action");

        const auto test_action_info = reticolo::runtime::metadata::describe_action("TestAction");
        require(test_action_info.has_value(), "missing TestAction metadata");
        require(test_action_info->canonical_name == "TestAction", "wrong canonical name for TestAction");
        require(test_action_info->canonical_precision == reticolo::registration::ActionPrecisionBinding::double_precision,
                "TestAction canonical precision should be explicit and double");
        require_contains(test_action_info->algorithms, "Metropolis", "missing Metropolis for TestAction");
        require_contains(test_action_info->algorithms, "HMC", "missing HMC for TestAction");

        auto test_action_module = reticolo::ModuleFactory::MakeModule("MonteCarlo", "TestAction");
        require(static_cast<bool>(test_action_module), "failed to create TestAction module");
        require(dynamic_cast<TestActionHandler*>(test_action_module.get()) != nullptr,
                "TestAction canonical action should bind to the declared canonical precision module");

        auto test_action_updater =
            reticolo::MMonteCarlo::AlgorithmFactory::MakeUpdater<TestActionAction>("Metropolis");
        require(static_cast<bool>(test_action_updater), "failed to create TestAction updater");
        const auto wfgr_info = reticolo::runtime::metadata::describe_action("WeakFieldEuclideanGR");
        require(wfgr_info.has_value(), "missing WeakFieldEuclideanGR metadata");
        require(wfgr_info->canonical_precision == reticolo::registration::ActionPrecisionBinding::double_precision,
                "WeakFieldEuclideanGR canonical precision should be explicit and double");
        require(!wfgr_info->supports_hmc, "WeakFieldEuclideanGR should not support HMC");
        require(wfgr_info->algorithms.size() == 1, "WeakFieldEuclideanGR should expose exactly one algorithm");
        require_contains(wfgr_info->algorithms, "Metropolis", "missing Metropolis for WeakFieldEuclideanGR");

        auto wfgr_module = reticolo::ModuleFactory::MakeModule("MonteCarlo", "WeakFieldEuclideanGR");
        require(static_cast<bool>(wfgr_module), "failed to create WeakFieldEuclideanGR module");
        require(dynamic_cast<WfgrHandler*>(wfgr_module.get()) != nullptr,
                "WeakFieldEuclideanGR canonical action should bind to the declared canonical precision module");

        auto rbg_module = reticolo::ModuleFactory::MakeModule("MonteCarlo", "RelativisticBoseGas");
        require(static_cast<bool>(rbg_module), "failed to create RelativisticBoseGas module");
        require(dynamic_cast<RbgHandler*>(rbg_module.get()) != nullptr,
                "RelativisticBoseGas canonical action should bind to the declared canonical precision module");

        auto wfgr_updater = reticolo::MMonteCarlo::AlgorithmFactory::MakeUpdater<Wfgr>("Metropolis");
        require(static_cast<bool>(wfgr_updater), "failed to create WeakFieldEuclideanGR updater");

        auto rbg_updater = reticolo::MMonteCarlo::AlgorithmFactory::MakeUpdater<Rbg>("HMC");
        require(static_cast<bool>(rbg_updater), "failed to create RelativisticBoseGas HMC updater");

        try {
            reticolo::ModuleFactory::ValidateModuleAction("MonteCarlo", "NotAnAction");
            require(false, "invalid action should have thrown");
        } catch (const std::runtime_error& err) {
            require_message_contains(err, "Available actions", "invalid action error should list available actions");
            require_message_contains(err, "RelativisticBoseGas", "invalid action error missing known action");
        }

        try {
            reticolo::MMonteCarlo::AlgorithmFactory::ValidateUpdaterName<Wfgr>("HMC");
            require(false, "invalid updater should have thrown");
        } catch (const std::runtime_error& err) {
            require_message_contains(err, "Available algorithms", "invalid updater error should list algorithms");
            require_message_contains(err, "Metropolis", "invalid updater error missing supported algorithm");
        }
    } catch (const std::exception& err) {
        std::fprintf(stderr, "Exception: %s\n", err.what());
        return 1;
    }

    return 0;
}
