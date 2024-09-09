/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <concepts>
#include <cstdlib>
#include <memory>
#include <random>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

template <class Action>
class HMC : public MCAlgorithmBase<Action> {
  public:
    /* Types definitions */
    using action_type = Action::action_type;
    using field_type = Action::field_type;
    using impl_type = Action::impl_type;
    using lattice_type = Lattice<field_type>;
    using monte_carlo_data_type = MMonteCarlo::data<action_type>;

  private:
    /* HMC support fields */
    std::unique_ptr<lattice_type> _Mom;
    std::unique_ptr<lattice_type> _Forces;
    std::unique_ptr<lattice_type> _OldField;

    /* HMC settings */
    impl_type _Stepsize;
    uint      _Steps;

    /* Distributions */
    std::uniform_real_distribution<impl_type> _Unif;  // Uniform distribution [0.0, 1.0]
    std::normal_distribution<impl_type>       _Norm;  // Normal distibution (mean: 0.0, stddev: 1.0)

  public:
    HMC() = default;

    /* setup */
    void setup(const YAML::Node& Params, const Lattice<field_type>& Field) override {
        /* Parse Parameters */
        _Stepsize = Params["stepsize"].as<impl_type>();
        _Steps = Params["steps"].as<uint>();

        /* Allocate auxiliary fields */
        _Mom = std::make_unique<Lattice<field_type>>(Field);
        _Forces = std::make_unique<Lattice<field_type>>(Field);
        _OldField = std::make_unique<Lattice<field_type>>(Field);

        /* Initialize distributions */
        _Unif = std::uniform_real_distribution<impl_type>(0.0, 1.0);
        _Norm = std::normal_distribution<impl_type>(0.0, 1.0);
        static_assert(std::same_as<typename std::uniform_real_distribution<impl_type>::result_type, impl_type>,
                      "type error");
    }

    /* execution */
    void updateField(Lattice<field_type>& field, Action& action, monte_carlo_data_type& state, RNGType& rng) override {
        int       NSites = field.getNsites();
        impl_type Half = 0.5;
        // save the old field configuration;
        *_OldField = field;
        // Generate random momenta and compute initial kinetic term
        impl_type OldK = 0.0;
        for (int Site = 0; Site < NSites; Site++) {
            randomize((*_Mom)[Site], 1.0, _Norm, rng);
            OldK += dot((*_Mom)[Site]);
        }
        OldK *= 0.5;
        // Compute Forces
        action.compute_Forces(field, (*_Forces));
        // Momenta half step
        for (int Site = 0; Site < NSites; Site++) {
            (*_Mom)[Site] -= Half * _Stepsize * (*_Forces)[Site];
        }
        // Leapfrog algorithm
        for (uint Step = 0; Step < _Steps; Step++) {
            // Update field
            for (int Site = 0; Site < NSites; Site++) {
                field[Site] += _Stepsize * (*_Mom)[Site];
            }
            // Compute updated forces
            action.compute_Forces(field, (*_Forces));
            // Update momenta
            for (int Site = 0; Site < NSites; Site++) {
                (*_Mom)[Site] -= _Stepsize * (*_Forces)[Site];
            }
        }
        // Momenta half step roll-back
        for (int Site = 0; Site < NSites; Site++) {
            (*_Mom)[Site] += Half * _Stepsize * (*_Forces)[Site];
        }
        // Compute final action
        action_type OldS = state.getS();
        action_type NewS = action.compute_S(field);
        // Compute end kinetic term
        impl_type NewK = 0.0;
        for (int Site = 0; Site < NSites; Site++) {
            NewK += dot((*_Mom)[Site]);
        }
        NewK *= 0.5;

        // Final Metropolis check
        if (exp(OldK - NewK + make_real(OldS - NewS)) > _Unif(rng)) {
            state.update(1, NewS - OldS);
        } else {
            field = *_OldField;
            state.update(0, 0.0);
        }
    }
};

}  // namespace reticolo::MMonteCarlo
