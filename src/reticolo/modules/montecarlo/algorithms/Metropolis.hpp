/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <random>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/factory/MCAlgorithmRegistry.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

/*--------------------------------------------------------------------------------------------------
  Metropolis algorithm class declaration
--------------------------------------------------------------------------------------------------*/
template <class Action, class TGen = std::mt19937_64>
class Metropolis : public MCAlgorithmBase<Action, TGen> {
  public:
    /* Types */
    using action_type = Action::action_type;
    using field_type = Action::field_type;
    using impl_type = Action::impl_type;
    using lattice_type = Lattice<field_type>;
    using size_type = Lattice<field_type>::size_type;
    using monte_carlo_data_type = MMonteCarlo::data<action_type>;

    /* Constructor */
    Metropolis() = default;

    /* setup */
    void setup(const YAML::Node& Params, const lattice_type& Field) override;

    /* execution */
    void updateField(lattice_type& field, Action& action, monte_carlo_data_type& state, TGen& rng) override;

  private:
    /* Metropolis settings */
    impl_type _ProposalWidth;

    /* Distributions */
    std::uniform_real_distribution<impl_type> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<impl_type> _Unifc;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<impl_type>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0)
};

/*--------------------------------------------------------------------------------------------------
  Metropolis<Action>::setup(...) implementation
--------------------------------------------------------------------------------------------------*/
template <class Action, class TGen>
inline void Metropolis<Action, TGen>::setup(const YAML::Node& Params, const lattice_type& /*unused*/) {
    /* Parse parameters */
    _ProposalWidth = Params["prop_width"].as<impl_type>();

    /* Initialize distributions */
    _Unif = std::uniform_real_distribution<impl_type>(0.0, 1.0);
    _Unifc = std::uniform_real_distribution<impl_type>(-1.0, 1.0);
    _Norm = std::normal_distribution<impl_type>(0.0, 1.0);
}

/*--------------------------------------------------------------------------------------------------
  Metropolis<Action>::updateField(...) implementation
--------------------------------------------------------------------------------------------------*/
template <class Action, class TGen>
inline void Metropolis<Action, TGen>::updateField(lattice_type& field, Action& action, monte_carlo_data_type& state,
                                                  TGen& rng) {
    size_type   Acc = 0;       // acceptance
    action_type SVarTot(0.0);  // cumulative action variation
    // Reset distributions
    _Norm.reset();
    _Unif.reset();
    // Loop over the entire lattice
    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        // Generate a randomized local field variation
        field_type FieldVar;  // local field variation
        randomize(FieldVar, _ProposalWidth, _Norm, rng);
        // Compute the associated action variation
        action_type SVar = action.compute_dS_loc(field, FieldVar, Site);
        // Metropolis acceptance check
        if (exp(-make_real(SVar)) > _Unif(rng)) {
            Acc++;
            field[Site] += FieldVar;
            SVarTot += SVar;
        }
    }
    state.update(static_cast<impl_type>(Acc) / field.getNsites(), SVarTot);
}

template <class Action, class TGen = std::mt19937_64>
inline void register_metropolis_algorithm() {
    static const bool Registered = []() {
        register_algorithm_type<Action, Metropolis<Action, TGen>, TGen>("Metropolis");
        return true;
    }();
    (void)Registered;
}

}  // namespace reticolo::MMonteCarlo
