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

#include <omp.h>

#include <random>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

/*--------------------------------------------------------------------------------------------------
  Metropolis algorithm class declaration
--------------------------------------------------------------------------------------------------*/
template <class Action>
class Metropolis : public MCAlgorithmBase<Action> {
  public:
    /* Types */
    using action_type = Action::action_type;
    using field_type = Action::field_type;
    using impl_type = Action::impl_type;
    using lattice_type = Lattice<field_type>;
    using monte_carlo_data_type = MMonteCarlo::data<action_type>;

    /* Constructor */
    Metropolis() = default;

    /* setup */
    void setup(const YAML::Node& Params, const Lattice<field_type>& Field) override;

    /* execution */
    void updateField(Lattice<field_type>& field, Action& action, monte_carlo_data_type& state, RNGType& rng) override;

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
template <class Action>
inline void Metropolis<Action>::setup(const YAML::Node& Params, const Lattice<field_type>& Field) {
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
template <class Action>
inline void Metropolis<Action>::updateField(Lattice<field_type>& field, Action& action, monte_carlo_data_type& state,
                                            RNGType& rng) {
    uint        Acc = 0;       // acceptance
    action_type SVarTot(0.0);  // cumulative action variation
    // Loop over the entire lattice
    for (uint Site = 0; Site < field.getNsites(); Site++) {
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
    state.update(((impl_type)Acc) / field.getNsites(), SVarTot);
}

}  // namespace reticolo::MMonteCarlo
