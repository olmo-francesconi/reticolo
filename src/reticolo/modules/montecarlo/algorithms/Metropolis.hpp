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

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

template <class Action>
class Metropolis : public MCAlgorithmBase<Action> {
  private:
    /* Types definitions */
    using action_type = typename Action::ActionType;
    using field_type = typename Action::FieldType;
    using lattice_type = Lattice<field_type>;
    using monte_carlo_data_type = MMonteCarlo::data<action_type>;

    /* HMC settings */
    RealD _ProposalWidth;

    /* Distributions */
    std::uniform_real_distribution<RealD> _Unif;  // Uniform distribution [0.0, 1.0]
    std::normal_distribution<RealD>       _Norm;  // Normal distibution (mean: 0.0, stddev: 1.0)

  public:
    Metropolis() = default;

    /* setup */
    void setup(const YAML::Node& Params, const Lattice<field_type>& Field) override {
        /* Parse parameters */
        _ProposalWidth = Params["prop_width"].as<RealD>();

        /* Initialize distributions */
        _Unif = std::uniform_real_distribution<RealD>(0.0, 1.0);
        _Norm = std::normal_distribution<RealD>(0.0, 1.0);
    }

    /* execution */
    void updateField(Lattice<field_type>& field, Action& action, monte_carlo_data_type& state, RNGType& rng) override {
        uint        Acc = 0;       // acceptance
        action_type SVarTot(0.0);  // cumulative action variation

        // Loop over the entire lattice
        for (int Site = 0; Site < field.getNsites(); Site++) {
            // uint ThId = omp_get_thread_num();
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
        state.update(((RealD)Acc) / field.getNsites(), SVarTot);
    }
};

}  // namespace reticolo::MMonteCarlo
