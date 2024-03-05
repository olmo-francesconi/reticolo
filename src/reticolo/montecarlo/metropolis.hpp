/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <string>

#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/types/core.hpp"

namespace reticolo::montecarlo {

template <class T>
concept MetropolisCapable = T::IsMetropolisCapable;

template <MetropolisCapable Action>
class Metropolis : public MonteCarloHandler<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;

  public:
    /* Inherit all base class constructors from MonteCarloHandler */
    using MonteCarloHandler<Action>::MonteCarloHandler;

    /* override virtual updateField() method */
    void updateField() override {
        uint       Acc = 0;       // acceptance
        ActionType SVarTot(0.0);  // cumulative action variation

        // Loop over the entire lattice
        for (uint Site = 0; Site < this->_Field.getNsites(); Site++) {
            // uint ThId = omp_get_thread_num();
            // Generate a randomized local field variation
            FieldType FieldVar;  // local field variation
            randomize(FieldVar, 0.2, this->_UnifC, this->_Rng);

            // Compute the associated action variation
            ActionType SVar = this->_Action.compute_dS_loc(this->_Field, FieldVar, Site);

            // Metropolis acceptance check
            if (exp(-SVar.real()) > this->_Unif(this->_Rng)) {
                Acc++;
                this->_Field[Site] += FieldVar;
                SVarTot += SVar;
            }
        }

        // Update montecarlo stats
        this->_McStats.update(static_cast<double>(Acc) / this->_Field.getNsites(), SVarTot);
    }
};

/* Argument deduction guide */
template <MetropolisCapable Action>
Metropolis(std::string, Action&, uintvect<Action::Dims>, uint, std::string&) -> Metropolis<Action>;

}  // namespace reticolo::montecarlo