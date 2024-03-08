/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo::montecarlo {

template <class T>
concept MetropolisCapable = T::IsMetropolisCapable;

template <MetropolisCapable Action, size_t dim>
class Metropolis : public MonteCarloHandler<Action, dim> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;

    /* Introduced names from MonteCarloHandler (avoids using ...) */
    using MonteCarloHandler<Action, dim>::_Field;
    using MonteCarloHandler<Action, dim>::_Action;
    using MonteCarloHandler<Action, dim>::_McStats;
    using MonteCarloHandler<Action, dim>::_Unif;
    using MonteCarloHandler<Action, dim>::_UnifC;
    using MonteCarloHandler<Action, dim>::_Norm;
    using MonteCarloHandler<Action, dim>::_Rng;
    using MonteCarloHandler<Action, dim>::_Logger;
    using MonteCarloHandler<Action, dim>::_T;

    /* Metropolis settings */
    double _ProposalWidth = 1.0;

  public:
    /* Constructor */
    Metropolis(std::string run_name, Action& action, Lattice<FieldType, dim>& field, uint seed,
               const std::string& output_path);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double propWidth) { _ProposalWidth = propWidth; }
};

template <MetropolisCapable Action, size_t dim>
Metropolis<Action, dim>::Metropolis(std::string run_name, Action& action, Lattice<FieldType, dim>& field, uint seed,
                                    const std::string& output_path)
    : MonteCarloHandler<Action, dim>(run_name, action, field, seed, output_path) {
    // Log stuff
    _Logger << IO::LI_time() +
                   std::format("MetropolisWorker - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <MetropolisCapable Action, size_t dim>
void Metropolis<Action, dim>::updateField() {
    uint       Acc = 0;       // acceptance
    ActionType SVarTot(0.0);  // cumulative action variation

    // Loop over the entire lattice
    for (int Site = 0; Site < _Field.getNsites(); Site++) {
        // uint ThId = omp_get_thread_num();
        // Generate a randomized local field variation
        FieldType FieldVar;  // local field variation
        randomize(FieldVar, _ProposalWidth, _UnifC, _Rng);

        // Compute the associated action variation
        ActionType SVar = _Action.compute_dS_loc(_Field, FieldVar, Site);

        // Metropolis acceptance check
        if (exp(-make_real(SVar)) > _Unif(_Rng)) {
            Acc++;
            _Field[Site] += FieldVar;
            SVarTot += SVar;
        }
    }

    // Update montecarlo stats
    _McStats.update(static_cast<double>(Acc) / _Field.getNsites(), SVarTot);
}

/* Argument deduction guide */
template <MetropolisCapable Action, size_t dim>
Metropolis(std::string, Action&, Lattice<typename Action::FieldType, dim>, uint, std::string&)
    -> Metropolis<Action, dim>;
}  // namespace reticolo::montecarlo