/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo::montecarlo {

template <class T>
concept MetropolisCapable = T::IsMetropolisCapable;

template <MetropolisCapable Action>
class Metropolis : public MonteCarloHandler<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType, Action::Dims>;

    /* Introduced names from MonteCarloHandler (avoids using ...) */
    using MonteCarloHandler<Action>::_Field;
    using MonteCarloHandler<Action>::_Action;
    using MonteCarloHandler<Action>::_McStats;
    using MonteCarloHandler<Action>::_Unif;
    using MonteCarloHandler<Action>::_UnifC;
    using MonteCarloHandler<Action>::_Norm;
    using MonteCarloHandler<Action>::_Rng;
    using MonteCarloHandler<Action>::_Logger;
    using MonteCarloHandler<Action>::_T;

    /* Metropolis settings */
    double _ProposalWidth = 1.0;

  public:
    /* Constructor */
    Metropolis(std::string handler_name, Action& action, LatticeType& field, uint seed, const std::string& output_path,
               bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double propWidth) { _ProposalWidth = propWidth; }
};

template <MetropolisCapable Action>
Metropolis<Action>::Metropolis(std::string        handler_name,  //
                               Action&            action,        //
                               LatticeType&       field,         //
                               uint               seed,          //
                               const std::string& output_path,   //
                               bool               StdOut,        //
                               bool               save_data,     //
                               bool               save_config)
    : montecarlo::MonteCarloHandler<Action>(output_path,    //
                                            handler_name,   //
                                            action, field,  //
                                            seed,           //
                                            StdOut,         //
                                            save_data,      //
                                            save_config) {
    // Log stuff
    _Logger << IO::LI_time() +
                   std::format("MetropolisWorker - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <MetropolisCapable Action>
void Metropolis<Action>::updateField() {
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
    _McStats.update(((double)Acc) / _Field.getNsites(), SVarTot);
}

/* Argument deduction guide */
template <MetropolisCapable Action>
Metropolis(std::string, Action&, Lattice<typename Action::FieldType, Action::Dims>, uint, std::string&, bool, bool,
           bool) -> Metropolis<Action>;
}  // namespace reticolo::montecarlo
