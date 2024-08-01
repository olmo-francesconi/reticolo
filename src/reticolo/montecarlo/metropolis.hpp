/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <string>

#include "reticolo/lattice/Lattice.hpp"
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
    using LatticeType = Lattice<typename Action::FieldType>;

    /* Introduced names from MonteCarloHandler (avoids using ...) */
    using MonteCarloHandler<Action>::m_Field;
    using MonteCarloHandler<Action>::m_Action;
    using MonteCarloHandler<Action>::m_McStats;
    using MonteCarloHandler<Action>::m_Unif;
    using MonteCarloHandler<Action>::m_UnifC;
    using MonteCarloHandler<Action>::m_Norm;
    using MonteCarloHandler<Action>::m_Rng;
    using MonteCarloHandler<Action>::m_Logger;
    using MonteCarloHandler<Action>::m_Timer;

    /* Metropolis settings */
    double m_ProposalWidth = 1.0;

  public:
    /* Constructor */
    Metropolis(std::string handler_name, Action& action, LatticeType& field, uint seed, const std::string& output_path,
               bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double propWidth) { m_ProposalWidth = propWidth; }
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
    m_Logger << IO::LI_time() +
                    std::format("MetropolisWorker - Initialization completed in {:.3f} ms\n", m_Timer.elapsed_ms());
}

template <MetropolisCapable Action>
void Metropolis<Action>::updateField() {
    uint       Acc = 0;       // acceptance
    ActionType SVarTot(0.0);  // cumulative action variation

    // Loop over the entire lattice
    for (int Site = 0; Site < m_Field.getNsites(); Site++) {
        // uint ThId = omp_get_thread_num();
        // Generate a randomized local field variation
        FieldType FieldVar;  // local field variation
        randomize(FieldVar, m_ProposalWidth, m_UnifC, m_Rng);

        // Compute the associated action variation
        ActionType SVar = m_Action.compute_dS_loc(m_Field, FieldVar, Site);

        // Metropolis acceptance check
        if (exp(-make_real(SVar)) > m_Unif(m_Rng)) {
            Acc++;
            m_Field[Site] += FieldVar;
            SVarTot += SVar;
        }
    }
    m_McStats.update(((double)Acc) / m_Field.getNsites(), SVarTot);
}

/* Argument deduction guide */
template <MetropolisCapable Action>
Metropolis(std::string, Action&, Lattice<typename Action::FieldType>, uint, std::string&, bool, bool,
           bool) -> Metropolis<Action>;
}  // namespace reticolo::montecarlo
