/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/LlrHmcWorker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cmath>
#include <format>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/random.hpp"

namespace fs = std::filesystem;

namespace reticolo::LLR {

template <class T>
concept LLRCapable = T::IsLLRCapable;

/*--------------------------------------------------------------------------------------------------
    LLRWorker Class Declaration
--------------------------------------------------------------------------------------------------*/

template <LLRCapable Action>
class LLRHMCWorker : public montecarlo::MonteCarloHandler<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* Introduced names from MonteCarloHandler (avoids using this->...) */
    using montecarlo::MonteCarloHandler<Action>::m_Field;
    using montecarlo::MonteCarloHandler<Action>::m_Action;
    using montecarlo::MonteCarloHandler<Action>::m_McStats;
    using montecarlo::MonteCarloHandler<Action>::m_NMeasurements;
    using montecarlo::MonteCarloHandler<Action>::m_Unif;
    using montecarlo::MonteCarloHandler<Action>::m_UnifC;
    using montecarlo::MonteCarloHandler<Action>::m_Norm;
    using montecarlo::MonteCarloHandler<Action>::m_Rng;
    using montecarlo::MonteCarloHandler<Action>::m_Logger;
    using montecarlo::MonteCarloHandler<Action>::m_Timer;

    /* HMC support fields */
    LatticeType m_Mom;
    LatticeType m_Forces;
    LatticeType m_OldField;

    /* HMC settings */
    double m_Stepsize;
    uint   m_Steps;

    /* LLR parameters */
    double m_Sk;
    double m_Ak;
    double m_Width;

  public:
    /* Constructor */
    LLRHMCWorker(std::string handler_name, Action& action, LatticeType& field, uint seed,
                 const std::string& output_path, bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double traj_length, uint steps) {
        m_Steps = steps;
        m_Stepsize = traj_length / m_Steps;
    }
    void setLLRParams(double ak, double sk, double width) {
        m_Ak = ak;
        m_Sk = sk;
        m_Width = width;
    }

    void set_ak(double ak) { m_Ak = ak; }
};

template <LLRCapable Action>
LLRHMCWorker<Action>::LLRHMCWorker(   //
    std::string        handler_name,  //
    Action&            action,        //
    LatticeType&       field,         //
    uint               seed,          //
    const std::string& output_path,   //
    bool               StdOut,        //
    bool               save_data,     //
    bool               save_config)
    : montecarlo::MonteCarloHandler<Action>(  //
          output_path,                        //
          handler_name,                       //
          action, field,                      //
          seed,                               //
          StdOut,                             //
          save_data,                          //
          save_config),
      m_Mom(field.getSizes()),
      m_Forces(field.getSizes()),
      m_OldField(field.getSizes()) {
    // Log stuff
    m_Logger << IO::LI_time() +
                    std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", m_Timer.elapsed_ms());
}

template <LLRCapable Action>
void LLRHMCWorker<Action>::updateField() {
    int NSites = m_Field.getNsites();
    // save the old field configuration;
    m_OldField = m_Field;
    // Generate random momenta and compute initial kinetic term
    RealD OldK(0.0);
    for (int Site = 0; Site < NSites; Site++) {
        randomize(m_Mom[Site], 1.0, m_Norm, m_Rng);
        OldK += dot(m_Mom[Site]);
    }
    OldK *= 0.5;
    // Compute Forces
    m_Action.compute_LLRForces(m_Field, m_Forces, m_Sk, m_Width, m_Ak);
    // Momenta half step
    for (int Site = 0; Site < NSites; Site++) {
        m_Mom[Site] -= 0.5 * m_Stepsize * m_Forces[Site];
    }
    // Leapfrog algorithm
    for (uint Step = 0; Step < m_Steps; Step++) {
        // Update field
        for (int Site = 0; Site < NSites; Site++) {
            m_Field[Site] += m_Stepsize * m_Mom[Site];
        }
        // Compute updated forces
        m_Action.compute_LLRForces(m_Field, m_Forces, m_Sk, m_Width, m_Ak);
        // Update momenta
        for (int Site = 0; Site < NSites; Site++) {
            m_Mom[Site] -= m_Stepsize * m_Forces[Site];
        }
    }
    // Momenta half step roll-back
    for (int Site = 0; Site < m_Mom.getNsites(); Site++) {
        m_Mom[Site] += 0.5 * m_Stepsize * m_Forces[Site];
    }
    // Compute final action
    ActionType OldS = m_McStats.getS();
    ActionType NewS = m_Action.compute_S(m_Field);
    // Compute end kinetic term
    RealD NewK(0.0);
    for (int Site = 0; Site < m_Mom.getNsites(); Site++) {
        NewK += dot(m_Mom[Site]);
    }
    NewK *= 0.5;
    RealD SVarFull = NewS.real() - OldS.real() +                                                //
                     (NewS.imag() - m_Sk) * (NewS.imag() - m_Sk) / (2.0 * m_Width * m_Width) -  //
                     (OldS.imag() - m_Sk) * (OldS.imag() - m_Sk) / (2.0 * m_Width * m_Width) +
                     m_Ak * (NewS.imag() - OldS.imag());

    // Final Metropolis check
    if (exp(OldK - NewK - SVarFull) > m_Unif(m_Rng)) {
        m_McStats.update(1, NewS - m_McStats.getS());
    } else {
        m_Field = m_OldField;
        m_McStats.update(0, 0.0);
    }
}

/* Argument deduction guide */
template <LLRCapable Action>
LLRHMCWorker(std::string, Action&, Lattice<typename Action::FieldType>, uint, std::string&) -> LLRHMCWorker<Action>;

}  // namespace reticolo::LLR
