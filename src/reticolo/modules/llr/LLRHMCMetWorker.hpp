/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/LlrHmcMetWorker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cmath>
#include <format>
#include <string>

#include "reticolo/lattice/Lattice.hpp"
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
class LLRHMCMetWorker : public montecarlo::MonteCarloHandler<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* HMC support fields */
    LatticeType m_Mom;
    LatticeType m_Forces;
    LatticeType m_OldField;

    /* HMC settings */
    double m_Stepsize;
    uint   m_Steps;

    /* Met settings */
    double m_ProposalWidth;

    /* LLR parameters */
    double m_Sk;
    double m_Ak;
    double m_Width;

    /* Introduced names from MonteCarloHandler (avoids using this->...) */
    using montecarlo::MonteCarloHandler<Action>::_Field;
    using montecarlo::MonteCarloHandler<Action>::_Action;
    using montecarlo::MonteCarloHandler<Action>::_McStats;
    using montecarlo::MonteCarloHandler<Action>::_NMeasurements;
    using montecarlo::MonteCarloHandler<Action>::_Unif;
    using montecarlo::MonteCarloHandler<Action>::_UnifC;
    using montecarlo::MonteCarloHandler<Action>::_Norm;
    using montecarlo::MonteCarloHandler<Action>::_Rng;
    using montecarlo::MonteCarloHandler<Action>::_Logger;
    using montecarlo::MonteCarloHandler<Action>::_T;

  public:
    /* Constructor */
    LLRHMCMetWorker(std::string handler_name, Action& action, LatticeType& field, uint seed,
                    const std::string& output_path, bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double hmc_traj_len, uint hmc_steps, double met_prop_width) {
        m_Steps = hmc_steps;
        m_Stepsize = hmc_traj_len / hmc_steps;
        m_ProposalWidth = met_prop_width;
    }
    void setLLRParams(double ak, double sk, double width) {
        m_Ak = ak;
        m_Sk = sk;
        m_Width = width;
    }

    void set_ak(double ak) { m_Ak = ak; }
};

template <LLRCapable Action>
LLRHMCMetWorker<Action>::LLRHMCMetWorker(  //
    std::string        handler_name,       //
    Action&            action,             //
    LatticeType&       field,              //
    uint               seed,               //
    const std::string& output_path,        //
    bool               StdOut,             //
    bool               save_data,          //
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
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <LLRCapable Action>
void LLRHMCMetWorker<Action>::updateField() {
    int NSites = _Field.getNsites();
    // save the old field configuration;
    m_OldField = _Field;
    // Generate random momenta and compute initial kinetic term
    RealD OldK(0.0);
    for (int Site = 0; Site < NSites; Site++) {
        randomize(m_Mom[Site], 1.0, _Norm, _Rng);
        OldK += dot(m_Mom[Site]);
    }
    OldK *= 0.5;
    // Compute Forces
    _Action.compute_LLRForces(_Field, m_Forces, m_Sk, m_Width, m_Ak);
    // Momenta half step
    for (int Site = 0; Site < NSites; Site++) {
        m_Mom[Site] -= 0.5 * m_Stepsize * m_Forces[Site];
    }
    // Leapfrog algorithm
    for (uint Step = 0; Step < m_Steps; Step++) {
        // Update field
        for (int Site = 0; Site < NSites; Site++) {
            _Field[Site] += m_Stepsize * m_Mom[Site];
        }
        // Compute updated forces
        _Action.compute_LLRForces(_Field, m_Forces, m_Sk, m_Width, m_Ak);
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
    ActionType OldS = _McStats.getS();
    ActionType NewS = _Action.compute_S(_Field);
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
    if (exp(OldK - NewK - SVarFull) > _Unif(_Rng)) {
        this->_McStats.update(1, NewS - _McStats.getS());
    } else {
        this->_Field = m_OldField;
        this->_McStats.update(0, 0.0);
    }

    // Extra Metropolis update to decorrelate the imaginary part of the action
    uint Acc = 0;  // acceptance

    double WindowWeight;  // weight of the windowing function   [normal distribution]
    double LlrWeight;     // weight of the llr reweighting      [exp(-ak(S-Sk)]

    ActionType SVar;               // local action variation
    ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    FieldType FieldVar;  // local field variation

    for (uint Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(FieldVar, 0.2, _UnifC, _Rng);

        SVar = _Action.compute_dS_loc(_Field, FieldVar, Site);

        WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * _McStats._SIm - 2.0 * m_Sk)) / (2.0 * m_Width * m_Width);
        LlrWeight = m_Ak * SVar.imag();

        if (exp(-(SVar.real() + WindowWeight + LlrWeight)) > _Unif(_Rng)) {
            Acc++;
            _Field[Site] += FieldVar;
            SVarTot += SVar;
            _McStats._SRe += SVar.real();
            _McStats._SIm += SVar.imag();
        }
    }

    _McStats._Acceptance = static_cast<double>(Acc) / _Field.getNsites();
    _McStats._DSRe = SVarTot.real();
    _McStats._DSIm = SVarTot.imag();
}

/* Argument deduction guide */
template <LLRCapable Action>
LLRHMCMetWorker(std::string, Action&, Lattice<typename Action::FieldType, Action::Dims>, uint, std::string&)
    -> LLRHMCMetWorker<Action>;

}  // namespace reticolo::LLR