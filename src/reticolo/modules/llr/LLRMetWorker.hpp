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
class LLRMetWorker : public montecarlo::MonteCarloHandler<Action> {
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

    /* Met settings */
    double m_ProposalWidth;

    /* LLR parameters */
    double m_Sk;
    double m_Ak;
    double m_Width;

  public:
    /* Constructor */
    LLRMetWorker(std::string handler_name, Action& action, LatticeType& field, uint seed,
                 const std::string& output_path, bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double propWidth) { m_ProposalWidth = propWidth; }

    void setLLRParams(double ak, double sk, double width) {
        m_Ak = ak;
        m_Sk = sk;
        m_Width = width;
    }

    void set_ak(double ak) { m_Ak = ak; }
};

template <LLRCapable Action>
LLRMetWorker<Action>::LLRMetWorker(std::string        handler_name,  //
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
                    std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", m_Timer.elapsed_ms());
}

template <LLRCapable Action>
void LLRMetWorker<Action>::updateField() {
    uint Acc = 0;  // acceptance

    double WindowWeight;  // weight of the windowing function   [normal distribution]
    double LlrWeight;     // weight of the llr reweighting      [exp(-ak(S-Sk)]

    ActionType SVar;               // local action variation
    ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    FieldType FieldVar;  // local field variation

    for (uint Site = 0; Site < m_Field.getNsites(); Site++) {
        randomize(FieldVar, m_ProposalWidth, m_UnifC, m_Rng);

        SVar = m_Action.compute_dS_loc(m_Field, FieldVar, Site);

        WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * m_McStats._SIm - 2.0 * m_Sk)) / (2.0 * m_Width * m_Width);
        LlrWeight = m_Ak * SVar.imag();

        if (exp(-(SVar.real() + WindowWeight + LlrWeight)) > m_Unif(m_Rng)) {
            Acc++;
            m_Field[Site] += FieldVar;
            SVarTot += SVar;
            m_McStats._SRe += SVar.real();
            m_McStats._SIm += SVar.imag();
        }
    }

    m_McStats._Acceptance = static_cast<double>(Acc) / m_Field.getNsites();
    m_McStats._DSRe = SVarTot.real();
    m_McStats._DSIm = SVarTot.imag();
}

/* Argument deduction guide */
template <LLRCapable Action>
LLRMetWorker(std::string, Action&, Lattice<typename Action::FieldType>, uint, std::string&) -> LLRMetWorker<Action>;

}  // namespace reticolo::LLR
