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
    using LatticeType = Lattice<typename Action::FieldType, Action::Dims>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* Met settings */
    double _ProposalWidth;

    /* LLR parameters */
    double _Sk;
    double _Ak;
    double _Width;

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
    LLRMetWorker(std::string handler_name, Action& action, LatticeType& field, uint seed,
                 const std::string& output_path, bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(double propWidth) { _ProposalWidth = propWidth; }

    void setLLRParams(double ak, double sk, double width) {
        _Ak = ak;
        _Sk = sk;
        _Width = width;
    }

    void set_ak(double ak) { _Ak = ak; }
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
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <LLRCapable Action>
void LLRMetWorker<Action>::updateField() {
    uint Acc = 0;  // acceptance

    double WindowWeight;  // weight of the windowing function   [normal distribution]
    double LlrWeight;     // weight of the llr reweighting      [exp(-ak(S-Sk)]

    ActionType SVar;               // local action variation
    ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    FieldType FieldVar;  // local field variation

    for (uint Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(FieldVar, _ProposalWidth, _UnifC, _Rng);

        SVar = _Action.compute_dS_loc(_Field, FieldVar, Site);

        WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * _McStats._SIm - 2.0 * _Sk)) / (2.0 * _Width * _Width);
        LlrWeight = _Ak * SVar.imag();

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
LLRMetWorker(std::string, Action&, Lattice<typename Action::FieldType, Action::Dims>, uint, std::string&)
    -> LLRMetWorker<Action>;

}  // namespace reticolo::LLR