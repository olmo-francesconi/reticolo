/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/LlrHmcWorker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

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
class LlrHmcWorker : public montecarlo::MonteCarloHandler<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType, Action::Dims>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* HMC support fields */
    LatticeType _Mom;
    LatticeType _Forces;
    LatticeType _OldField;

    /* HMC settings */
    double _Stepsize;
    uint   _Steps;

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
    LlrHmcWorker(std::string handler_name, Action& action, LatticeType& field, uint seed,
                 const std::string& output_path, bool StdOut, bool save_data, bool save_config);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(uint steps, double stepsize) {
        _Stepsize = stepsize;
        _Steps = steps;
    }
    void setLLRParams(double ak, double sk, double width) {
        _Ak = ak;
        _Sk = sk;
        _Width = width;
    }

    void set_ak(double ak) { _Ak = ak; }
};

template <LLRCapable Action>
LlrHmcWorker<Action>::LlrHmcWorker(   //
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
      _Mom(field.getSizes()),
      _Forces(field.getSizes()),
      _OldField(field.getSizes()) {
    // Log stuff
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <LLRCapable Action>
void LlrHmcWorker<Action>::updateField() {
    int NSites = _Field.getNsites();
    // save the old field configuration;
    _OldField = _Field;
    // Generate random momenta and compute initial kinetic term
    RealD OldK(0.0);
    for (int Site = 0; Site < NSites; Site++) {
        randomize(_Mom[Site], 1.0, _Norm, _Rng);
        OldK += dot(_Mom[Site]);
    }
    OldK *= 0.5;
    // Compute Forces
    _Action.compute_LLRForces(_Field, _Forces, _Sk, _Width, _Ak);
    // Momenta half step
    for (int Site = 0; Site < NSites; Site++) {
        _Mom[Site] -= 0.5 * _Stepsize * _Forces[Site];
    }
    // Leapfrog algorithm
    for (uint Step = 0; Step < _Steps; Step++) {
        // Update field
        for (int Site = 0; Site < NSites; Site++) {
            _Field[Site] += _Stepsize * _Mom[Site];
        }
        // Compute updated forces
        _Action.compute_LLRForces(_Field, _Forces, _Sk, _Width, _Ak);
        // Update momenta
        for (int Site = 0; Site < NSites; Site++) {
            _Mom[Site] -= _Stepsize * _Forces[Site];
        }
    }
    // Momenta half step roll-back
    for (int Site = 0; Site < _Mom.getNsites(); Site++) {
        _Mom[Site] += 0.5 * _Stepsize * _Forces[Site];
    }
    // Compute final action
    ActionType OldS = _McStats.getS();
    ActionType NewS = _Action.compute_S(_Field);
    // Compute end kinetic term
    RealD NewK(0.0);
    for (int Site = 0; Site < _Mom.getNsites(); Site++) {
        NewK += dot(_Mom[Site]);
    }
    NewK *= 0.5;
    RealD SVarFull = NewS.real() - OldS.real() +                                            //
                     (NewS.imag() - _Sk) * (NewS.imag() - _Sk) / (2.0 * _Width * _Width) -  //
                     (OldS.imag() - _Sk) * (OldS.imag() - _Sk) / (2.0 * _Width * _Width) +
                     _Ak * (NewS.imag() - OldS.imag());

    // Final Metropolis check
    if (exp(OldK - NewK - SVarFull) > _Unif(_Rng)) {
        this->_McStats.update(1, NewS - _McStats.getS());
    } else {
        this->_Field = _OldField;
        this->_McStats.update(0, 0.0);
    }
}

/* Argument deduction guide */
template <LLRCapable Action>
LlrHmcWorker(std::string, Action&, Lattice<typename Action::FieldType, Action::Dims>, uint, std::string&)
    -> LlrHmcWorker<Action>;

}  // namespace reticolo::LLR