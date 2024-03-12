/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cmath>
#include <cstddef>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"

namespace reticolo::montecarlo {

template <class T>
concept HmcCapable = T::IsHmcCapable;

template <HmcCapable Action, size_t dim>
class HMC : public MonteCarloHandler<Action, dim> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType, dim>;

    /* HMC support fields */
    LatticeType _Mom;
    LatticeType _Forces;
    LatticeType _OldField;

    /* HMC settings */
    double _Stepsize;
    uint   _Steps;

    /* Introduced names from MonteCarloHandler (avoids using this->...) */
    using MonteCarloHandler<Action, dim>::_Field;
    using MonteCarloHandler<Action, dim>::_Action;
    using MonteCarloHandler<Action, dim>::_McStats;
    using MonteCarloHandler<Action, dim>::_Unif;
    using MonteCarloHandler<Action, dim>::_UnifC;
    using MonteCarloHandler<Action, dim>::_Norm;
    using MonteCarloHandler<Action, dim>::_Rng;
    using MonteCarloHandler<Action, dim>::_Logger;
    using MonteCarloHandler<Action, dim>::_T;

  public:
    /* Constructor */
    HMC(std::string run_name, Action& action, Lattice<FieldType, dim>& field, uint seed,
        const std::string& output_path);

    /* override virtual updateField() method */
    void updateField() override;

    /* Initialize parameters */
    void setParams(uint steps, double stepsize) {
        _Stepsize = stepsize;
        _Steps = steps;
    }
};

template <HmcCapable Action, size_t dim>
HMC<Action, dim>::HMC(std::string run_name, Action& action, Lattice<FieldType, dim>& field, uint seed,
                      const std::string& output_path)
    : MonteCarloHandler<Action, dim>(run_name, action, field, seed, output_path),
      _Mom(field.getSizes()),
      _Forces(field.getSizes()),
      _OldField(field.getSizes()) {
    // Log stuff
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
}

template <HmcCapable Action, size_t dim>
void HMC<Action, dim>::updateField() {
    int NSites = _Field.getNsites();
    // save the old field configuration;
    _OldField = _Field;
    // Compute initial action
    ActionType OldS = _Action.compute_S(_Field);
    // Generate random momenta and compute initial kinetic term
    RealD OldK(0.0);
    for (int Site = 0; Site < NSites; Site++) {
        randomize(_Mom[Site], 1.0, _Norm, _Rng);
        OldK += dot(_Mom[Site]);
    }
    OldK *= 0.5;
    // Compute Forces
    _Action.compute_Forces(_Field, _Forces);
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
        _Action.compute_Forces(_Field, _Forces);
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
    ActionType NewS = _Action.compute_S(_Field);
    // Compute end kinetic term
    RealD NewK(0.0);
    for (int Site = 0; Site < _Mom.getNsites(); Site++) {
        NewK += dot(_Mom[Site]);
    }
    NewK *= 0.5;
    // Final Metropolis check
    if (exp(OldK - NewK + make_real(OldS - NewS)) > _Unif(_Rng)) {
        this->_McStats.update(1, NewS - OldS);
    } else {
        this->_Field = _OldField;
        this->_McStats.update(0, 0.0);
    }
}

/* Argument deduction guide */
template <HmcCapable Action, size_t dim>
HMC(std::string, Action&, Lattice<typename Action::FieldType, dim>, uint, std::string&) -> HMC<Action, dim>;
}  // namespace reticolo::montecarlo