/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cmath>
#include <string>
#include <type_traits>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloHandler.hpp"
#include "reticolo/types/concepts.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"
#include "reticolo/types/random.hpp"

namespace reticolo::montecarlo {

template <class T>
concept HmcCapable = T::IsHmcCapable;

template <HmcCapable Action>
class HMC : public MonteCarloHandler<Action> {
  private:
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType, Action::Dims>;

    /* HMC support fields */
    LatticeType _Mom;
    LatticeType _Forces;
    LatticeType _OldField;

    /* HMC settings */
    double _Stepsize;
    uint   _Steps;

  public:
    /* Inherit all base class constructors from MonteCarloHandler */
    using MonteCarloHandler<Action>::MonteCarloHandler;

    void initialize(uint steps, double stepsize) {
        _Stepsize = stepsize;
        _Steps = steps;

        _Mom.init(this->_Field.getSizes());
        _Forces.init(this->_Field.getSizes());
    }

    /* override virtual updateField() method */
    void updateField() override {
        // save the old field configuration;
        _OldField = this->_Field;

        uint NSites = this->_Field.getNsites();

        ActionType OldS = this->_Action.compute_S(this->_Field);
        RealD      OldK(0.0);
        // Generate gaussian-distributed momenta
        for (uint Site = 0; Site < NSites; Site++) {
            randomize(_Mom[Site], 1.0, this->_Norm, this->_Rng);
            if constexpr (ComplexValue<FieldType>) {
                OldK += _Mom[Site].real() * _Mom[Site].real() + _Mom[Site].imag() * _Mom[Site].imag();
            } else {
                OldK += _Mom[Site] * _Mom[Site];
            }
        }
        OldK *= 0.5;

        // Compute Forces
        this->_Action.compute_Forces(this->_Field, _Forces);
        // Momenta half step
        for (uint Site = 0; Site < NSites; Site++) {
            _Mom[Site] -= 0.5 * _Stepsize * _Forces[Site];
        }

        // Leapfrog algorithm
        for (uint Step = 0; Step < _Steps - 1; Step++) {
            // update field
            for (uint Site = 0; Site < NSites; Site++) {
                this->_Field[Site] += _Stepsize * _Mom[Site];
            }

            // compute Updated forces
            this->_Action.compute_Forces(this->_Field, _Forces);

            // Update momenta
            for (uint Site = 0; Site < NSites; Site++) {
                _Mom[Site] -= _Stepsize * _Forces[Site];
            }
        }

        // update field
        for (uint Site = 0; Site < NSites; Site++) {
            this->_Field[Site] += _Stepsize * _Mom[Site];
        }

        // compute Updated forces
        this->_Action.compute_Forces(this->_Field, _Forces);

        // Update momenta
        for (uint Site = 0; Site < _Mom.getNsites(); Site++) {
            _Mom[Site] -= 0.5 * _Stepsize * _Forces[Site];
        }

        ActionType NewS = this->_Action.compute_S(this->_Field);
        RealD      NewK(0.0);
        // Compute end Hamiltonian
        for (uint Site = 0; Site < _Mom.getNsites(); Site++) {
            if constexpr (ComplexValue<ActionType>) {
                NewK += _Mom[Site].real() * _Mom[Site].real() + _Mom[Site].imag() * _Mom[Site].imag();
            } else {
                NewK += _Mom[Site] * _Mom[Site];
            }
        }
        NewK *= 0.5;

        double unif = this->_Unif(this->_Rng);

        // std::cout << NewS << ", " << OldS << '\n';
        // std::cout << NewH + NewS << ", " << OldH + OldS << '\n';
        // std::cout << NewK + NewS.real() - (OldK + OldS.real()) << '\n';
        // std::cout << exp(-(NewH + NewS - (OldH + OldS)).real()) << " : " << unif << '\n';

        if (exp(-(NewK + NewS.real() - (OldK + OldS.real()))) > unif) {
            this->_McStats.setS(NewS);
        } else {
            this->_Field = _OldField;
            this->_McStats.setS(OldS);
        }

        // Update montecarlo stats
    }
};

/* Argument deduction guide */
template <HmcCapable Action>
HMC(std::string, Action&, uintvect<Action::Dims>, uint, std::string&) -> HMC<Action>;
}  // namespace reticolo::montecarlo