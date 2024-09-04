/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <omp.h>

#include <memory>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/modules/montecarlo/algorithms/AlgorithmBase.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::montecarlo {

template <class Action>
class LLRHMC : public AlgorithmBase<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using monte_carlo_data_type = montecarlo::data<typename Action::ActionType>;

    /* HMC support fields */
    std::unique_ptr<LatticeType> _Mom;
    std::unique_ptr<LatticeType> _Forces;
    std::unique_ptr<LatticeType> _OldField;

    /* HMC settings */
    double _Stepsize;
    uint   _Steps;

    /* LLR parameters */
    double _Sk;
    double _Ak;
    double _Width;

  public:
    LLRHMC() = default;

    /* setup */
    void setup(const YAML::Node& HMCParams, const Lattice<FieldType>& Field) override {
        _Stepsize = HMCParams["stepsize"].as<double>();
        _Steps = HMCParams["steps"].as<uint>();

        _Mom = std::make_unique<Lattice<FieldType>>(Field);
        _Forces = std::make_unique<Lattice<FieldType>>(Field);
        _OldField = std::make_unique<Lattice<FieldType>>(Field);
    }

    /* execution */
    void updateField(Lattice<FieldType>& field, Action& action, monte_carlo_data_type& state) override {
        // int NSites = field.getNsites();
        // // save the old field configuration;
        // *_OldField = field;
        // // Generate random momenta and compute initial kinetic term
        // RealD OldK(0.0);
        // for (int Site = 0; Site < NSites; Site++) {
        //     // randomize(_Mom[Site], 1.0, _Norm, _Rng);
        //     OldK += dot(_Mom[Site]);
        // }
        // OldK *= 0.5;
        // // Compute Forces
        // action.compute_LLRForces(field, *_Forces, _Sk, _Width, _Ak);
        // // Momenta half step
        // for (int Site = 0; Site < NSites; Site++) {
        //     _Mom[Site] -= 0.5 * _Stepsize * _Forces[Site];
        // }
        // // Leapfrog algorithm
        // for (uint Step = 0; Step < _Steps; Step++) {
        //     // Update field
        //     for (int Site = 0; Site < NSites; Site++) {
        //         field[Site] += _Stepsize * _Mom[Site];
        //     }
        //     // Compute updated forces
        //     action.compute_LLRForces(field, _Forces, _Sk, _Width, _Ak);
        //     // Update momenta
        //     for (int Site = 0; Site < NSites; Site++) {
        //         _Mom[Site] -= _Stepsize * _Forces[Site];
        //     }
        // }
        // // Momenta half step roll-back
        // for (int Site = 0; Site < _Mom.getNsites(); Site++) {
        //     _Mom[Site] += 0.5 * _Stepsize * _Forces[Site];
        // }
        // // Compute final action
        // // ActionType OldS = m_McStats.getS();
        // ActionType NewS = action.compute_S(field);
        // // Compute end kinetic term
        // RealD NewK(0.0);
        // for (int Site = 0; Site < _Mom.getNsites(); Site++) {
        //     NewK += dot(_Mom[Site]);
        // }
        // NewK *= 0.5;
        // // RealD SVarFull = NewS.real() - OldS.real() +                                            //
        // //                  (NewS.imag() - _Sk) * (NewS.imag() - _Sk) / (2.0 * _Width * _Width) -  //
        // //                  (OldS.imag() - _Sk) * (OldS.imag() - _Sk) / (2.0 * _Width * _Width) +
        // //                  _Ak * (NewS.imag() - OldS.imag());

        // // // Final Metropolis check
        // // if (exp(OldK - NewK - SVarFull) > m_Unif(m_Rng)) {
        // //     m_McStats.update(1, NewS - m_McStats.getS());
        // // } else {
        // //     field = _OldField;
        // //     m_McStats.update(0, 0.0);
        // // }
    };
};

}  // namespace reticolo::montecarlo
