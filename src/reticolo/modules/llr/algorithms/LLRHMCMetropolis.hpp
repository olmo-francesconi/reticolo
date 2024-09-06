/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: mc/hmc.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <omp.h>

#include <memory>
#include <random>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/modules/montecarlo/algorithms/AlgorithmBase.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/random.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

template <class Action>
class LLRHMCMetropolis : public MCAlgorithmBase<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using monte_carlo_data_type = MMonteCarlo::data<typename Action::ActionType>;

    /* HMC support fields */
    std::unique_ptr<LatticeType> _Mom;
    std::unique_ptr<LatticeType> _Forces;
    std::unique_ptr<LatticeType> _OldField;

    /* HMC settings */
    double _Stepsize;
    uint   _Steps;

    /* Met settings */
    double _ProposalWidth;

    /* LLR parameters */
    double _Sk;
    double _Ak;
    double _Width;

    /* Distributions */
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0)
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]

  public:
    LLRHMCMetropolis() = default;

    /* setup */
    void setup(const YAML::Node& Params, const Lattice<FieldType>& Field) override {
        /* Parse Metropolis params */
        _ProposalWidth = Params["prop_width"].as<double>();

        /* Parse HMC params */
        _Stepsize = Params["stepsize"].as<double>();
        _Steps = Params["steps"].as<uint>();

        _Sk = Params["Sk"].as<double>();
        _Ak = Params["ak"].as<double>();
        _Width = Params["width"].as<double>();

        /* Allocate auxiliary fields */
        _Mom = std::make_unique<Lattice<FieldType>>(Field);
        _Forces = std::make_unique<Lattice<FieldType>>(Field);
        _OldField = std::make_unique<Lattice<FieldType>>(Field);
    }

    /* execution */
    void updateField(Lattice<FieldType>& field, Action& action, monte_carlo_data_type& state, RNGType& rng) override {
        // int NSites = field.getNsites();
        // // save the old field configuration;
        // *_OldField = field;
        // // Generate random momenta and compute initial kinetic term
        // RealD OldK(0.0);
        // for (int Site = 0; Site < NSites; Site++) {
        //     // randomize(_Mom[Site], 1.0, _Norm, _Rng);
        //     OldK += dot((*_Mom)[Site]);
        // }
        // OldK *= 0.5;
        // // Compute Forces
        // action.compute_Forces(field, (*_Forces));
        // // Momenta half step
        // for (int Site = 0; Site < NSites; Site++) {
        //     (*_Mom)[Site] -= 0.5 * _Stepsize * (*_Forces)[Site];
        // }
        // // Leapfrog algorithm
        // for (uint Step = 0; Step < _Steps; Step++) {
        //     // Update field
        //     for (int Site = 0; Site < NSites; Site++) {
        //         field[Site] += _Stepsize * (*_Mom)[Site];
        //     }
        //     // Compute updated forces
        //     action.compute_Forces(field, (*_Forces));
        //     // Update momenta
        //     for (int Site = 0; Site < NSites; Site++) {
        //         (*_Mom)[Site] -= _Stepsize * (*_Forces)[Site];
        //     }
        // }
        // // Momenta half step roll-back
        // for (int Site = 0; Site < NSites; Site++) {
        //     (*_Mom)[Site] += 0.5 * _Stepsize * (*_Forces)[Site];
        // }
        // // Compute final action
        // ActionType NewS = action.compute_S(field);
        // // Compute end kinetic term
        // RealD NewK(0.0);
        // for (int Site = 0; Site < NSites; Site++) {
        //     NewK += dot((*_Mom)[Site]);
        // }
        // NewK *= 0.5;
        // // Final Metropolis check
        // // if (exp(OldK - NewK + make_real(_McStats.getS() - NewS)) > m_Unif(_Rng)) {
        // //     _McStats.update(1, NewS - _McStats.getS());
        // // } else {
        // //     _Field = _OldField;
        // //     _McStats.update(0, 0.0);
        // // }
    }
};

}  // namespace reticolo::MMonteCarlo

// template <HmcCapable Action>
// class HMC : public MonteCarloHandler<Action> {
//   private:
//     /* Types definitions */
//     using ActionType = typename Action::ActionType;
//     using FieldType = typename Action::FieldType;
//     using LatticeType = Lattice<typename Action::FieldType>;

//     /* HMC support fields */
//     LatticeType _Mom;
//     LatticeType _Forces;
//     LatticeType _OldField;

//     /* HMC settings */
//     double _Stepsize;
//     uint   _Steps;

//     /* Introduced names from MonteCarloHandler (avoids using this->...) */
//     using MonteCarloHandler<Action>::_Field;
//     using MonteCarloHandler<Action>::_Action;
//     using MonteCarloHandler<Action>::_McStats;
//     using MonteCarloHandler<Action>::_Unif;
//     using MonteCarloHandler<Action>::_UnifC;
//     using MonteCarloHandler<Action>::_Norm;
//     using MonteCarloHandler<Action>::_Rng;
//     using MonteCarloHandler<Action>::_Logger;
//     using MonteCarloHandler<Action>::_Timer;

//   public:
//     /* Constructor */
//     HMC(std::string handler_name, Action& action, LatticeType& field, uint seed, const std::string& output_path,
//         bool StdOut, bool save_data, bool save_config);

//     /* override virtual updateField() method */
//     void updateField() override;

//     /* Initialize parameters */
//     void setParams(double traj_length, uint steps) {
//         _Steps = steps;
//         _Stepsize = traj_length / _Steps;
//     }
// };

// template <HmcCapable Action>
// HMC<Action>::HMC(std::string        handler_name,  //
//                  Action&            action,        //
//                  LatticeType&       field,         //
//                  uint               seed,          //
//                  const std::string& output_path,   //
//                  bool               StdOut,        //
//                  bool               save_data,     //
//                  bool               save_config)
//     : montecarlo::MonteCarloHandler<Action>(output_path,    //
//                                             handler_name,   //
//                                             action, field,  //
//                                             seed,           //
//                                             StdOut,         //
//                                             save_data,      //
//                                             save_config),
//       _Mom(field.getSizes()),
//       _Forces(field.getSizes()),
//       _OldField(field.getSizes()) {
//     // Log stuff
//     _Logger << IO::LI_time() +
//                    std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _Timer.elapsed_ms());
// }

// template <HmcCapable Action>
// void HMC<Action>::updateField() {
//     int NSites = _Field.getNsites();
//     // save the old field configuration;
//     _OldField = _Field;
//     // Generate random momenta and compute initial kinetic term
//     RealD OldK(0.0);
//     for (int Site = 0; Site < NSites; Site++) {
//         randomize(_Mom[Site], 1.0, _Norm, _Rng);
//         OldK += dot(_Mom[Site]);
//     }
//     OldK *= 0.5;
//     // Compute Forces
//     _Action.compute_Forces(_Field, _Forces);
//     // Momenta half step
//     for (int Site = 0; Site < NSites; Site++) {
//         _Mom[Site] -= 0.5 * _Stepsize * _Forces[Site];
//     }
//     // Leapfrog algorithm
//     for (uint Step = 0; Step < _Steps; Step++) {
//         // Update field
//         for (int Site = 0; Site < NSites; Site++) {
//             _Field[Site] += _Stepsize * _Mom[Site];
//         }
//         // Compute updated forces
//         _Action.compute_Forces(_Field, _Forces);
//         // Update momenta
//         for (int Site = 0; Site < NSites; Site++) {
//             _Mom[Site] -= _Stepsize * _Forces[Site];
//         }
//     }
//     // Momenta half step roll-back
//     for (int Site = 0; Site < _Mom.getNsites(); Site++) {
//         _Mom[Site] += 0.5 * _Stepsize * _Forces[Site];
//     }
//     // Compute final action
//     ActionType NewS = _Action.compute_S(_Field);
//     // Compute end kinetic term
//     RealD NewK(0.0);
//     for (int Site = 0; Site < _Mom.getNsites(); Site++) {
//         NewK += dot(_Mom[Site]);
//     }
//     NewK *= 0.5;
//     // Final Metropolis check
//     if (exp(OldK - NewK + make_real(_McStats.getS() - NewS)) > m_Unif(_Rng)) {
//         _McStats.update(1, NewS - _McStats.getS());
//     } else {
//         _Field = _OldField;
//         _McStats.update(0, 0.0);
//     }
// }

// /* Argument deduction guide */
// template <HmcCapable Action>
// HMC(std::string, Action&, Lattice<typename Action::FieldType>, uint, std::string&, bool, bool, bool) -> HMC<Action>;
// }  // namespace reticolo::montecarlo
