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
class LLRMetropolis : public AlgorithmBase<Action> {
  private:
    /* Types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using monte_carlo_data_type = montecarlo::data<typename Action::ActionType>;

    /* Metropolis parameters */
    double _ProposalWidth;

    /* LLR parameters */
    double _Sk;
    double _Ak;
    double _Width;

  public:
    LLRMetropolis() = default;

    /* setup */
    void setup(const YAML::Node& HMCParams, const Lattice<FieldType>& Field) override {
        _ProposalWidth = HMCParams["prop_width"].as<double>();
    }

    /* execution */
    void updateField(Lattice<FieldType>& field, Action& action, monte_carlo_data_type& state) override {
        //     uint Acc = 0;  // acceptance

        //     double WindowWeight;  // weight of the windowing function   [normal distribution]
        //     double LlrWeight;     // weight of the llr reweighting      [exp(-ak(S-Sk)]

        //     ActionType SVar;               // local action variation
        //     ActionType SVarTot(0.0, 0.0);  // cumulative action variation

        //     FieldType FieldVar;  // local field variation

        //     for (uint Site = 0; Site < field.getNsites(); Site++) {
        //         // randomize(FieldVar, _ProposalWidth, _UnifC, _Rng);

        //         SVar = action.compute_dS_loc(field, FieldVar, Site);

        //         WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * state._SIm - 2.0 * _Sk)) / (2.0 * _Width *
        //         _Width);

        //         LlrWeight = _Ak * SVar.imag();

        //         // if (exp(-(SVar.real() + WindowWeight + LlrWeight)) > _Unif(_Rng)) {
        //         //     Acc++;
        //         //     field[Site] += FieldVar;
        //         //     SVarTot += SVar;
        //         //     state._SRe += SVar.real();
        //         //     state._SIm += SVar.imag();
        //         // }
        //     }

        //     state._Acceptance = static_cast<double>(Acc) / field.getNsites();
        //     state._DSRe = SVarTot.real();
        //     state._DSIm = SVarTot.imag();
    };
};
}  // namespace reticolo::montecarlo
