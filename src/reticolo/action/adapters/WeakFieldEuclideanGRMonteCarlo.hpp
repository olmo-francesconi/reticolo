/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/adapters/WeakFieldEuclideanGRMonteCarlo.hpp

*******************************************************************************/

#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo {

template <>
inline void MMonteCarlo::Metropolis<action::WeakFieldEuclideanGR<RealF>>::updateField(
    lattice_type&                        field,   //
    action::WeakFieldEuclideanGR<RealF>& action,  //
    monte_carlo_data_type&               state,   //
    std::mt19937_64&                     rng)     //
{
    impl_type MrsglU;   // Marsaglia polar method support variables
    impl_type MrsglV;   //
    impl_type MrsglS;   //
    impl_type MrsglFP;  //
    impl_type Scale = _ProposalWidth * action._LPFm / action._AA;

    size_type   Acc = 0;       // acceptance
    action_type SVarTot(0.0);  // cumulative action variation

    for (const auto& Site : action._ToUpdate) {
        field_type FieldOld = field[Site];

        for (int i = 0; i < 10; i++) {
            do {
                MrsglU = _Unifc(rng);
                MrsglV = _Unifc(rng);
                MrsglS = MrsglU * MrsglU + MrsglV * MrsglV;
            } while (MrsglS > 1 || MrsglS == 0);
            MrsglFP = std::sqrt(-2 * std::log(MrsglS) / MrsglS);
            field[Site][i] += MrsglV * MrsglFP * Scale;
            field[Site][++i] += MrsglU * MrsglFP * Scale;
        }

        std::vector<action_type> LGRPost;
        action_type              ActionChange = 0.0;
        for (size_type& CheckSite : action._Checks[Site]) {
            impl_type Tmp = action.compute_LGR_loc(field, CheckSite);
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - action._LGR[CheckSite];
            } else {
                break;
            }
        }

        if (LGRPost.size() == 5 && exp(-ActionChange) > _Unif(rng)) {
            Acc++;
            SVarTot += ActionChange;
            for (size_type CheckSite = 0; CheckSite < 5; CheckSite++) {
                action._LGR[action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
            }
        } else {
            field[Site] = FieldOld;
        }
    }
    state.update(static_cast<impl_type>(Acc) / field.getNsites(), SVarTot);
}

template <>
inline void MMonteCarlo::Metropolis<action::WeakFieldEuclideanGR<RealD>, std::mt19937_64>::updateField(
    lattice_type&                        field,   //
    action::WeakFieldEuclideanGR<RealD>& action,  //
    monte_carlo_data_type&               state,   //
    std::mt19937_64&                     rng)     //
{
    impl_type MrsglU;   // Marsaglia polar method support variables
    impl_type MrsglV;   //
    impl_type MrsglS;   //
    impl_type MrsglFP;  //
    impl_type Scale = _ProposalWidth * action._LPFm / action._AA;

    size_type   Acc = 0;
    action_type SVarTot(0.0);

    field_type FieldOld;

    for (const auto& Site : action._ToUpdate) {
        FieldOld = field[Site];

        for (int i = 0; i < 10; i++) {
            do {
                MrsglU = _Unifc(rng);
                MrsglV = _Unifc(rng);
                MrsglS = MrsglU * MrsglU + MrsglV * MrsglV;
            } while (MrsglS > 1 || MrsglS == 0);
            MrsglFP = std::sqrt(-2 * std::log(MrsglS) / MrsglS);
            field[Site][i] += MrsglV * MrsglFP * Scale;
            field[Site][++i] += MrsglU * MrsglFP * Scale;
        }

        std::vector<action_type> LGRPost;
        action_type              ActionChange = 0.0;
        for (size_type& CheckSite : action._Checks[Site]) {
            impl_type Tmp = action.compute_LGR_loc(field, CheckSite);
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - action._LGR[CheckSite];
            } else {
                break;
            }
        }

        if (LGRPost.size() == 5 && exp(-ActionChange) > _Unif(rng)) {
            Acc++;
            SVarTot += ActionChange;
            for (size_type CheckSite = 0; CheckSite < 5; CheckSite++) {
                action._LGR[action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
            }
        } else {
            field[Site] = FieldOld;
        }
    }
    state.update(static_cast<impl_type>(Acc) / field.getNsites(), SVarTot);
}

}  // namespace reticolo
