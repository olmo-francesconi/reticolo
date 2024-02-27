/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <unistd.h>

#include <array>
#include <format>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/metropolis.hpp"
#include "reticolo/physics/constants.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"
#include "reticolo/types/random.hpp"

namespace pc = reticolo::PhysicalConstants;

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  WeakFieldEuclideanGR Class Declaration
--------------------------------------------------------------------------------------------------*/

class WeakFieldEuclideanGR {
  public:
    /* Types and public action metadata */
    using FieldType = reticolo::HField<RealD>;  // Type of the field variables
    using ActionType = RealD;                   // Return type fo the action
    const static int Dims = 4;                  // Dimensions of the action
    const static int Stencil = 2;               // Minimum step size for multi-thread safety

    /* Action parameters */
    struct Params {
        double beta;

        Params() : beta(5.7){};
        Params(double beta) : beta(beta){};
    } p;

    /* Observables */
    struct Observables {
        double R;

        [[nodiscard]] auto dump_str() -> std::string { return std::format("{:+8e}", R); }
        [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({R}); }
    };
    static auto make_obs_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(Observables));
        Type.insertMember("R", HOFFSET(Observables, R), H5::PredType::NATIVE_DOUBLE);
        return Type;
    }

    /* Constructors */
    WeakFieldEuclideanGR() = default;                              // Default
    WeakFieldEuclideanGR(WeakFieldEuclideanGR& other) = default;   // Copy
    WeakFieldEuclideanGR(WeakFieldEuclideanGR&& other) = default;  // Move

    /* Initializer Construtors */
    WeakFieldEuclideanGR(double beta) : p(beta){};  // Parameter List
    WeakFieldEuclideanGR(Params par) : p(par){};    // Parameter struct

    /* Destructor*/
    ~WeakFieldEuclideanGR() = default;

    /* Sync with lattice */
    void lattice_sync(const Lattice<FieldType, 4>& field);

    /* Gloabal and local action computations */
    auto compute_S(const Lattice<FieldType, 4>& field) -> ActionType;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, uint site) -> ActionType;
    auto compute_dS_loc(const Lattice<FieldType, 4>& field, const FieldType& dphi, uint site) -> ActionType;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables { return {0}; };

    /* Log stuff*/
    static auto action_name() -> std::string { return "Relativistic Bose Gas (phi^4)"; };
    auto        action_parameters() -> std::string {
        std::stringstream Res;
        Res << "[ beta : " << std::format("{:4.1f}", p.beta) << " ]";
        return Res.str();
    }
    /*--------------------------------------------------------------------------
      Custom variables and methods for WeakFieldEuclideanGR Class
    --------------------------------------------------------------------------*/

    /* Physical parameters */
    const double hbarc_GeVfm = pc::hbar * pc::c / pc::e * 1e15 * 1e-9;       // ~0.197 GeV fm
    const double GN_fm2 = 6.70883e-39 * pow(hbarc_GeVfm, 2);                 // ~2.6e-40 fm^2
    const double lP_fm = sqrt(GN_fm2);                                       // ~1.6e-20 fm
    const double kappa_GeV2 = pow(hbarc_GeVfm, 2) / (16.0 * M_PI * GN_fm2);  // ~3e-36 GeV^2

    double aa;
    double a_invGeV;
    double kappa;

    /* Vector storing the current values of the curvature for each lattice point*/
    std::vector<ActionType>          _LGR;
    std::vector<std::array<uint, 5>> _Checks;

    /* compute the check indexes */
    void make_checks(const Lattice<FieldType, 4>& field);

    /* Compute the curvature at a specific lattice point */
    void               update_LGR(const Lattice<FieldType, 4>& field);
    [[nodiscard]] auto compute_LGR_loc(const Lattice<FieldType, 4>& field, uint site) const -> ActionType;
    auto               check_pos_R(uint site, std::vector<ActionType> RPost, ActionType DR = 0.0) -> bool;
};

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/

auto WeakFieldEuclideanGR::compute_S(const Lattice<FieldType, 4>& field) -> ActionType {
    // Update the lattice
    update_LGR(field);
    // Accumulate the Lagrangian
    ActionType STot = std::reduce(_LGR.begin(), _LGR.end());

    return STot;
}

void WeakFieldEuclideanGR::lattice_sync(const Lattice<FieldType, 4>& field) {
    // Initialize parameters values here
    aa = 0.5 * exp(-1.6804 - 1.7331 * (p.beta - 6.0) + 0.7849 * pow(p.beta - 6.0, 2) - 0.4428 * pow(p.beta - 6.0, 3));
    a_invGeV = aa / hbarc_GeVfm;
    kappa = kappa_GeV2 * a_invGeV * a_invGeV;

    // Resize and cleat the curvature latice to match the lattice sizes
    _LGR.resize(field.getNsites(), 0.0);

    // Build the check indexing vector
    make_checks(field);

    // Updates the curvature values
    update_LGR(field);
}

/*--------------------------------------------------------------------------------------------------
  Private methods Implementation
--------------------------------------------------------------------------------------------------*/

void WeakFieldEuclideanGR::make_checks(const Lattice<FieldType, 4>& field) {
    _Checks.clear();
    for (uint Site = 0; Site < field.getNsites(); Site++) {
        _Checks.push_back({
            Site,                    //
            field.prevId(Site, _t),  //
            field.prevId(Site, _x),  //
            field.prevId(Site, _y),  //
            field.prevId(Site, _z)   //
        });
    }
}

void WeakFieldEuclideanGR::update_LGR(const Lattice<FieldType, 4>& field) {
    for (uint Site = 0; Site < _LGR.size(); Site++) {
        _LGR[Site] = compute_LGR_loc(field, Site);
    }
}

auto WeakFieldEuclideanGR::compute_LGR_loc(const Lattice<FieldType, 4>& field, uint site) const -> ActionType {
    // Compute local derivatives
    std::array<FieldType, 4> Dhmn;
    HField_math::diff(Dhmn[_t], field.next(site, _t), field[site]);
    HField_math::diff(Dhmn[_x], field.next(site, _x), field[site]);
    HField_math::diff(Dhmn[_y], field.next(site, _y), field[site]);
    HField_math::diff(Dhmn[_z], field.next(site, _z), field[site]);

    std::array<double, 4> DTrh;
    for (int Rho = 0; Rho < 4; Rho++) {
        DTrh[Rho] = Dhmn[Rho][MUNU_00] + Dhmn[Rho][MUNU_11] + Dhmn[Rho][MUNU_22] + Dhmn[Rho][MUNU_33];
    }

    double Ans1 = 0.0;
    double Ans2 = 0.0;
    double Ans3 = 0.0;
    double Ans4 = 0.0;

    // -1/4 (d_rho h_mu,nu * d_rho h_mu,nu)
    for (int Rho = 0; Rho < 4; Rho++) {
        Ans1 += Dhmn[Rho][MUNU_00] * Dhmn[Rho][MUNU_00] + Dhmn[Rho][MUNU_01] * Dhmn[Rho][MUNU_01] +
                Dhmn[Rho][MUNU_02] * Dhmn[Rho][MUNU_02] + Dhmn[Rho][MUNU_03] * Dhmn[Rho][MUNU_03] +
                Dhmn[Rho][MUNU_10] * Dhmn[Rho][MUNU_10] + Dhmn[Rho][MUNU_11] * Dhmn[Rho][MUNU_11] +
                Dhmn[Rho][MUNU_12] * Dhmn[Rho][MUNU_12] + Dhmn[Rho][MUNU_13] * Dhmn[Rho][MUNU_13] +
                Dhmn[Rho][MUNU_20] * Dhmn[Rho][MUNU_20] + Dhmn[Rho][MUNU_21] * Dhmn[Rho][MUNU_21] +
                Dhmn[Rho][MUNU_22] * Dhmn[Rho][MUNU_22] + Dhmn[Rho][MUNU_23] * Dhmn[Rho][MUNU_23] +
                Dhmn[Rho][MUNU_30] * Dhmn[Rho][MUNU_30] + Dhmn[Rho][MUNU_31] * Dhmn[Rho][MUNU_31] +
                Dhmn[Rho][MUNU_32] * Dhmn[Rho][MUNU_32] + Dhmn[Rho][MUNU_33] * Dhmn[Rho][MUNU_33];
    }
    Ans1 *= -0.25;

    // +1/4 (d_mu Tr(h) * d_mu Tr(h))
    Ans2 = +0.25 * (DTrh[0] * DTrh[0] + DTrh[1] * DTrh[1] + DTrh[2] * DTrh[2] + DTrh[3] * DTrh[3]);

    // -1/2 (d_nu h_mu,nu * d_mu Tr(h))
    Ans3 = -0.5 * (Dhmn[0][MUNU_00] * DTrh[0] + Dhmn[0][MUNU_10] * DTrh[1] + Dhmn[0][MUNU_20] * DTrh[2] +
                   Dhmn[0][MUNU_30] * DTrh[3] + Dhmn[1][MUNU_01] * DTrh[0] + Dhmn[1][MUNU_11] * DTrh[1] +
                   Dhmn[1][MUNU_21] * DTrh[2] + Dhmn[1][MUNU_31] * DTrh[3] + Dhmn[2][MUNU_02] * DTrh[0] +
                   Dhmn[2][MUNU_12] * DTrh[1] + Dhmn[2][MUNU_22] * DTrh[2] + Dhmn[2][MUNU_32] * DTrh[3] +
                   Dhmn[3][MUNU_03] * DTrh[0] + Dhmn[3][MUNU_13] * DTrh[1] + Dhmn[3][MUNU_23] * DTrh[2] +
                   Dhmn[3][MUNU_33] * DTrh[3]);

    // +1/2 (d_rho h_mu,nu * d_nu h_rho,mu)
    // rho = 0
    Ans4 += Dhmn[0][MUNU_00] * Dhmn[0][MUNU_00] + Dhmn[0][MUNU_01] * Dhmn[1][MUNU_00] +
            Dhmn[0][MUNU_02] * Dhmn[2][MUNU_00] + Dhmn[0][MUNU_03] * Dhmn[3][MUNU_00] +
            Dhmn[0][MUNU_10] * Dhmn[0][MUNU_01] + Dhmn[0][MUNU_11] * Dhmn[1][MUNU_01] +
            Dhmn[0][MUNU_12] * Dhmn[2][MUNU_01] + Dhmn[0][MUNU_13] * Dhmn[3][MUNU_01] +
            Dhmn[0][MUNU_20] * Dhmn[0][MUNU_02] + Dhmn[0][MUNU_21] * Dhmn[1][MUNU_02] +
            Dhmn[0][MUNU_22] * Dhmn[2][MUNU_02] + Dhmn[0][MUNU_23] * Dhmn[3][MUNU_02] +
            Dhmn[0][MUNU_30] * Dhmn[0][MUNU_03] + Dhmn[0][MUNU_31] * Dhmn[1][MUNU_03] +
            Dhmn[0][MUNU_32] * Dhmn[2][MUNU_03] + Dhmn[0][MUNU_33] * Dhmn[3][MUNU_03];
    // rho = 1
    Ans4 += Dhmn[1][MUNU_00] * Dhmn[0][MUNU_10] + Dhmn[1][MUNU_01] * Dhmn[1][MUNU_10] +
            Dhmn[1][MUNU_02] * Dhmn[2][MUNU_10] + Dhmn[1][MUNU_03] * Dhmn[3][MUNU_10] +
            Dhmn[1][MUNU_10] * Dhmn[0][MUNU_11] + Dhmn[1][MUNU_11] * Dhmn[1][MUNU_11] +
            Dhmn[1][MUNU_12] * Dhmn[2][MUNU_11] + Dhmn[1][MUNU_13] * Dhmn[3][MUNU_11] +
            Dhmn[1][MUNU_20] * Dhmn[0][MUNU_12] + Dhmn[1][MUNU_21] * Dhmn[1][MUNU_12] +
            Dhmn[1][MUNU_22] * Dhmn[2][MUNU_12] + Dhmn[1][MUNU_23] * Dhmn[3][MUNU_12] +
            Dhmn[1][MUNU_30] * Dhmn[0][MUNU_13] + Dhmn[1][MUNU_31] * Dhmn[1][MUNU_13] +
            Dhmn[1][MUNU_32] * Dhmn[2][MUNU_13] + Dhmn[1][MUNU_33] * Dhmn[3][MUNU_13];
    // rho = 2
    Ans4 += Dhmn[2][MUNU_00] * Dhmn[0][MUNU_20] + Dhmn[2][MUNU_01] * Dhmn[1][MUNU_20] +
            Dhmn[2][MUNU_02] * Dhmn[2][MUNU_20] + Dhmn[2][MUNU_03] * Dhmn[3][MUNU_20] +
            Dhmn[2][MUNU_10] * Dhmn[0][MUNU_21] + Dhmn[2][MUNU_11] * Dhmn[1][MUNU_21] +
            Dhmn[2][MUNU_12] * Dhmn[2][MUNU_21] + Dhmn[2][MUNU_13] * Dhmn[3][MUNU_21] +
            Dhmn[2][MUNU_20] * Dhmn[0][MUNU_22] + Dhmn[2][MUNU_21] * Dhmn[1][MUNU_22] +
            Dhmn[2][MUNU_22] * Dhmn[2][MUNU_22] + Dhmn[2][MUNU_23] * Dhmn[3][MUNU_22] +
            Dhmn[2][MUNU_30] * Dhmn[0][MUNU_23] + Dhmn[2][MUNU_31] * Dhmn[1][MUNU_23] +
            Dhmn[2][MUNU_32] * Dhmn[2][MUNU_23] + Dhmn[2][MUNU_33] * Dhmn[3][MUNU_23];
    // rho = 3
    Ans4 += Dhmn[3][MUNU_00] * Dhmn[0][MUNU_30] + Dhmn[3][MUNU_01] * Dhmn[1][MUNU_30] +
            Dhmn[3][MUNU_02] * Dhmn[2][MUNU_30] + Dhmn[3][MUNU_03] * Dhmn[3][MUNU_30] +
            Dhmn[3][MUNU_10] * Dhmn[0][MUNU_31] + Dhmn[3][MUNU_11] * Dhmn[1][MUNU_31] +
            Dhmn[3][MUNU_12] * Dhmn[2][MUNU_31] + Dhmn[3][MUNU_13] * Dhmn[3][MUNU_31] +
            Dhmn[3][MUNU_20] * Dhmn[0][MUNU_32] + Dhmn[3][MUNU_21] * Dhmn[1][MUNU_32] +
            Dhmn[3][MUNU_22] * Dhmn[2][MUNU_32] + Dhmn[3][MUNU_23] * Dhmn[3][MUNU_32] +
            Dhmn[3][MUNU_30] * Dhmn[0][MUNU_33] + Dhmn[3][MUNU_31] * Dhmn[1][MUNU_33] +
            Dhmn[3][MUNU_32] * Dhmn[2][MUNU_33] + Dhmn[3][MUNU_33] * Dhmn[3][MUNU_33];
    Ans4 *= 0.5;

    return kappa * (Ans1 + Ans2 + Ans3 + Ans4);
}
}  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  montecarlo::metropolis::sweep() Specialization
--------------------------------------------------------------------------------------------------*/

namespace reticolo {

template <>
void montecarlo::MetropolisWorker<action::WeakFieldEuclideanGR>::sweep() {
    uint       Acc = 0;       // acceptance
    ActionType SVarTot(0.0);  // cumulative action variation

    _McStats.softReset();

    for (uint Site = 0; Site < _Field.getNsites(); Site++) {
        // Generate a randomized local field variation
        FieldType FieldVar;  // local field variation
        FieldType FieldOld = _Field[Site];
        RealD     Scale = 0.1 * _Action.lP_fm / _Action.aa;
        randomize(FieldVar, Scale, _UnifC, _Rng);
        HField_math::sum(_Field[Site], _Field[Site], FieldVar);

        // Compute the updated Lagrangian in the surrounding sites
        std::vector<ActionType> LGRPost;   // Vector storing the new curvature values in the check sites
        ActionType              DS = 0.0;  // Cumulative curvature variation
        for (uint& CheckSite : _Action._Checks[Site]) {
            RealD Tmp = _Action.compute_LGR_loc(_Field, CheckSite);
            // std::cout << Tmp << " ";
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                DS += Tmp - _Action._LGR[CheckSite];
            }
        }
        // Metropolis acceptance + positive action
        if (LGRPost.size() == 5 && exp(-DS) > _Unif(_Rng)) {
            Acc++;
            SVarTot += DS;
            for (uint CheckSite = 0; CheckSite < 5; CheckSite++) {
                _Action._LGR[_Action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
            }
        } else {
            _Field[Site] = FieldOld;
        }
    }

    _McStats._Acceptance += Acc;
    _McStats._S += SVarTot;
    _McStats._DS += SVarTot;
}  // namespace reticolo

}  // namespace reticolo
