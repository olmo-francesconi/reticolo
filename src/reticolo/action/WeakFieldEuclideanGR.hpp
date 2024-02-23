/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <array>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  WeakFieldEuclideanGR Class Declaration
--------------------------------------------------------------------------------------------------*/

template <RealValue TField, RealValue TAction>
class WeakFieldEuclideanGR {
  public:
    /* Types and public action metadata */
    using FieldType = HField<TField>;  // Type of the field variables
    using ActionType = TAction;        // Return type fo the action
    const static int Dims = 4;         // Dimensions of the action
    const static int Stencil = 2;      // Minimum step size for multi-thread safety

    /* Action parameters */
    struct Params {
        double lambda;

        Params() : lambda(1.0){};
        Params(double lambda) : lambda(lambda){};
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
    WeakFieldEuclideanGR(double lambda) : p(lambda){};  // Parameter List
    WeakFieldEuclideanGR(Params par) : p(par){};        // Parameter struct

    /* Destructor*/
    ~WeakFieldEuclideanGR() = default;

    /* Sync with lattice */
    void lattice_sync(const Lattice<FieldType, 4>& field);

    /* Gloabal and local action computations */
    auto compute_S(const Lattice<FieldType, 4>& field) const -> ActionType;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, uint site) const -> ActionType;
    auto compute_dS_loc(const Lattice<TField, 4>& field, const TField& dphi, uint site) const -> ActionType;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables;

    /* Log stuff*/
    auto action_name() -> std::string { return "Relativistic Bose Gas (phi^4)"; };
    auto action_parameters() -> std::string {
        std::stringstream Res;
        Res << "[ lambda : " << std::format("{:4.1f}", p.lambda) << " ]";
        return Res.str();
    }

  private:
    /* Vector storing the current values of the curvature for each lattice point*/
    std::vector<TAction> _R;

    /* Compute the curvature at a specific lattice point */
    auto compute_R(const Lattice<FieldType, 4>& field, uint site) -> TAction;
};

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/
template <RealValue TField, RealValue TAction>
void WeakFieldEuclideanGR<TField, TAction>::lattice_sync(const Lattice<FieldType, 4>& field) {
    _R.resize(field.getNsites(), 0.0);

    for (uint Site = 0; Site < _R.size(); Site++) {
        _R[Site] = compute_R(field, Site);
    }
}

/*--------------------------------------------------------------------------------------------------
  Private methods Implementatin
--------------------------------------------------------------------------------------------------*/
template <RealValue TField, RealValue TAction>
auto WeakFieldEuclideanGR<TField, TAction>::compute_R(const Lattice<FieldType, 4>& field, uint site) -> TAction {
    // Compute local derivatives
    std::array<FieldType, 4> Dh;
    HField_math::diff(Dh[_t], field.next(site, _t), field[site]);
    HField_math::diff(Dh[_x], field.next(site, _x), field[site]);
    HField_math::diff(Dh[_y], field.next(site, _y), field[site]);
    HField_math::diff(Dh[_z], field.next(site, _z), field[site]);

    std::array<double, 4> DTrh;
    for (int Rho = 0; Rho < 4; Rho++) {
        DTrh[Rho] = Dh[Rho][MUNU_00] + Dh[Rho][MUNU_11] + Dh[Rho][MUNU_22] + Dh[Rho][MUNU_33];
    }

    double Ans1 = 0.0;
    double Ans2 = 0.0;
    double Ans3 = 0.0;
    double Ans4 = 0.0;

    // -1/4 (d_rho h_mu,nu * d_rho h_mu,nu)
    for (int Rho = 0; Rho < 4; Rho++) {
        Ans1 += Dh[Rho][MUNU_00] * Dh[Rho][MUNU_00] + Dh[Rho][MUNU_01] * Dh[Rho][MUNU_01] +
                Dh[Rho][MUNU_02] * Dh[Rho][MUNU_02] + Dh[Rho][MUNU_03] * Dh[Rho][MUNU_03] +
                Dh[Rho][MUNU_10] * Dh[Rho][MUNU_10] + Dh[Rho][MUNU_11] * Dh[Rho][MUNU_11] +
                Dh[Rho][MUNU_12] * Dh[Rho][MUNU_12] + Dh[Rho][MUNU_13] * Dh[Rho][MUNU_13] +
                Dh[Rho][MUNU_20] * Dh[Rho][MUNU_20] + Dh[Rho][MUNU_21] * Dh[Rho][MUNU_21] +
                Dh[Rho][MUNU_22] * Dh[Rho][MUNU_22] + Dh[Rho][MUNU_23] * Dh[Rho][MUNU_23] +
                Dh[Rho][MUNU_30] * Dh[Rho][MUNU_30] + Dh[Rho][MUNU_31] * Dh[Rho][MUNU_31] +
                Dh[Rho][MUNU_32] * Dh[Rho][MUNU_32] + Dh[Rho][MUNU_33] * Dh[Rho][MUNU_33];
    }
    Ans1 *= -0.25;

    // +1/4 (d_mu Tr(h) * d_mu Tr(h))
    Ans2 = +0.25 * (DTrh[0] * DTrh[0] + DTrh[1] * DTrh[1] + DTrh[2] * DTrh[2] + DTrh[3] * DTrh[3]);

    // -1/2 (d_nu h_mu,nu * d_mu Tr(h))
    Ans3 = -0.5 *
           (Dh[0][MUNU_00] * DTrh[0] + Dh[0][MUNU_10] * DTrh[1] + Dh[0][MUNU_20] * DTrh[2] + Dh[0][MUNU_30] * DTrh[3] +
            Dh[1][MUNU_01] * DTrh[0] + Dh[1][MUNU_11] * DTrh[1] + Dh[1][MUNU_21] * DTrh[2] + Dh[1][MUNU_31] * DTrh[3] +
            Dh[2][MUNU_02] * DTrh[0] + Dh[2][MUNU_12] * DTrh[1] + Dh[2][MUNU_22] * DTrh[2] + Dh[2][MUNU_32] * DTrh[3] +
            Dh[3][MUNU_03] * DTrh[0] + Dh[3][MUNU_13] * DTrh[1] + Dh[3][MUNU_23] * DTrh[2] + Dh[3][MUNU_33] * DTrh[3]);

    // +1/2 (d_rho h_mu,nu * d_nu h_rho,mu)
    // rho = 0
    Ans4 += Dh[0][MUNU_00] * Dh[0][MUNU_00] + Dh[0][MUNU_01] * Dh[1][MUNU_00] + Dh[0][MUNU_02] * Dh[2][MUNU_00] +
            Dh[0][MUNU_03] * Dh[3][MUNU_00] + Dh[0][MUNU_10] * Dh[0][MUNU_01] + Dh[0][MUNU_11] * Dh[1][MUNU_01] +
            Dh[0][MUNU_12] * Dh[2][MUNU_01] + Dh[0][MUNU_13] * Dh[3][MUNU_01] + Dh[0][MUNU_20] * Dh[0][MUNU_02] +
            Dh[0][MUNU_21] * Dh[1][MUNU_02] + Dh[0][MUNU_22] * Dh[2][MUNU_02] + Dh[0][MUNU_23] * Dh[3][MUNU_02] +
            Dh[0][MUNU_30] * Dh[0][MUNU_03] + Dh[0][MUNU_31] * Dh[1][MUNU_03] + Dh[0][MUNU_32] * Dh[2][MUNU_03] +
            Dh[0][MUNU_33] * Dh[3][MUNU_03];
    // rho = 1
    Ans4 += Dh[1][MUNU_00] * Dh[0][MUNU_10] + Dh[1][MUNU_01] * Dh[1][MUNU_10] + Dh[1][MUNU_02] * Dh[2][MUNU_10] +
            Dh[1][MUNU_03] * Dh[3][MUNU_10] + Dh[1][MUNU_10] * Dh[0][MUNU_11] + Dh[1][MUNU_11] * Dh[1][MUNU_11] +
            Dh[1][MUNU_12] * Dh[2][MUNU_11] + Dh[1][MUNU_13] * Dh[3][MUNU_11] + Dh[1][MUNU_20] * Dh[0][MUNU_12] +
            Dh[1][MUNU_21] * Dh[1][MUNU_12] + Dh[1][MUNU_22] * Dh[2][MUNU_12] + Dh[1][MUNU_23] * Dh[3][MUNU_12] +
            Dh[1][MUNU_30] * Dh[0][MUNU_13] + Dh[1][MUNU_31] * Dh[1][MUNU_13] + Dh[1][MUNU_32] * Dh[2][MUNU_13] +
            Dh[1][MUNU_33] * Dh[3][MUNU_13];
    // rho = 2
    Ans4 += Dh[2][MUNU_00] * Dh[0][MUNU_20] + Dh[2][MUNU_01] * Dh[1][MUNU_20] + Dh[2][MUNU_02] * Dh[2][MUNU_20] +
            Dh[2][MUNU_03] * Dh[3][MUNU_20] + Dh[2][MUNU_10] * Dh[0][MUNU_21] + Dh[2][MUNU_11] * Dh[1][MUNU_21] +
            Dh[2][MUNU_12] * Dh[2][MUNU_21] + Dh[2][MUNU_13] * Dh[3][MUNU_21] + Dh[2][MUNU_20] * Dh[0][MUNU_22] +
            Dh[2][MUNU_21] * Dh[1][MUNU_22] + Dh[2][MUNU_22] * Dh[2][MUNU_22] + Dh[2][MUNU_23] * Dh[3][MUNU_22] +
            Dh[2][MUNU_30] * Dh[0][MUNU_23] + Dh[2][MUNU_31] * Dh[1][MUNU_23] + Dh[2][MUNU_32] * Dh[2][MUNU_23] +
            Dh[2][MUNU_33] * Dh[3][MUNU_23];
    // rho = 3
    Ans4 += Dh[3][MUNU_00] * Dh[0][MUNU_30] + Dh[3][MUNU_01] * Dh[1][MUNU_30] + Dh[3][MUNU_02] * Dh[2][MUNU_30] +
            Dh[3][MUNU_03] * Dh[3][MUNU_30] + Dh[3][MUNU_10] * Dh[0][MUNU_31] + Dh[3][MUNU_11] * Dh[1][MUNU_31] +
            Dh[3][MUNU_12] * Dh[2][MUNU_31] + Dh[3][MUNU_13] * Dh[3][MUNU_31] + Dh[3][MUNU_20] * Dh[0][MUNU_32] +
            Dh[3][MUNU_21] * Dh[1][MUNU_32] + Dh[3][MUNU_22] * Dh[2][MUNU_32] + Dh[3][MUNU_23] * Dh[3][MUNU_32] +
            Dh[3][MUNU_30] * Dh[0][MUNU_33] + Dh[3][MUNU_31] * Dh[1][MUNU_33] + Dh[3][MUNU_32] * Dh[2][MUNU_33] +
            Dh[3][MUNU_33] * Dh[3][MUNU_33];
    Ans4 *= 0.5;

    // std::cout << ans1 << "\t" << ans2 << "\t" << ans3 << "\t" << ans4 << "\t" << ans1 + ans2 + ans3 + ans4 <<
    // std::endl;
    std::cout << site << " : " << std::format("{:8e}", Ans1 + Ans2 + Ans3 + Ans4) << '\n';
    return {Ans1 + Ans2 + Ans3 + Ans4};
}
}  // namespace reticolo::action