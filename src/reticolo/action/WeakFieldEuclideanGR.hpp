/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <format>
#include <numeric>
#include <string>
#include <vector>

#include "reticolo/action/action_base.hpp"
#include "reticolo/lattice/lattice.hpp"
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

class WeakFieldEuclideanGR : public action::ActionBase<HField<RealD>, RealD, 4> {
  public:
    /* Types and public action metadata */
    using FieldType = HField<RealD>;  // Type of the field variables
    using ActionType = RealD;         // Return type fo the action
    const static int Dims = 4;        // Dimensions of the action
    const static int Stencil = 2;     // Minimum step size for multi-thread safety

    /* Algorithm capabilities */
    const static bool IsMetropolisCapable = true;
    const static bool IsHmcCapable = true;

    /* Action parameters */
    struct Params {
        double beta;
        Params() : beta(5.7){};
        Params(double beta) : beta(beta){};
    } p;

    /* Observables */
    struct Observables {
        double R;
    };
    friend auto make_H5_Type<Observables>() {
        hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(Observables));
        H5Tinsert(DataTypeHid, "R", HOFFSET(Observables, R), H5T_NATIVE_DOUBLE);
        return DataTypeHid;
    }

    /* Constructors */
    WeakFieldEuclideanGR(Lattice<FieldType, 4>& field, double beta) : p(beta) { lattice_sync(field); };
    WeakFieldEuclideanGR(Lattice<FieldType, 4>& field, Params par) : p(par) { lattice_sync(field); };

    /* Destructor*/
    ~WeakFieldEuclideanGR() = default;

    /* Sync with lattice */
    void lattice_sync(const Lattice<FieldType, 4>& field) override;

    /* Gloabal and local action computations */
    auto compute_S(const Lattice<FieldType, 4>& field) -> ActionType override;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, int site) -> ActionType override;
    auto compute_dS_loc(const Lattice<FieldType, 4>& field, const FieldType& dphi, int site) -> ActionType override;

    /* HMC methods */
    void compute_Forces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& Forces) override;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables { return {0}; };

    /* Log stuff*/
    auto name() -> std::string override { return "Weak Field Euclidean General Relativity"; };
    auto name_short() -> std::string override { return "LQGR"; };

    auto parameters() -> std::string override {
        std::string ParamStr = std::format("[ beta : {:4.1f} ]", p.beta);
        return ParamStr;
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
    std::vector<ActionType>         _LGR;
    std::vector<std::array<int, 5>> _Checks;

    /* compute the check indexes */
    void make_checks(const Lattice<FieldType, 4>& field);

    /* Compute the curvature at a specific lattice point */
    void               update_LGR(const Lattice<FieldType, 4>& field);
    [[nodiscard]] auto compute_LGR_loc(const Lattice<FieldType, 4>& field, int site) const -> ActionType;
    auto               check_pos_R(int site, std::vector<ActionType> RPost, ActionType DR = 0.0) -> bool;
    static void        compute_Force_loc(const Lattice<FieldType, 4>& field, FieldType& force, int site);

};  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/
inline void WeakFieldEuclideanGR::lattice_sync(const Lattice<FieldType, 4>& field) {
    // Initialize parameters values here
    aa = 0.5 * exp(-1.6804 - 1.7331 * (p.beta - 6.0) + 0.7849 * pow(p.beta - 6.0, 2) - 0.4428 * pow(p.beta - 6.0, 3));
    a_invGeV = aa / hbarc_GeVfm;
    kappa = kappa_GeV2 * a_invGeV * a_invGeV;

    // Resize and clears the curvature latice to match the lattice sizes
    _LGR.resize(field.getNsites(), 0.0);

    // Build the check indexing vector
    make_checks(field);

    // Updates the curvature values
    update_LGR(field);
}

inline auto WeakFieldEuclideanGR::compute_S(const Lattice<FieldType, 4>& field) -> ActionType {
    // Update the lattice
    update_LGR(field);
    // Accumulate the Lagrangian
    ActionType STot = std::reduce(_LGR.begin(), _LGR.end());

    return STot;
}

inline auto WeakFieldEuclideanGR::compute_S_loc(const Lattice<FieldType, 4>& field, int site) -> ActionType {
    return 0.0;
};

inline auto WeakFieldEuclideanGR::compute_dS_loc(const Lattice<FieldType, 4>& field, const FieldType& dphi, int site)
    -> ActionType {
    return 0.0;
};

inline void WeakFieldEuclideanGR::compute_Forces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& forces) {
    std::array<FieldType, 4>                Dhc;  // Central derivatives
    std::array<std::array<FieldType, 4>, 4> Dhd;  // Diagonal derivatives
    std::array<FieldType, 4>                Jhc;  // Central sum
    std::array<std::array<FieldType, 4>, 4> Jhd;  // Diagonal Sum

    for (int Site = 0; Site < field.getNsites(); Site++) {
        // clang-format off
        // Compute central derivatives
        HField_math::diff(Dhc[_t], field.next(Site, _t), field.prev(Site, _t), 0.5);
        HField_math::diff(Dhc[_x], field.next(Site, _x), field.prev(Site, _x), 0.5);
        HField_math::diff(Dhc[_y], field.next(Site, _y), field.prev(Site, _y), 0.5);
        HField_math::diff(Dhc[_z], field.next(Site, _z), field.prev(Site, _z), 0.5);
        // Compute diagonal derivatives
        HField_math::diff(Dhd[_t][_x], field.next(field.prevId(Site, _x), _t), field.next(field.prevId(Site, _t), _x), 0.5);
        HField_math::diff(Dhd[_t][_y], field.next(field.prevId(Site, _y), _t), field.next(field.prevId(Site, _t), _y), 0.5);
        HField_math::diff(Dhd[_t][_z], field.next(field.prevId(Site, _z), _t), field.next(field.prevId(Site, _t), _z), 0.5);
        HField_math::diff(Dhd[_x][_t], field.next(field.prevId(Site, _t), _x), field.next(field.prevId(Site, _x), _t), 0.5);
        HField_math::diff(Dhd[_x][_y], field.next(field.prevId(Site, _y), _x), field.next(field.prevId(Site, _x), _y), 0.5);
        HField_math::diff(Dhd[_x][_z], field.next(field.prevId(Site, _z), _x), field.next(field.prevId(Site, _x), _z), 0.5);
        HField_math::diff(Dhd[_y][_t], field.next(field.prevId(Site, _t), _y), field.next(field.prevId(Site, _y), _t), 0.5);
        HField_math::diff(Dhd[_y][_x], field.next(field.prevId(Site, _x), _y), field.next(field.prevId(Site, _y), _x), 0.5);
        HField_math::diff(Dhd[_y][_z], field.next(field.prevId(Site, _z), _y), field.next(field.prevId(Site, _y), _z), 0.5);
        HField_math::diff(Dhd[_z][_t], field.next(field.prevId(Site, _t), _z), field.next(field.prevId(Site, _z), _t), 0.5);
        HField_math::diff(Dhd[_z][_x], field.next(field.prevId(Site, _x), _z), field.next(field.prevId(Site, _z), _x), 0.5);
        HField_math::diff(Dhd[_z][_y], field.next(field.prevId(Site, _y), _z), field.next(field.prevId(Site, _z), _y), 0.5);
        // Compute central sums
        HField_math::sum(Jhc[_t], field.next(Site, _t), field.prev(Site, _t), 0.5);
        HField_math::sum(Jhc[_x], field.next(Site, _x), field.prev(Site, _x), 0.5);
        HField_math::sum(Jhc[_y], field.next(Site, _y), field.prev(Site, _y), 0.5);
        HField_math::sum(Jhc[_z], field.next(Site, _z), field.prev(Site, _z), 0.5);
        // Compute diagonal sums
        HField_math::sum(Jhd[_t][_x], field.next(field.prevId(Site, _x), _t), field.next(field.prevId(Site, _t), _x), 0.5);
        HField_math::sum(Jhd[_t][_y], field.next(field.prevId(Site, _y), _t), field.next(field.prevId(Site, _t), _y), 0.5);
        HField_math::sum(Jhd[_t][_z], field.next(field.prevId(Site, _z), _t), field.next(field.prevId(Site, _t), _z), 0.5);
        HField_math::sum(Jhd[_x][_t], field.next(field.prevId(Site, _t), _x), field.next(field.prevId(Site, _x), _t), 0.5);
        HField_math::sum(Jhd[_x][_y], field.next(field.prevId(Site, _y), _x), field.next(field.prevId(Site, _x), _y), 0.5);
        HField_math::sum(Jhd[_x][_z], field.next(field.prevId(Site, _z), _x), field.next(field.prevId(Site, _x), _z), 0.5);
        HField_math::sum(Jhd[_y][_t], field.next(field.prevId(Site, _t), _y), field.next(field.prevId(Site, _y), _t), 0.5);
        HField_math::sum(Jhd[_y][_x], field.next(field.prevId(Site, _x), _y), field.next(field.prevId(Site, _y), _x), 0.5);
        HField_math::sum(Jhd[_y][_z], field.next(field.prevId(Site, _z), _y), field.next(field.prevId(Site, _y), _z), 0.5);
        HField_math::sum(Jhd[_z][_t], field.next(field.prevId(Site, _t), _z), field.next(field.prevId(Site, _z), _t), 0.5);
        HField_math::sum(Jhd[_z][_x], field.next(field.prevId(Site, _x), _z), field.next(field.prevId(Site, _z), _x), 0.5);
        HField_math::sum(Jhd[_z][_y], field.next(field.prevId(Site, _y), _z), field.next(field.prevId(Site, _z), _y), 0.5);
        // clang-format on

        // Diagonal Components
        forces[Site][MUNU_00] = 2.0 * field[Site][MUNU_11] - Jhc[_y][MUNU_11] - Jhc[_z][MUNU_11]                      //
                                + 2.0 * field[Site][MUNU_22] - Jhc[_x][MUNU_22] - Jhc[_z][MUNU_22]                    //
                                + 2.0 * field[Site][MUNU_33] - Jhc[_x][MUNU_33] - Jhc[_y][MUNU_33]                    //
                                - Dhc[_t][MUNU_01] + Dhc[_x][MUNU_01] + Dhd[_t][_x][MUNU_01]                          //
                                - Dhc[_t][MUNU_02] + Dhc[_y][MUNU_02] + Dhd[_t][_y][MUNU_02]                          //
                                - Dhc[_t][MUNU_03] + Dhc[_z][MUNU_03] + Dhd[_t][_z][MUNU_03]                          //
                                - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_y][MUNU_12] - Jhd[_x][_y][MUNU_12]   //
                                - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13]   //
                                - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

        forces[Site][MUNU_11] = 2.0 * field[Site][MUNU_00] - Jhc[_y][MUNU_00] - Jhc[_z][MUNU_00]                      //
                                + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_z][MUNU_22]                    //
                                + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_y][MUNU_33]                    //
                                + Dhc[_t][MUNU_01] - Dhc[_x][MUNU_01] - Dhd[_t][_x][MUNU_01]                          //
                                - Dhc[_x][MUNU_12] + Dhc[_y][MUNU_12] + Dhd[_x][_y][MUNU_12]                          //
                                - Dhc[_x][MUNU_13] + Dhc[_z][MUNU_13] + Dhd[_x][_z][MUNU_13]                          //
                                - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
                                - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
                                - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

        forces[Site][MUNU_22] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_z][MUNU_00]                      //
                                + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_z][MUNU_11]                    //
                                + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_x][MUNU_33]                    //
                                + Dhc[_t][MUNU_02] - Dhc[_y][MUNU_02] - Dhd[_t][_y][MUNU_02]                          //
                                + Dhc[_x][MUNU_12] - Dhc[_y][MUNU_12] - Dhd[_x][_y][MUNU_12]                          //
                                - Dhc[_y][MUNU_23] + Dhc[_z][MUNU_23] + Dhd[_y][_z][MUNU_23]                          //
                                - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
                                - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
                                - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13];  //

        forces[Site][MUNU_33] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_y][MUNU_00]                      //
                                + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_y][MUNU_11]                    //
                                + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_x][MUNU_22]                    //
                                + Dhc[_t][MUNU_03] - Dhc[_z][MUNU_03] - Dhd[_t][_z][MUNU_03]                          //
                                + Dhc[_x][MUNU_13] - Dhc[_z][MUNU_13] - Dhd[_x][_z][MUNU_13]                          //
                                + Dhc[_y][MUNU_23] - Dhc[_z][MUNU_23] - Dhd[_y][_z][MUNU_23]                          //
                                - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
                                - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
                                - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_z][MUNU_12] - Jhd[_x][_y][MUNU_12];  //

        // Time-Space Components
        // clang-format off
        forces[Site][MUNU_01] = Dhc[_t][MUNU_00] - Dhc[_x][MUNU_00] - Dhd[_t][_x][MUNU_00]                  //
                                - Dhc[_t][MUNU_11] + Dhc[_x][MUNU_11] + Dhd[_t][_x][MUNU_11]                //
                                + Jhc[_t][MUNU_22] + Jhc[_x][MUNU_22] - Jhd[_t][_x][MUNU_22]                //
                                + Jhc[_t][MUNU_33] + Jhc[_x][MUNU_33] - Jhd[_t][_x][MUNU_33]                //
                                - 4.0 * field[Site][MUNU_01] - field[Site][MUNU_22] - field[Site][MUNU_33]  //
                                + 2.0 * (Jhc[_y][MUNU_01] + Jhc[_z][MUNU_01])                               //
                                + field[Site][MUNU_02] - field.next(Site, _x)[MUNU_02] - field.prev(Site, _y)[MUNU_02] + field.next(field.prevId(Site, _y), _x)[MUNU_02]  //
                                + field[Site][MUNU_03] - field.next(Site, _x)[MUNU_03] - field.prev(Site, _z)[MUNU_03] + field.next(field.prevId(Site, _z), _x)[MUNU_03]  //
                                + field[Site][MUNU_12] - field.next(Site, _t)[MUNU_12] - field.prev(Site, _y)[MUNU_12] + field.next(field.prevId(Site, _y), _t)[MUNU_12]  //
                                + field[Site][MUNU_13] - field.next(Site, _t)[MUNU_13] - field.prev(Site, _z)[MUNU_13] + field.next(field.prevId(Site, _z), _t)[MUNU_13];

        forces[Site][MUNU_02] = Dhc[_t][MUNU_00] - Dhc[_y][MUNU_00] - Dhd[_t][_y][MUNU_00]                  //
                                - Dhc[_t][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_t][_y][MUNU_22]                //
                                + Jhc[_t][MUNU_11] + Jhc[_y][MUNU_11] - Jhd[_t][_y][MUNU_11]                //
                                + Jhc[_t][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_t][_y][MUNU_33]                //
                                - 4.0 * field[Site][MUNU_02] - field[Site][MUNU_11] - field[Site][MUNU_33]  //
                                + 2.0 * (Jhc[_x][MUNU_02] + Jhc[_z][MUNU_02])                               //
                                + field[Site][MUNU_01] - field.next(Site, _y)[MUNU_01] - field.prev(Site, _x)[MUNU_01] + field.next(field.prevId(Site, _x), _y)[MUNU_01]  //
                                + field[Site][MUNU_03] - field.next(Site, _y)[MUNU_03] - field.prev(Site, _z)[MUNU_03] + field.next(field.prevId(Site, _z), _y)[MUNU_03]  //
                                + field[Site][MUNU_12] - field.next(Site, _t)[MUNU_12] - field.prev(Site, _x)[MUNU_12] + field.next(field.prevId(Site, _x), _t)[MUNU_12]  //
                                + field[Site][MUNU_23] - field.next(Site, _t)[MUNU_23] - field.prev(Site, _z)[MUNU_23] + field.next(field.prevId(Site, _z), _t)[MUNU_23];

        forces[Site][MUNU_03] = Dhc[_t][MUNU_00] - Dhc[_z][MUNU_00] - Dhd[_t][_z][MUNU_00]                  //
                                - Dhc[_t][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_t][_z][MUNU_33]                //
                                + Jhc[_t][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_t][_z][MUNU_11]                //
                                + Jhc[_t][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_t][_z][MUNU_22]                //
                                - 4.0 * field[Site][MUNU_03] - field[Site][MUNU_11] - field[Site][MUNU_22]  //
                                + 2.0 * (Jhc[_x][MUNU_03] + Jhc[_y][MUNU_03])                               //
                                + field[Site][MUNU_01] - field.next(Site, _z)[MUNU_01] - field.prev(Site, _x)[MUNU_01] + field.next(field.prevId(Site, _x), _z)[MUNU_01]  //
                                + field[Site][MUNU_02] - field.next(Site, _z)[MUNU_02] - field.prev(Site, _y)[MUNU_02] + field.next(field.prevId(Site, _y), _z)[MUNU_02]  //
                                + field[Site][MUNU_13] - field.next(Site, _t)[MUNU_13] - field.prev(Site, _y)[MUNU_13] + field.next(field.prevId(Site, _y), _t)[MUNU_13]  //
                                + field[Site][MUNU_23] - field.next(Site, _t)[MUNU_23] - field.prev(Site, _y)[MUNU_23] + field.next(field.prevId(Site, _y), _t)[MUNU_23];

        // Space-Space Components
        forces[Site][MUNU_12] = Dhc[_x][MUNU_11] - Dhc[_y][MUNU_11] - Dhd[_x][_y][MUNU_11]                  //
                                - Dhc[_x][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_x][_y][MUNU_22]                //
                                + Jhc[_x][MUNU_00] + Jhc[_y][MUNU_00] - Jhd[_x][_y][MUNU_00]                //
                                + Jhc[_x][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_x][_y][MUNU_33]                //
                                - 4.0 * field[Site][MUNU_12] - field[Site][MUNU_00] - field[Site][MUNU_33]  //
                                + 2.0 * (Jhc[_t][MUNU_12] + Jhc[_z][MUNU_12])                               //
                                + field[Site][MUNU_01] - field.next(Site, _y)[MUNU_01] - field.prev(Site, _t)[MUNU_01] + field.next(field.prevId(Site, _t), _y)[MUNU_01]  //
                                + field[Site][MUNU_02] - field.next(Site, _x)[MUNU_02] - field.prev(Site, _t)[MUNU_02] + field.next(field.prevId(Site, _t), _x)[MUNU_02]  //
                                + field[Site][MUNU_13] - field.next(Site, _y)[MUNU_13] - field.prev(Site, _z)[MUNU_13] + field.next(field.prevId(Site, _z), _y)[MUNU_13]  //
                                + field[Site][MUNU_23] - field.next(Site, _x)[MUNU_23] - field.prev(Site, _z)[MUNU_23] + field.next(field.prevId(Site, _z), _x)[MUNU_23];

        forces[Site][MUNU_13] = Dhc[_x][MUNU_11] - Dhc[_z][MUNU_11] - Dhd[_x][_z][MUNU_11]                  //
                                - Dhc[_x][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_x][_z][MUNU_33]                //
                                + Jhc[_x][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_x][_z][MUNU_00]                //
                                + Jhc[_x][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_x][_z][MUNU_22]                //
                                - 4.0 * field[Site][MUNU_13] - field[Site][MUNU_00] - field[Site][MUNU_22]  //
                                + 2.0 * (Jhc[_t][MUNU_13] + Jhc[_y][MUNU_13])                               //
                                + field[Site][MUNU_01] - field.next(Site, _z)[MUNU_01] - field.prev(Site, _t)[MUNU_01] + field.next(field.prevId(Site, _t), _z)[MUNU_01]  //
                                + field[Site][MUNU_03] - field.next(Site, _x)[MUNU_03] - field.prev(Site, _t)[MUNU_03] + field.next(field.prevId(Site, _t), _x)[MUNU_03]  //
                                + field[Site][MUNU_12] - field.next(Site, _z)[MUNU_12] - field.prev(Site, _y)[MUNU_12] + field.next(field.prevId(Site, _y), _z)[MUNU_12]  //
                                + field[Site][MUNU_23] - field.next(Site, _x)[MUNU_23] - field.prev(Site, _y)[MUNU_23] + field.next(field.prevId(Site, _y), _x)[MUNU_23];

        forces[Site][MUNU_23] = Dhc[_y][MUNU_22] - Dhc[_z][MUNU_22] - Dhd[_y][_z][MUNU_22]                  //
                                - Dhc[_y][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_y][_z][MUNU_33]                //
                                + Jhc[_y][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_y][_z][MUNU_00]                //
                                + Jhc[_y][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_y][_z][MUNU_11]                //
                                - 4.0 * field[Site][MUNU_23] - field[Site][MUNU_00] - field[Site][MUNU_11]  //
                                + 2.0 * (Jhc[_t][MUNU_23] + Jhc[_x][MUNU_23])                               //
                                + field[Site][MUNU_02] - field.next(Site, _z)[MUNU_02] - field.prev(Site, _t)[MUNU_02] + field.next(field.prevId(Site, _t), _z)[MUNU_02]  //
                                + field[Site][MUNU_03] - field.next(Site, _y)[MUNU_03] - field.prev(Site, _t)[MUNU_03] + field.next(field.prevId(Site, _t), _y)[MUNU_03]  //
                                + field[Site][MUNU_12] - field.next(Site, _z)[MUNU_12] - field.prev(Site, _x)[MUNU_12] + field.next(field.prevId(Site, _x), _z)[MUNU_12]  //
                                + field[Site][MUNU_13] - field.next(Site, _y)[MUNU_13] - field.prev(Site, _x)[MUNU_13] + field.next(field.prevId(Site, _x), _y)[MUNU_13];
        // clang-format on

        forces[Site] *= kappa;
    }
}
/*--------------------------------------------------------------------------------------------------
      Custom variables and methods for WeakFieldEuclideanGR Class implementation
--------------------------------------------------------------------------------------------------*/

inline void WeakFieldEuclideanGR::make_checks(const Lattice<FieldType, 4>& field) {
    _Checks.clear();
    for (int Site = 0; Site < field.getNsites(); Site++) {
        _Checks.push_back({
            Site,                    //
            field.prevId(Site, _t),  //
            field.prevId(Site, _x),  //
            field.prevId(Site, _y),  //
            field.prevId(Site, _z)   //
        });
    }
}

inline void WeakFieldEuclideanGR::update_LGR(const Lattice<FieldType, 4>& field) {
    for (uint Site = 0; Site < _LGR.size(); Site++) {
        _LGR[Site] = compute_LGR_loc(field, Site);
    }
}

inline auto WeakFieldEuclideanGR::compute_LGR_loc(const Lattice<FieldType, 4>& field, int site) const -> ActionType {
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

inline void WeakFieldEuclideanGR::compute_Force_loc(const Lattice<FieldType, 4>& field, FieldType& force, int Site) {
    std::array<FieldType, 4>                Dhc;  // Central derivatives
    std::array<std::array<FieldType, 4>, 4> Dhd;  // Diagonal derivatives
    std::array<FieldType, 4>                Jhc;  // Central sum
    std::array<std::array<FieldType, 4>, 4> Jhd;  // Diagonal Sum

    // clang-format off
    // Compute central derivatives
    HField_math::diff(Dhc[_t], field.next(Site, _t), field.prev(Site, _t), 0.5);
    HField_math::diff(Dhc[_x], field.next(Site, _x), field.prev(Site, _x), 0.5);
    HField_math::diff(Dhc[_y], field.next(Site, _y), field.prev(Site, _y), 0.5);
    HField_math::diff(Dhc[_z], field.next(Site, _z), field.prev(Site, _z), 0.5);
    // Compute diagonal derivatives
    HField_math::diff(Dhd[_t][_x], field.next(field.prevId(Site, _x), _t), field.next(field.prevId(Site, _t), _x), 0.5);
    HField_math::diff(Dhd[_t][_y], field.next(field.prevId(Site, _y), _t), field.next(field.prevId(Site, _t), _y), 0.5);
    HField_math::diff(Dhd[_t][_z], field.next(field.prevId(Site, _z), _t), field.next(field.prevId(Site, _t), _z), 0.5);
    HField_math::diff(Dhd[_x][_t], field.next(field.prevId(Site, _t), _x), field.next(field.prevId(Site, _x), _t), 0.5);
    HField_math::diff(Dhd[_x][_y], field.next(field.prevId(Site, _y), _x), field.next(field.prevId(Site, _x), _y), 0.5);
    HField_math::diff(Dhd[_x][_z], field.next(field.prevId(Site, _z), _x), field.next(field.prevId(Site, _x), _z), 0.5);
    HField_math::diff(Dhd[_y][_t], field.next(field.prevId(Site, _t), _y), field.next(field.prevId(Site, _y), _t), 0.5);
    HField_math::diff(Dhd[_y][_x], field.next(field.prevId(Site, _x), _y), field.next(field.prevId(Site, _y), _x), 0.5);
    HField_math::diff(Dhd[_y][_z], field.next(field.prevId(Site, _z), _y), field.next(field.prevId(Site, _y), _z), 0.5);
    HField_math::diff(Dhd[_z][_t], field.next(field.prevId(Site, _t), _z), field.next(field.prevId(Site, _z), _t), 0.5);
    HField_math::diff(Dhd[_z][_x], field.next(field.prevId(Site, _x), _z), field.next(field.prevId(Site, _z), _x), 0.5);
    HField_math::diff(Dhd[_z][_y], field.next(field.prevId(Site, _y), _z), field.next(field.prevId(Site, _z), _y), 0.5);
    // Compute central sums
    HField_math::sum(Jhc[_t], field.next(Site, _t), field.prev(Site, _t), 0.5);
    HField_math::sum(Jhc[_x], field.next(Site, _x), field.prev(Site, _x), 0.5);
    HField_math::sum(Jhc[_y], field.next(Site, _y), field.prev(Site, _y), 0.5);
    HField_math::sum(Jhc[_z], field.next(Site, _z), field.prev(Site, _z), 0.5);
    // Compute diagonal sums
    HField_math::sum(Jhd[_t][_x], field.next(field.prevId(Site, _x), _t), field.next(field.prevId(Site, _t), _x), 0.5);
    HField_math::sum(Jhd[_t][_y], field.next(field.prevId(Site, _y), _t), field.next(field.prevId(Site, _t), _y), 0.5);
    HField_math::sum(Jhd[_t][_z], field.next(field.prevId(Site, _z), _t), field.next(field.prevId(Site, _t), _z), 0.5);
    HField_math::sum(Jhd[_x][_t], field.next(field.prevId(Site, _t), _x), field.next(field.prevId(Site, _x), _t), 0.5);
    HField_math::sum(Jhd[_x][_y], field.next(field.prevId(Site, _y), _x), field.next(field.prevId(Site, _x), _y), 0.5);
    HField_math::sum(Jhd[_x][_z], field.next(field.prevId(Site, _z), _x), field.next(field.prevId(Site, _x), _z), 0.5);
    HField_math::sum(Jhd[_y][_t], field.next(field.prevId(Site, _t), _y), field.next(field.prevId(Site, _y), _t), 0.5);
    HField_math::sum(Jhd[_y][_x], field.next(field.prevId(Site, _x), _y), field.next(field.prevId(Site, _y), _x), 0.5);
    HField_math::sum(Jhd[_y][_z], field.next(field.prevId(Site, _z), _y), field.next(field.prevId(Site, _y), _z), 0.5);
    HField_math::sum(Jhd[_z][_t], field.next(field.prevId(Site, _t), _z), field.next(field.prevId(Site, _z), _t), 0.5);
    HField_math::sum(Jhd[_z][_x], field.next(field.prevId(Site, _x), _z), field.next(field.prevId(Site, _z), _x), 0.5);
    HField_math::sum(Jhd[_z][_y], field.next(field.prevId(Site, _y), _z), field.next(field.prevId(Site, _z), _y), 0.5);
    // clang-format on

    // Diagonal Components
    force[MUNU_00] = 2.0 * field[Site][MUNU_11] - Jhc[_y][MUNU_11] - Jhc[_z][MUNU_11]                      //
                     + 2.0 * field[Site][MUNU_22] - Jhc[_x][MUNU_22] - Jhc[_z][MUNU_22]                    //
                     + 2.0 * field[Site][MUNU_33] - Jhc[_x][MUNU_33] - Jhc[_y][MUNU_33]                    //
                     - Dhc[_t][MUNU_01] + Dhc[_x][MUNU_01] + Dhd[_t][_x][MUNU_01]                          //
                     - Dhc[_t][MUNU_02] + Dhc[_y][MUNU_02] + Dhd[_t][_y][MUNU_02]                          //
                     - Dhc[_t][MUNU_03] + Dhc[_z][MUNU_03] + Dhd[_t][_z][MUNU_03]                          //
                     - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_y][MUNU_12] - Jhd[_x][_y][MUNU_12]   //
                     - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13]   //
                     - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

    force[MUNU_11] = 2.0 * field[Site][MUNU_00] - Jhc[_y][MUNU_00] - Jhc[_z][MUNU_00]                      //
                     + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_z][MUNU_22]                    //
                     + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_y][MUNU_33]                    //
                     + Dhc[_t][MUNU_01] - Dhc[_x][MUNU_01] - Dhd[_t][_x][MUNU_01]                          //
                     - Dhc[_x][MUNU_12] + Dhc[_y][MUNU_12] + Dhd[_x][_y][MUNU_12]                          //
                     - Dhc[_x][MUNU_13] + Dhc[_z][MUNU_13] + Dhd[_x][_z][MUNU_13]                          //
                     - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
                     - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
                     - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

    force[MUNU_22] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_z][MUNU_00]                      //
                     + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_z][MUNU_11]                    //
                     + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_x][MUNU_33]                    //
                     + Dhc[_t][MUNU_02] - Dhc[_y][MUNU_02] - Dhd[_t][_y][MUNU_02]                          //
                     + Dhc[_x][MUNU_12] - Dhc[_y][MUNU_12] - Dhd[_x][_y][MUNU_12]                          //
                     - Dhc[_y][MUNU_23] + Dhc[_z][MUNU_23] + Dhd[_y][_z][MUNU_23]                          //
                     - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
                     - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
                     - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13];  //

    force[MUNU_33] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_y][MUNU_00]                      //
                     + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_y][MUNU_11]                    //
                     + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_x][MUNU_22]                    //
                     + Dhc[_t][MUNU_03] - Dhc[_z][MUNU_03] - Dhd[_t][_z][MUNU_03]                          //
                     + Dhc[_x][MUNU_13] - Dhc[_z][MUNU_13] - Dhd[_x][_z][MUNU_13]                          //
                     + Dhc[_y][MUNU_23] - Dhc[_z][MUNU_23] - Dhd[_y][_z][MUNU_23]                          //
                     - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
                     - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
                     - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_z][MUNU_12] - Jhd[_x][_y][MUNU_12];  //

    // Time-Space Components
    // clang-format off
    force[MUNU_01] = Dhc[_t][MUNU_00] - Dhc[_x][MUNU_00] - Dhd[_t][_x][MUNU_00]                         //
                            - Dhc[_t][MUNU_11] + Dhc[_x][MUNU_11] + Dhd[_t][_x][MUNU_11]                //
                            + Jhc[_t][MUNU_22] + Jhc[_x][MUNU_22] - Jhd[_t][_x][MUNU_22]                //
                            + Jhc[_t][MUNU_33] + Jhc[_x][MUNU_33] - Jhd[_t][_x][MUNU_33]                //
                            - 4.0 * field[Site][MUNU_01] - field[Site][MUNU_22] - field[Site][MUNU_33]  //
                            + 2.0 * (Jhc[_y][MUNU_01] + Jhc[_z][MUNU_01])                               //
                            + field[Site][MUNU_02] - field.next(Site, _x)[MUNU_02] - field.prev(Site, _y)[MUNU_02] + field.next(field.prevId(Site, _y), _x)[MUNU_02]  //
                            + field[Site][MUNU_03] - field.next(Site, _x)[MUNU_03] - field.prev(Site, _z)[MUNU_03] + field.next(field.prevId(Site, _z), _x)[MUNU_03]  //
                            + field[Site][MUNU_12] - field.next(Site, _t)[MUNU_12] - field.prev(Site, _y)[MUNU_12] + field.next(field.prevId(Site, _y), _t)[MUNU_12]  //
                            + field[Site][MUNU_13] - field.next(Site, _t)[MUNU_13] - field.prev(Site, _z)[MUNU_13] + field.next(field.prevId(Site, _z), _t)[MUNU_13];

    force[MUNU_02] = Dhc[_t][MUNU_00] - Dhc[_y][MUNU_00] - Dhd[_t][_y][MUNU_00]                         //
                            - Dhc[_t][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_t][_y][MUNU_22]                //
                            + Jhc[_t][MUNU_11] + Jhc[_y][MUNU_11] - Jhd[_t][_y][MUNU_11]                //
                            + Jhc[_t][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_t][_y][MUNU_33]                //
                            - 4.0 * field[Site][MUNU_02] - field[Site][MUNU_11] - field[Site][MUNU_33]  //
                            + 2.0 * (Jhc[_x][MUNU_02] + Jhc[_z][MUNU_02])                               //
                            + field[Site][MUNU_01] - field.next(Site, _y)[MUNU_01] - field.prev(Site, _x)[MUNU_01] + field.next(field.prevId(Site, _x), _y)[MUNU_01]  //
                            + field[Site][MUNU_03] - field.next(Site, _y)[MUNU_03] - field.prev(Site, _z)[MUNU_03] + field.next(field.prevId(Site, _z), _y)[MUNU_03]  //
                            + field[Site][MUNU_12] - field.next(Site, _t)[MUNU_12] - field.prev(Site, _x)[MUNU_12] + field.next(field.prevId(Site, _x), _t)[MUNU_12]  //
                            + field[Site][MUNU_23] - field.next(Site, _t)[MUNU_23] - field.prev(Site, _z)[MUNU_23] + field.next(field.prevId(Site, _z), _t)[MUNU_23];

    force[MUNU_03] = Dhc[_t][MUNU_00] - Dhc[_z][MUNU_00] - Dhd[_t][_z][MUNU_00]                         //
                            - Dhc[_t][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_t][_z][MUNU_33]                //
                            + Jhc[_t][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_t][_z][MUNU_11]                //
                            + Jhc[_t][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_t][_z][MUNU_22]                //
                            - 4.0 * field[Site][MUNU_03] - field[Site][MUNU_11] - field[Site][MUNU_22]  //
                            + 2.0 * (Jhc[_x][MUNU_03] + Jhc[_y][MUNU_03])                               //
                            + field[Site][MUNU_01] - field.next(Site, _z)[MUNU_01] - field.prev(Site, _x)[MUNU_01] + field.next(field.prevId(Site, _x), _z)[MUNU_01]  //
                            + field[Site][MUNU_02] - field.next(Site, _z)[MUNU_02] - field.prev(Site, _y)[MUNU_02] + field.next(field.prevId(Site, _y), _z)[MUNU_02]  //
                            + field[Site][MUNU_13] - field.next(Site, _t)[MUNU_13] - field.prev(Site, _y)[MUNU_13] + field.next(field.prevId(Site, _y), _t)[MUNU_13]  //
                            + field[Site][MUNU_23] - field.next(Site, _t)[MUNU_23] - field.prev(Site, _y)[MUNU_23] + field.next(field.prevId(Site, _y), _t)[MUNU_23];

    // Space-Space Components
    force[MUNU_12] = Dhc[_x][MUNU_11] - Dhc[_y][MUNU_11] - Dhd[_x][_y][MUNU_11]                         //
                            - Dhc[_x][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_x][_y][MUNU_22]                //
                            + Jhc[_x][MUNU_00] + Jhc[_y][MUNU_00] - Jhd[_x][_y][MUNU_00]                //
                            + Jhc[_x][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_x][_y][MUNU_33]                //
                            - 4.0 * field[Site][MUNU_12] - field[Site][MUNU_00] - field[Site][MUNU_33]  //
                            + 2.0 * (Jhc[_t][MUNU_12] + Jhc[_z][MUNU_12])                               //
                            + field[Site][MUNU_01] - field.next(Site, _y)[MUNU_01] - field.prev(Site, _t)[MUNU_01] + field.next(field.prevId(Site, _t), _y)[MUNU_01]  //
                            + field[Site][MUNU_02] - field.next(Site, _x)[MUNU_02] - field.prev(Site, _t)[MUNU_02] + field.next(field.prevId(Site, _t), _x)[MUNU_02]  //
                            + field[Site][MUNU_13] - field.next(Site, _y)[MUNU_13] - field.prev(Site, _z)[MUNU_13] + field.next(field.prevId(Site, _z), _y)[MUNU_13]  //
                            + field[Site][MUNU_23] - field.next(Site, _x)[MUNU_23] - field.prev(Site, _z)[MUNU_23] + field.next(field.prevId(Site, _z), _x)[MUNU_23];

    force[MUNU_13] = Dhc[_x][MUNU_11] - Dhc[_z][MUNU_11] - Dhd[_x][_z][MUNU_11]                  //
                            - Dhc[_x][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_x][_z][MUNU_33]                //
                            + Jhc[_x][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_x][_z][MUNU_00]                //
                            + Jhc[_x][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_x][_z][MUNU_22]                //
                            - 4.0 * field[Site][MUNU_13] - field[Site][MUNU_00] - field[Site][MUNU_22]  //
                            + 2.0 * (Jhc[_t][MUNU_13] + Jhc[_y][MUNU_13])                               //
                            + field[Site][MUNU_01] - field.next(Site, _z)[MUNU_01] - field.prev(Site, _t)[MUNU_01] + field.next(field.prevId(Site, _t), _z)[MUNU_01]  //
                            + field[Site][MUNU_03] - field.next(Site, _x)[MUNU_03] - field.prev(Site, _t)[MUNU_03] + field.next(field.prevId(Site, _t), _x)[MUNU_03]  //
                            + field[Site][MUNU_12] - field.next(Site, _z)[MUNU_12] - field.prev(Site, _y)[MUNU_12] + field.next(field.prevId(Site, _y), _z)[MUNU_12]  //
                            + field[Site][MUNU_23] - field.next(Site, _x)[MUNU_23] - field.prev(Site, _y)[MUNU_23] + field.next(field.prevId(Site, _y), _x)[MUNU_23];

    force[MUNU_23] = Dhc[_y][MUNU_22] - Dhc[_z][MUNU_22] - Dhd[_y][_z][MUNU_22]                  //
                            - Dhc[_y][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_y][_z][MUNU_33]                //
                            + Jhc[_y][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_y][_z][MUNU_00]                //
                            + Jhc[_y][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_y][_z][MUNU_11]                //
                            - 4.0 * field[Site][MUNU_23] - field[Site][MUNU_00] - field[Site][MUNU_11]  //
                            + 2.0 * (Jhc[_t][MUNU_23] + Jhc[_x][MUNU_23])                               //
                            + field[Site][MUNU_02] - field.next(Site, _z)[MUNU_02] - field.prev(Site, _t)[MUNU_02] + field.next(field.prevId(Site, _t), _z)[MUNU_02]  //
                            + field[Site][MUNU_03] - field.next(Site, _y)[MUNU_03] - field.prev(Site, _t)[MUNU_03] + field.next(field.prevId(Site, _t), _y)[MUNU_03]  //
                            + field[Site][MUNU_12] - field.next(Site, _z)[MUNU_12] - field.prev(Site, _x)[MUNU_12] + field.next(field.prevId(Site, _x), _z)[MUNU_12]  //
                            + field[Site][MUNU_13] - field.next(Site, _y)[MUNU_13] - field.prev(Site, _x)[MUNU_13] + field.next(field.prevId(Site, _x), _y)[MUNU_13];
    // clang-format on

    // force *= kappa;
}
}  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  montecarlo::Metropolis::updateField() Specialization
--------------------------------------------------------------------------------------------------*/
#include "reticolo/montecarlo/Metropolis.hpp"

template <>
void reticolo::montecarlo::Metropolis<reticolo::action::WeakFieldEuclideanGR>::updateField() {
    uint       Acc = 0;       // acceptance
    ActionType SVarTot(0.0);  // cumulative action variation

    for (int Site = 0; Site < _Field.getNsites(); Site++) {
        // Generate a randomized local field variation
        FieldType FieldVar;  // local field variation
        FieldType FieldOld = _Field[Site];
        RealD     Scale = _ProposalWidth * _Action.lP_fm / _Action.aa;
        randomize(FieldVar, Scale, _UnifC, _Rng);
        HField_math::sum(_Field[Site], _Field[Site], FieldVar);

        // Compute the updated Lagrangian in the surrounding sites
        std::vector<ActionType> LGRPost;             // Vector storing the new curvature values in the check sites
        ActionType              ActionChange = 0.0;  // Cumulative curvature variation
        for (int& CheckSite : _Action._Checks[Site]) {
            RealD Tmp = _Action.compute_LGR_loc(_Field, CheckSite);
            // std::cout << Tmp << " ";
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - _Action._LGR[CheckSite];
            }
        }
        // Metropolis acceptance + positive action
        if (LGRPost.size() == 5 && exp(-ActionChange) > _Unif(_Rng)) {
            Acc++;
            SVarTot += ActionChange;
            for (uint CheckSite = 0; CheckSite < 5; CheckSite++) {
                _Action._LGR[_Action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
            }
        } else {
            _Field[Site] = FieldOld;
        }
    }
    _McStats.update(static_cast<double>(Acc) / _Field.getNsites(), SVarTot);
}

/*--------------------------------------------------------------------------------------------------
  montecarlo::HMC::updateField() Specialization
--------------------------------------------------------------------------------------------------*/
#include "reticolo/montecarlo/HMC.hpp"

template <>
void reticolo::montecarlo::HMC<reticolo::action::WeakFieldEuclideanGR>::updateField() {
    int        NSites = _Field.getNsites();
    int        Acc = 0;
    ActionType SvarTot(0.0);
    FieldType  OldField;
    FieldType  Mom;
    FieldType  Force;

    for (int Site = 0; Site < NSites; Site++) {
        // save the old field configuration;
        OldField = _Field[Site];
        // Compute start Hamiltonian
        randomize(Mom, 1.0, _Norm, _Rng);
        RealD OldK = 0.5 * Mom.dot();
        // Compute Forces
        _Action.compute_Force_loc(_Field, Force, Site);
        // Momenta half step
        Mom += 0.5 * _Stepsize * Force;
        // Leapfrog algorithm
        for (uint Step = 0; Step < _Steps; Step++) {
            // Update field
            _Field[Site] += _Stepsize * Mom;
            // Compute new forces
            _Action.compute_Force_loc(_Field, Force, Site);
            // Update momenta
            Mom -= _Stepsize * Force;
        }
        // Half step momenta roll-back
        Mom += 0.5 * _Stepsize * Force;
        // Compute end Hamiltonian
        RealD NewK = 0.5 * Mom.dot();
        // Compute the updated Lagrangian in the surrounding sites
        std::vector<ActionType> LGRPost;             // Vector storing the new curvature values in the check sites
        ActionType              ActionChange = 0.0;  // Cumulative curvature variation
        for (int& CheckSite : _Action._Checks[Site]) {
            RealD Tmp = _Action.compute_LGR_loc(_Field, CheckSite) / _Action.kappa;
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - _Action._LGR[CheckSite];
            } else {
                break;
            }
        }
        // Metropolis acceptance + positive action
        if (LGRPost.size() == 5 && exp(-(ActionChange + NewK - OldK)) > _Unif(_Rng)) {
            Acc++;
            SvarTot += ActionChange;
            for (uint CheckSite = 0; CheckSite < 5; CheckSite++) {
                _Action._LGR[_Action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
            }
        } else {
            _Field[Site] = OldField;
        }
    }
    _McStats.update(static_cast<double>(Acc) / NSites, SvarTot);
}
