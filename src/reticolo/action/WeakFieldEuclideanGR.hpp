/*******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include <array>
#include <cmath>
#include <concepts>
#include <format>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reticolo/action/actionBase.hpp"
#include "reticolo/core/physics/constants.hpp"
#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep
#include "reticolo/core/types/coord.hpp"
#include "reticolo/core/types/hfield.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "yaml-cpp/node/node.h"

namespace pc = reticolo::PhysicalConstants;

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  WeakFieldEuclideanGR Class Declaration
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
class WeakFieldEuclideanGR : public ActionBase<HField<TImpl>, TImpl, TImpl> {
  public:
    /* Types and public action metadata */
    using base_type = ActionBase<HField<TImpl>, TImpl, TImpl>;
    using action_type = base_type::action_type;
    using field_type = base_type::field_type;
    using size_type = base_type::size_type;
    using impl_type = TImpl;
    static constexpr int Dims = 4;     // Dimensions of the action
    static constexpr int Stencil = 2;  // Minimum step size for multi-thread safety

    /* Algorithm capabilities */
    static constexpr bool IsMetropolisCapable = true;
    static constexpr bool IsHmcCapable = false;
    static constexpr bool IsLLRCapable = false;

    /* Action parameters */
    struct Params {
        impl_type beta;
        Params() : beta(5.7) {};
        Params(impl_type beta) : beta(beta) {};
    } p;

    /* Observables */
    struct Observables {
        impl_type R;
        auto      operator+=(const Observables& rhs) -> Observables {
            R += rhs.R;
            return *this;
        };
        auto operator/=(const impl_type& rhs) -> Observables {
            R /= rhs;
            return *this;
        };
    };

    /* Constructors */
    WeakFieldEuclideanGR() = default;

    /* setup */
    void setup(const YAML::Node& ActionParams) override;

    /* Sync with lattice */
    void lattice_sync(Lattice<field_type>& field) override;

    /* Gloabal and local action computations */
    auto compute_S(Lattice<field_type>& field) -> action_type override;
    auto compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type override;
    auto compute_dS_loc(Lattice<field_type>& field, const field_type& dphi, size_type site) -> action_type override;

    /* HMC methods */
    void compute_Forces(Lattice<field_type>& field, Lattice<field_type>& Forces) override {};

    void compute_LLRForces(Lattice<field_type>& /*unused*/, Lattice<field_type>& /*unused*/, TImpl /*unused*/,
                           TImpl /*unused*/, TImpl /*unused*/) override {};

    /* Perform the measurements or returns updated Observable values*/
    auto Measure(const Lattice<field_type>& field) -> Observables;

    /* Log stuff*/
    auto GetName() -> std::string override {
        std::string Res = "Weak Field Euclidean General Relativity";
        Res += std::same_as<impl_type, RealF> ? " [float]" : " [double]";
        return Res;
    };
    auto GetParameters() -> std::string override {
        std::string ParamStr = std::format("[ beta : {:4.1f} ]", p.beta);
        return ParamStr;
    }

    /*--------------------------------------------------------------------------
        Custom variables and methods for WeakFieldEuclideanGR Class
    --------------------------------------------------------------------------*/

    /* Physical parameters */
    const impl_type _HbarcGeVfm = static_cast<impl_type>(pc::hbar * pc::c / pc::e * 1e15 * 1e-9);  // ~0.197 GeV fm
    const impl_type _GnFm2 = static_cast<impl_type>(6.70883e-39 * pow(_HbarcGeVfm, 2));            // ~2.6e-40 fm^2
    const impl_type _LPFm = static_cast<impl_type>(sqrt(_GnFm2));                                  // ~1.6e-20 fm
    const impl_type _KappaGeV2 =
        static_cast<impl_type>(pow(_HbarcGeVfm, 2) / (16.0 * std::numbers::pi * _GnFm2));  // ~3e-36 GeV^2
    impl_type _AA;
    impl_type _AInvGeV;
    impl_type _Kappa;

    /* Boundary conditions */
    std::string _BC;
    field_type  _BVal;
    void        make_updatelist(Lattice<field_type>& field);

    /* Vector storing the current values of the curvature for each lattice point */
    std::vector<action_type>              _LGR;
    std::vector<std::array<size_type, 5>> _Checks;
    std::vector<size_type>                _ToUpdate;

    /* compute the check indexes */
    void make_checks(const Lattice<field_type>& field);

    /* Compute the curvature at a specific lattice point */
    void               update_LGR(const Lattice<field_type>& field);
    [[nodiscard]] auto compute_LGR_loc(const Lattice<field_type>& field, size_type site) const -> action_type;
    [[nodiscard]] auto check_pos_R(size_type site, std::vector<action_type> RPost, action_type DR = 0.0) -> bool;
    // static void        compute_Force_loc(const Lattice<field_type>& field, field_type& force, int site);

};  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
inline void WeakFieldEuclideanGR<TImpl>::setup(const YAML::Node& ActionParams) {
    p.beta = ActionParams["beta"].as<TImpl>();

    if (ActionParams["bc"]) {
        _BC = ActionParams["bc"].as<std::string>();
    } else {
        _BC = "periodic";
    }
}

template <RealValue TImpl>
inline void WeakFieldEuclideanGR<TImpl>::lattice_sync(Lattice<field_type>& field) {
    /* Initialize parameters values here */
    _AA = 0.5 *
          exp(-1.6804 - (1.7331 * (p.beta - 6.0)) + (0.7849 * pow(p.beta - 6.0, 2)) - (0.4428 * pow(p.beta - 6.0, 3)));
    _AInvGeV = _AA / _HbarcGeVfm;
    _Kappa = _KappaGeV2 * _AInvGeV * _AInvGeV;

    /* Resize and clears the curvature lattice to match the lattice sizes */
    _LGR.resize(field.getNsites());

    /* Build the check indexing vector */
    make_checks(field);

    /* Build the update list*/
    make_updatelist(field);

    /* Updates the curvature values */
    update_LGR(field);
}

template <RealValue TImpl>
inline auto WeakFieldEuclideanGR<TImpl>::compute_S(Lattice<field_type>& field) -> action_type {
    // Update the lattice
    update_LGR(field);
    // Accumulate the Lagrangian
    action_type STot = std::reduce(_LGR.begin(), _LGR.end());

    return STot;
}

template <RealValue TImpl>
inline auto WeakFieldEuclideanGR<TImpl>::compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type {
    return 0.0;
};

template <RealValue TImpl>
inline auto WeakFieldEuclideanGR<TImpl>::compute_dS_loc(Lattice<field_type>& field, const field_type& dphi,
                                                        size_type site) -> action_type {
    return 0.0;
};

// inline void WeakFieldEuclideanGR::compute_Forces(const Lattice<field_type>& field, Lattice<field_type>& forces) {
//     std::array<field_type, 4>                Dhc;  // Central derivatives
//     std::array<std::array<field_type, 4>, 4> Dhd;  // Diagonal derivatives
//     std::array<field_type, 4>                Jhc;  // Central sum
//     std::array<std::array<field_type, 4>, 4> Jhd;  // Diagonal Sum

//     for (int Site = 0; Site < field.getNsites(); Site++) {
//         // clang-format off
//         // Compute central derivatives
//         HField_math::diff(Dhc[_t], field.n(Site, _t), field.p(Site, _t), 0.5);
//         HField_math::diff(Dhc[_x], field.n(Site, _x), field.p(Site, _x), 0.5);
//         HField_math::diff(Dhc[_y], field.n(Site, _y), field.p(Site, _y), 0.5);
//         HField_math::diff(Dhc[_z], field.n(Site, _z), field.p(Site, _z), 0.5);
//         // Compute diagonal derivatives
//         HField_math::diff(Dhd[_t][_x], field.n(field.pI(Site, _x), _t), field.n(field.pI(Site, _t), _x), 0.5);
//         HField_math::diff(Dhd[_t][_y], field.n(field.pI(Site, _y), _t), field.n(field.pI(Site, _t), _y), 0.5);
//         HField_math::diff(Dhd[_t][_z], field.n(field.pI(Site, _z), _t), field.n(field.pI(Site, _t), _z), 0.5);
//         HField_math::diff(Dhd[_x][_t], field.n(field.pI(Site, _t), _x), field.n(field.pI(Site, _x), _t), 0.5);
//         HField_math::diff(Dhd[_x][_y], field.n(field.pI(Site, _y), _x), field.n(field.pI(Site, _x), _y), 0.5);
//         HField_math::diff(Dhd[_x][_z], field.n(field.pI(Site, _z), _x), field.n(field.pI(Site, _x), _z), 0.5);
//         HField_math::diff(Dhd[_y][_t], field.n(field.pI(Site, _t), _y), field.n(field.pI(Site, _y), _t), 0.5);
//         HField_math::diff(Dhd[_y][_x], field.n(field.pI(Site, _x), _y), field.n(field.pI(Site, _y), _x), 0.5);
//         HField_math::diff(Dhd[_y][_z], field.n(field.pI(Site, _z), _y), field.n(field.pI(Site, _y), _z), 0.5);
//         HField_math::diff(Dhd[_z][_t], field.n(field.pI(Site, _t), _z), field.n(field.pI(Site, _z), _t), 0.5);
//         HField_math::diff(Dhd[_z][_x], field.n(field.pI(Site, _x), _z), field.n(field.pI(Site, _z), _x), 0.5);
//         HField_math::diff(Dhd[_z][_y], field.n(field.pI(Site, _y), _z), field.n(field.pI(Site, _z), _y), 0.5);
//         // Compute central sums
//         HField_math::sum(Jhc[_t], field.n(Site, _t), field.p(Site, _t), 0.5);
//         HField_math::sum(Jhc[_x], field.n(Site, _x), field.p(Site, _x), 0.5);
//         HField_math::sum(Jhc[_y], field.n(Site, _y), field.p(Site, _y), 0.5);
//         HField_math::sum(Jhc[_z], field.n(Site, _z), field.p(Site, _z), 0.5);
//         // Compute diagonal sums
//         HField_math::sum(Jhd[_t][_x], field.n(field.pI(Site, _x), _t), field.n(field.pI(Site, _t), _x), 0.5);
//         HField_math::sum(Jhd[_t][_y], field.n(field.pI(Site, _y), _t), field.n(field.pI(Site, _t), _y), 0.5);
//         HField_math::sum(Jhd[_t][_z], field.n(field.pI(Site, _z), _t), field.n(field.pI(Site, _t), _z), 0.5);
//         HField_math::sum(Jhd[_x][_t], field.n(field.pI(Site, _t), _x), field.n(field.pI(Site, _x), _t), 0.5);
//         HField_math::sum(Jhd[_x][_y], field.n(field.pI(Site, _y), _x), field.n(field.pI(Site, _x), _y), 0.5);
//         HField_math::sum(Jhd[_x][_z], field.n(field.pI(Site, _z), _x), field.n(field.pI(Site, _x), _z), 0.5);
//         HField_math::sum(Jhd[_y][_t], field.n(field.pI(Site, _t), _y), field.n(field.pI(Site, _y), _t), 0.5);
//         HField_math::sum(Jhd[_y][_x], field.n(field.pI(Site, _x), _y), field.n(field.pI(Site, _y), _x), 0.5);
//         HField_math::sum(Jhd[_y][_z], field.n(field.pI(Site, _z), _y), field.n(field.pI(Site, _y), _z), 0.5);
//         HField_math::sum(Jhd[_z][_t], field.n(field.pI(Site, _t), _z), field.n(field.pI(Site, _z), _t), 0.5);
//         HField_math::sum(Jhd[_z][_x], field.n(field.pI(Site, _x), _z), field.n(field.pI(Site, _z), _x), 0.5);
//         HField_math::sum(Jhd[_z][_y], field.n(field.pI(Site, _y), _z), field.n(field.pI(Site, _z), _y), 0.5);
//         // clang-format on

//         // Diagonal Components
//         forces[Site][MUNU_00] = 2.0 * field[Site][MUNU_11] - Jhc[_y][MUNU_11] - Jhc[_z][MUNU_11] //
//                                 + 2.0 * field[Site][MUNU_22] - Jhc[_x][MUNU_22] - Jhc[_z][MUNU_22] //
//                                 + 2.0 * field[Site][MUNU_33] - Jhc[_x][MUNU_33] - Jhc[_y][MUNU_33] //
//                                 - Dhc[_t][MUNU_01] + Dhc[_x][MUNU_01] + Dhd[_t][_x][MUNU_01] //
//                                 - Dhc[_t][MUNU_02] + Dhc[_y][MUNU_02] + Dhd[_t][_y][MUNU_02] //
//                                 - Dhc[_t][MUNU_03] + Dhc[_z][MUNU_03] + Dhd[_t][_z][MUNU_03] //
//                                 - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_y][MUNU_12] -
//                                 Jhd[_x][_y][MUNU_12]
//                                 - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] -
//                                 Jhd[_x][_z][MUNU_13]
//                                 - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] -
//                                 Jhd[_y][_z][MUNU_23];

//         forces[Site][MUNU_11] = 2.0 * field[Site][MUNU_00] - Jhc[_y][MUNU_00] - Jhc[_z][MUNU_00] //
//                                 + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_z][MUNU_22] //
//                                 + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_y][MUNU_33] //
//                                 + Dhc[_t][MUNU_01] - Dhc[_x][MUNU_01] - Dhd[_t][_x][MUNU_01] //
//                                 - Dhc[_x][MUNU_12] + Dhc[_y][MUNU_12] + Dhd[_x][_y][MUNU_12] //
//                                 - Dhc[_x][MUNU_13] + Dhc[_z][MUNU_13] + Dhd[_x][_z][MUNU_13] //
//                                 - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] -
//                                 Jhd[_t][_y][MUNU_02]
//                                 - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] -
//                                 Jhd[_t][_z][MUNU_03]
//                                 - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] -
//                                 Jhd[_y][_z][MUNU_23];

//         forces[Site][MUNU_22] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_z][MUNU_00] //
//                                 + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_z][MUNU_11] //
//                                 + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_x][MUNU_33] //
//                                 + Dhc[_t][MUNU_02] - Dhc[_y][MUNU_02] - Dhd[_t][_y][MUNU_02] //
//                                 + Dhc[_x][MUNU_12] - Dhc[_y][MUNU_12] - Dhd[_x][_y][MUNU_12] //
//                                 - Dhc[_y][MUNU_23] + Dhc[_z][MUNU_23] + Dhd[_y][_z][MUNU_23] //
//                                 - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] -
//                                 Jhd[_t][_x][MUNU_01]
//                                 - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] -
//                                 Jhd[_t][_z][MUNU_03]
//                                 - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] -
//                                 Jhd[_x][_z][MUNU_13];

//         forces[Site][MUNU_33] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_y][MUNU_00] //
//                                 + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_y][MUNU_11] //
//                                 + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_x][MUNU_22] //
//                                 + Dhc[_t][MUNU_03] - Dhc[_z][MUNU_03] - Dhd[_t][_z][MUNU_03] //
//                                 + Dhc[_x][MUNU_13] - Dhc[_z][MUNU_13] - Dhd[_x][_z][MUNU_13] //
//                                 + Dhc[_y][MUNU_23] - Dhc[_z][MUNU_23] - Dhd[_y][_z][MUNU_23] //
//                                 - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] -
//                                 Jhd[_t][_x][MUNU_01]
//                                 - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] -
//                                 Jhd[_t][_y][MUNU_02]
//                                 - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_z][MUNU_12] -
//                                 Jhd[_x][_y][MUNU_12];

//         // Time-Space Components
//         // clang-format off
//         forces[Site][MUNU_01] = Dhc[_t][MUNU_00] - Dhc[_x][MUNU_00] - Dhd[_t][_x][MUNU_00]                  //
//                                 - Dhc[_t][MUNU_11] + Dhc[_x][MUNU_11] + Dhd[_t][_x][MUNU_11]                //
//                                 + Jhc[_t][MUNU_22] + Jhc[_x][MUNU_22] - Jhd[_t][_x][MUNU_22]                //
//                                 + Jhc[_t][MUNU_33] + Jhc[_x][MUNU_33] - Jhd[_t][_x][MUNU_33]                //
//                                 - 4.0 * field[Site][MUNU_01] - field[Site][MUNU_22] - field[Site][MUNU_33]  //
//                                 + 2.0 * (Jhc[_y][MUNU_01] + Jhc[_z][MUNU_01])                               //
//                                 + field[Site][MUNU_02] - field.n(Site, _x)[MUNU_02] - field.p(Site, _y)[MUNU_02]
//                                 + field.n(field.pI(Site, _y), _x)[MUNU_02]  //
//                                 + field[Site][MUNU_03] - field.n(Site, _x)[MUNU_03] - field.p(Site, _z)[MUNU_03]
//                                 + field.n(field.pI(Site, _z), _x)[MUNU_03]  //
//                                 + field[Site][MUNU_12] - field.n(Site, _t)[MUNU_12] - field.p(Site, _y)[MUNU_12]
//                                 + field.n(field.pI(Site, _y), _t)[MUNU_12]  //
//                                 + field[Site][MUNU_13] - field.n(Site, _t)[MUNU_13] - field.p(Site, _z)[MUNU_13]
//                                 + field.n(field.pI(Site, _z), _t)[MUNU_13];

//         forces[Site][MUNU_02] = Dhc[_t][MUNU_00] - Dhc[_y][MUNU_00] - Dhd[_t][_y][MUNU_00]                  //
//                                 - Dhc[_t][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_t][_y][MUNU_22]                //
//                                 + Jhc[_t][MUNU_11] + Jhc[_y][MUNU_11] - Jhd[_t][_y][MUNU_11]                //
//                                 + Jhc[_t][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_t][_y][MUNU_33]                //
//                                 - 4.0 * field[Site][MUNU_02] - field[Site][MUNU_11] - field[Site][MUNU_33]  //
//                                 + 2.0 * (Jhc[_x][MUNU_02] + Jhc[_z][MUNU_02])                               //
//                                 + field[Site][MUNU_01] - field.n(Site, _y)[MUNU_01] - field.p(Site, _x)[MUNU_01]
//                                 + field.n(field.pI(Site, _x), _y)[MUNU_01]  //
//                                 + field[Site][MUNU_03] - field.n(Site, _y)[MUNU_03] - field.p(Site, _z)[MUNU_03]
//                                 + field.n(field.pI(Site, _z), _y)[MUNU_03]  //
//                                 + field[Site][MUNU_12] - field.n(Site, _t)[MUNU_12] - field.p(Site, _x)[MUNU_12]
//                                 + field.n(field.pI(Site, _x), _t)[MUNU_12]  //
//                                 + field[Site][MUNU_23] - field.n(Site, _t)[MUNU_23] - field.p(Site, _z)[MUNU_23]
//                                 + field.n(field.pI(Site, _z), _t)[MUNU_23];

//         forces[Site][MUNU_03] = Dhc[_t][MUNU_00] - Dhc[_z][MUNU_00] - Dhd[_t][_z][MUNU_00]                  //
//                                 - Dhc[_t][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_t][_z][MUNU_33]                //
//                                 + Jhc[_t][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_t][_z][MUNU_11]                //
//                                 + Jhc[_t][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_t][_z][MUNU_22]                //
//                                 - 4.0 * field[Site][MUNU_03] - field[Site][MUNU_11] - field[Site][MUNU_22]  //
//                                 + 2.0 * (Jhc[_x][MUNU_03] + Jhc[_y][MUNU_03])                               //
//                                 + field[Site][MUNU_01] - field.n(Site, _z)[MUNU_01] - field.p(Site, _x)[MUNU_01]
//                                 + field.n(field.pI(Site, _x), _z)[MUNU_01]  //
//                                 + field[Site][MUNU_02] - field.n(Site, _z)[MUNU_02] - field.p(Site, _y)[MUNU_02]
//                                 + field.n(field.pI(Site, _y), _z)[MUNU_02]  //
//                                 + field[Site][MUNU_13] - field.n(Site, _t)[MUNU_13] - field.p(Site, _y)[MUNU_13]
//                                 + field.n(field.pI(Site, _y), _t)[MUNU_13]  //
//                                 + field[Site][MUNU_23] - field.n(Site, _t)[MUNU_23] - field.p(Site, _y)[MUNU_23]
//                                 + field.n(field.pI(Site, _y), _t)[MUNU_23];

//         // Space-Space Components
//         forces[Site][MUNU_12] = Dhc[_x][MUNU_11] - Dhc[_y][MUNU_11] - Dhd[_x][_y][MUNU_11]                  //
//                                 - Dhc[_x][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_x][_y][MUNU_22]                //
//                                 + Jhc[_x][MUNU_00] + Jhc[_y][MUNU_00] - Jhd[_x][_y][MUNU_00]                //
//                                 + Jhc[_x][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_x][_y][MUNU_33]                //
//                                 - 4.0 * field[Site][MUNU_12] - field[Site][MUNU_00] - field[Site][MUNU_33]  //
//                                 + 2.0 * (Jhc[_t][MUNU_12] + Jhc[_z][MUNU_12])                               //
//                                 + field[Site][MUNU_01] - field.n(Site, _y)[MUNU_01] - field.p(Site, _t)[MUNU_01]
//                                 + field.n(field.pI(Site, _t), _y)[MUNU_01]  //
//                                 + field[Site][MUNU_02] - field.n(Site, _x)[MUNU_02] - field.p(Site, _t)[MUNU_02]
//                                 + field.n(field.pI(Site, _t), _x)[MUNU_02]  //
//                                 + field[Site][MUNU_13] - field.n(Site, _y)[MUNU_13] - field.p(Site, _z)[MUNU_13]
//                                 + field.n(field.pI(Site, _z), _y)[MUNU_13]  //
//                                 + field[Site][MUNU_23] - field.n(Site, _x)[MUNU_23] - field.p(Site, _z)[MUNU_23]
//                                 + field.n(field.pI(Site, _z), _x)[MUNU_23];

//         forces[Site][MUNU_13] = Dhc[_x][MUNU_11] - Dhc[_z][MUNU_11] - Dhd[_x][_z][MUNU_11]                  //
//                                 - Dhc[_x][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_x][_z][MUNU_33]                //
//                                 + Jhc[_x][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_x][_z][MUNU_00]                //
//                                 + Jhc[_x][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_x][_z][MUNU_22]                //
//                                 - 4.0 * field[Site][MUNU_13] - field[Site][MUNU_00] - field[Site][MUNU_22]  //
//                                 + 2.0 * (Jhc[_t][MUNU_13] + Jhc[_y][MUNU_13])                               //
//                                 + field[Site][MUNU_01] - field.n(Site, _z)[MUNU_01] - field.p(Site, _t)[MUNU_01]
//                                 + field.n(field.pI(Site, _t), _z)[MUNU_01]  //
//                                 + field[Site][MUNU_03] - field.n(Site, _x)[MUNU_03] - field.p(Site, _t)[MUNU_03]
//                                 + field.n(field.pI(Site, _t), _x)[MUNU_03]  //
//                                 + field[Site][MUNU_12] - field.n(Site, _z)[MUNU_12] - field.p(Site, _y)[MUNU_12]
//                                 + field.n(field.pI(Site, _y), _z)[MUNU_12]  //
//                                 + field[Site][MUNU_23] - field.n(Site, _x)[MUNU_23] - field.p(Site, _y)[MUNU_23]
//                                 + field.n(field.pI(Site, _y), _x)[MUNU_23];

//         forces[Site][MUNU_23] = Dhc[_y][MUNU_22] - Dhc[_z][MUNU_22] - Dhd[_y][_z][MUNU_22]                  //
//                                 - Dhc[_y][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_y][_z][MUNU_33]                //
//                                 + Jhc[_y][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_y][_z][MUNU_00]                //
//                                 + Jhc[_y][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_y][_z][MUNU_11]                //
//                                 - 4.0 * field[Site][MUNU_23] - field[Site][MUNU_00] - field[Site][MUNU_11]  //
//                                 + 2.0 * (Jhc[_t][MUNU_23] + Jhc[_x][MUNU_23])                               //
//                                 + field[Site][MUNU_02] - field.n(Site, _z)[MUNU_02] - field.p(Site, _t)[MUNU_02]
//                                 + field.n(field.pI(Site, _t), _z)[MUNU_02]  //
//                                 + field[Site][MUNU_03] - field.n(Site, _y)[MUNU_03] - field.p(Site, _t)[MUNU_03]
//                                 + field.n(field.pI(Site, _t), _y)[MUNU_03]  //
//                                 + field[Site][MUNU_12] - field.n(Site, _z)[MUNU_12] - field.p(Site, _x)[MUNU_12]
//                                 + field.n(field.pI(Site, _x), _z)[MUNU_12]  //
//                                 + field[Site][MUNU_13] - field.n(Site, _y)[MUNU_13] - field.p(Site, _x)[MUNU_13]
//                                 + field.n(field.pI(Site, _x), _y)[MUNU_13];
//         // clang-format on

//         forces[Site] *= kappa;
//     }
// }

template <RealValue TImpl>
inline auto WeakFieldEuclideanGR<TImpl>::Measure(const Lattice<field_type>& field) -> Observables {
    return {std::reduce(_LGR.begin(), _LGR.end())};
}

/*--------------------------------------------------------------------------------------------------
      Custom variables and methods for WeakFieldEuclideanGR Class implementation
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
inline void WeakFieldEuclideanGR<TImpl>::make_checks(const Lattice<field_type>& field) {
    _Checks.clear();
    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        _Checks.push_back({
            Site,                         //
            field.Idx->prevId(Site, _t),  //
            field.Idx->prevId(Site, _x),  //
            field.Idx->prevId(Site, _y),  //
            field.Idx->prevId(Site, _z),  //
        });
    }
}

template <RealValue TImpl>
inline void WeakFieldEuclideanGR<TImpl>::make_updatelist(Lattice<field_type>& field) {
    _ToUpdate.clear();
    _ToUpdate.reserve(field.getNsites());

    /* Loop throught the lattice to generate all the closest neighbours */
    if (_BC == "periodic") {
        for (size_type Site = 0; Site < field.getNsites(); Site++) {
            _ToUpdate.push_back(Site);
        }
    } else if (_BC == "open") {
        auto                   Sizes = field.getSizes();
        std::vector<size_type> Coord(Sizes.size(), 0);
        for (size_type Site = 0; Site < field.getNsites(); Site++) {
            for (size_type i = 1; i < Sizes.size(); i++) {
                if (Coord[i] < 2 || Coord[i] > Sizes[i] - 3) {
                    reset(field[Site]);
                    goto skip;
                }
            }
            _ToUpdate.push_back(Site);
        skip:;
            advance_coord(Sizes, Coord);
        }
    } else {
        throw std::runtime_error(
            std::format("ACTION [{}]: Unsupported boundary conditions ({}), check input script", GetName(), _BC));
    }
}

template <RealValue TImpl>
inline void WeakFieldEuclideanGR<TImpl>::update_LGR(const Lattice<field_type>& field) {
    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        _LGR[Site] = compute_LGR_loc(field, Site);
    }
}

template <RealValue TImpl>
inline auto WeakFieldEuclideanGR<TImpl>::compute_LGR_loc(const Lattice<field_type>& field, size_type site) const
    -> action_type {
    // Compute local derivatives
    std::array<field_type, 4> Dhmn;
    HField_math::diff(Dhmn[_t], field.n(site, _t), field[site]);
    HField_math::diff(Dhmn[_x], field.n(site, _x), field[site]);
    HField_math::diff(Dhmn[_y], field.n(site, _y), field[site]);
    HField_math::diff(Dhmn[_z], field.n(site, _z), field[site]);

    std::array<impl_type, 4> DTrh;
    for (int Rho = 0; Rho < 4; Rho++) {
        DTrh[Rho] = Dhmn[Rho][MUNU_00] + Dhmn[Rho][MUNU_11] + Dhmn[Rho][MUNU_22] + Dhmn[Rho][MUNU_33];
    }

    impl_type Ans1 = 0.0;
    impl_type Ans2 = 0.0;
    impl_type Ans3 = 0.0;
    impl_type Ans4 = 0.0;

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

    return _Kappa * (Ans1 + Ans2 + Ans3 + Ans4);
}

// inline void WeakFieldEuclideanGR::compute_Force_loc(const Lattice<field_type>& field, field_type& force, int
// Site) {
//     std::array<field_type, 4>                Dhc;  // Central derivatives
//     std::array<std::array<field_type, 4>, 4> Dhd;  // Diagonal derivatives
//     std::array<field_type, 4>                Jhc;  // Central sum
//     std::array<std::array<field_type, 4>, 4> Jhd;  // Diagonal Sum

//     // clang-format off
//     // Compute central derivatives
//     HField_math::diff(Dhc[_t], field.n(Site, _t), field.p(Site, _t), 0.5);
//     HField_math::diff(Dhc[_x], field.n(Site, _x), field.p(Site, _x), 0.5);
//     HField_math::diff(Dhc[_y], field.n(Site, _y), field.p(Site, _y), 0.5);
//     HField_math::diff(Dhc[_z], field.n(Site, _z), field.p(Site, _z), 0.5);
//     // Compute diagonal derivatives
//     HField_math::diff(Dhd[_t][_x], field.n(field.pI(Site, _x), _t), field.n(field.pI(Site, _t), _x), 0.5);
//     HField_math::diff(Dhd[_t][_y], field.n(field.pI(Site, _y), _t), field.n(field.pI(Site, _t), _y), 0.5);
//     HField_math::diff(Dhd[_t][_z], field.n(field.pI(Site, _z), _t), field.n(field.pI(Site, _t), _z), 0.5);
//     HField_math::diff(Dhd[_x][_t], field.n(field.pI(Site, _t), _x), field.n(field.pI(Site, _x), _t), 0.5);
//     HField_math::diff(Dhd[_x][_y], field.n(field.pI(Site, _y), _x), field.n(field.pI(Site, _x), _y), 0.5);
//     HField_math::diff(Dhd[_x][_z], field.n(field.pI(Site, _z), _x), field.n(field.pI(Site, _x), _z), 0.5);
//     HField_math::diff(Dhd[_y][_t], field.n(field.pI(Site, _t), _y), field.n(field.pI(Site, _y), _t), 0.5);
//     HField_math::diff(Dhd[_y][_x], field.n(field.pI(Site, _x), _y), field.n(field.pI(Site, _y), _x), 0.5);
//     HField_math::diff(Dhd[_y][_z], field.n(field.pI(Site, _z), _y), field.n(field.pI(Site, _y), _z), 0.5);
//     HField_math::diff(Dhd[_z][_t], field.n(field.pI(Site, _t), _z), field.n(field.pI(Site, _z), _t), 0.5);
//     HField_math::diff(Dhd[_z][_x], field.n(field.pI(Site, _x), _z), field.n(field.pI(Site, _z), _x), 0.5);
//     HField_math::diff(Dhd[_z][_y], field.n(field.pI(Site, _y), _z), field.n(field.pI(Site, _z), _y), 0.5);
//     // Compute central sums
//     HField_math::sum(Jhc[_t], field.n(Site, _t), field.p(Site, _t), 0.5);
//     HField_math::sum(Jhc[_x], field.n(Site, _x), field.p(Site, _x), 0.5);
//     HField_math::sum(Jhc[_y], field.n(Site, _y), field.p(Site, _y), 0.5);
//     HField_math::sum(Jhc[_z], field.n(Site, _z), field.p(Site, _z), 0.5);
//     // Compute diagonal sums
//     HField_math::sum(Jhd[_t][_x], field.n(field.pI(Site, _x), _t), field.n(field.pI(Site, _t), _x), 0.5);
//     HField_math::sum(Jhd[_t][_y], field.n(field.pI(Site, _y), _t), field.n(field.pI(Site, _t), _y), 0.5);
//     HField_math::sum(Jhd[_t][_z], field.n(field.pI(Site, _z), _t), field.n(field.pI(Site, _t), _z), 0.5);
//     HField_math::sum(Jhd[_x][_t], field.n(field.pI(Site, _t), _x), field.n(field.pI(Site, _x), _t), 0.5);
//     HField_math::sum(Jhd[_x][_y], field.n(field.pI(Site, _y), _x), field.n(field.pI(Site, _x), _y), 0.5);
//     HField_math::sum(Jhd[_x][_z], field.n(field.pI(Site, _z), _x), field.n(field.pI(Site, _x), _z), 0.5);
//     HField_math::sum(Jhd[_y][_t], field.n(field.pI(Site, _t), _y), field.n(field.pI(Site, _y), _t), 0.5);
//     HField_math::sum(Jhd[_y][_x], field.n(field.pI(Site, _x), _y), field.n(field.pI(Site, _y), _x), 0.5);
//     HField_math::sum(Jhd[_y][_z], field.n(field.pI(Site, _z), _y), field.n(field.pI(Site, _y), _z), 0.5);
//     HField_math::sum(Jhd[_z][_t], field.n(field.pI(Site, _t), _z), field.n(field.pI(Site, _z), _t), 0.5);
//     HField_math::sum(Jhd[_z][_x], field.n(field.pI(Site, _x), _z), field.n(field.pI(Site, _z), _x), 0.5);
//     HField_math::sum(Jhd[_z][_y], field.n(field.pI(Site, _y), _z), field.n(field.pI(Site, _z), _y), 0.5);
//     // clang-format on

//     // Diagonal Components
//     force[MUNU_00] = 2.0 * field[Site][MUNU_11] - Jhc[_y][MUNU_11] - Jhc[_z][MUNU_11]                      //
//                      + 2.0 * field[Site][MUNU_22] - Jhc[_x][MUNU_22] - Jhc[_z][MUNU_22]                    //
//                      + 2.0 * field[Site][MUNU_33] - Jhc[_x][MUNU_33] - Jhc[_y][MUNU_33]                    //
//                      - Dhc[_t][MUNU_01] + Dhc[_x][MUNU_01] + Dhd[_t][_x][MUNU_01]                          //
//                      - Dhc[_t][MUNU_02] + Dhc[_y][MUNU_02] + Dhd[_t][_y][MUNU_02]                          //
//                      - Dhc[_t][MUNU_03] + Dhc[_z][MUNU_03] + Dhd[_t][_z][MUNU_03]                          //
//                      - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_y][MUNU_12] - Jhd[_x][_y][MUNU_12]   //
//                      - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13]   //
//                      - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

//     force[MUNU_11] = 2.0 * field[Site][MUNU_00] - Jhc[_y][MUNU_00] - Jhc[_z][MUNU_00]                      //
//                      + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_z][MUNU_22]                    //
//                      + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_y][MUNU_33]                    //
//                      + Dhc[_t][MUNU_01] - Dhc[_x][MUNU_01] - Dhd[_t][_x][MUNU_01]                          //
//                      - Dhc[_x][MUNU_12] + Dhc[_y][MUNU_12] + Dhd[_x][_y][MUNU_12]                          //
//                      - Dhc[_x][MUNU_13] + Dhc[_z][MUNU_13] + Dhd[_x][_z][MUNU_13]                          //
//                      - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
//                      - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
//                      - field[Site][MUNU_23] + Jhc[_y][MUNU_23] + Jhc[_z][MUNU_23] - Jhd[_y][_z][MUNU_23];  //

//     force[MUNU_22] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_z][MUNU_00]                      //
//                      + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_z][MUNU_11]                    //
//                      + 2.0 * field[Site][MUNU_33] - Jhc[_t][MUNU_33] - Jhc[_x][MUNU_33]                    //
//                      + Dhc[_t][MUNU_02] - Dhc[_y][MUNU_02] - Dhd[_t][_y][MUNU_02]                          //
//                      + Dhc[_x][MUNU_12] - Dhc[_y][MUNU_12] - Dhd[_x][_y][MUNU_12]                          //
//                      - Dhc[_y][MUNU_23] + Dhc[_z][MUNU_23] + Dhd[_y][_z][MUNU_23]                          //
//                      - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
//                      - field[Site][MUNU_03] + Jhc[_t][MUNU_03] + Jhc[_z][MUNU_03] - Jhd[_t][_z][MUNU_03]   //
//                      - field[Site][MUNU_13] + Jhc[_x][MUNU_13] + Jhc[_z][MUNU_13] - Jhd[_x][_z][MUNU_13];  //

//     force[MUNU_33] = 2.0 * field[Site][MUNU_00] - Jhc[_x][MUNU_00] - Jhc[_y][MUNU_00]                      //
//                      + 2.0 * field[Site][MUNU_11] - Jhc[_t][MUNU_11] - Jhc[_y][MUNU_11]                    //
//                      + 2.0 * field[Site][MUNU_22] - Jhc[_t][MUNU_22] - Jhc[_x][MUNU_22]                    //
//                      + Dhc[_t][MUNU_03] - Dhc[_z][MUNU_03] - Dhd[_t][_z][MUNU_03]                          //
//                      + Dhc[_x][MUNU_13] - Dhc[_z][MUNU_13] - Dhd[_x][_z][MUNU_13]                          //
//                      + Dhc[_y][MUNU_23] - Dhc[_z][MUNU_23] - Dhd[_y][_z][MUNU_23]                          //
//                      - field[Site][MUNU_01] + Jhc[_t][MUNU_01] + Jhc[_x][MUNU_01] - Jhd[_t][_x][MUNU_01]   //
//                      - field[Site][MUNU_02] + Jhc[_t][MUNU_02] + Jhc[_y][MUNU_02] - Jhd[_t][_y][MUNU_02]   //
//                      - field[Site][MUNU_12] + Jhc[_x][MUNU_12] + Jhc[_z][MUNU_12] - Jhd[_x][_y][MUNU_12];  //

//     // Time-Space Components
//     // clang-format off
//     force[MUNU_01] = Dhc[_t][MUNU_00] - Dhc[_x][MUNU_00] - Dhd[_t][_x][MUNU_00]                         //
//                             - Dhc[_t][MUNU_11] + Dhc[_x][MUNU_11] + Dhd[_t][_x][MUNU_11]                //
//                             + Jhc[_t][MUNU_22] + Jhc[_x][MUNU_22] - Jhd[_t][_x][MUNU_22]                //
//                             + Jhc[_t][MUNU_33] + Jhc[_x][MUNU_33] - Jhd[_t][_x][MUNU_33]                //
//                             - 4.0 * field[Site][MUNU_01] - field[Site][MUNU_22] - field[Site][MUNU_33]  //
//                             + 2.0 * (Jhc[_y][MUNU_01] + Jhc[_z][MUNU_01])                               //
//                             + field[Site][MUNU_02] - field.n(Site, _x)[MUNU_02] - field.p(Site, _y)[MUNU_02] +
//                             field.n(field.pI(Site, _y), _x)[MUNU_02]  //
//                             + field[Site][MUNU_03] - field.n(Site, _x)[MUNU_03] - field.p(Site, _z)[MUNU_03] +
//                             field.n(field.pI(Site, _z), _x)[MUNU_03]  //
//                             + field[Site][MUNU_12] - field.n(Site, _t)[MUNU_12] - field.p(Site, _y)[MUNU_12] +
//                             field.n(field.pI(Site, _y), _t)[MUNU_12]  //
//                             + field[Site][MUNU_13] - field.n(Site, _t)[MUNU_13] - field.p(Site, _z)[MUNU_13] +
//                             field.n(field.pI(Site, _z), _t)[MUNU_13];

//     force[MUNU_02] = Dhc[_t][MUNU_00] - Dhc[_y][MUNU_00] - Dhd[_t][_y][MUNU_00]                         //
//                             - Dhc[_t][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_t][_y][MUNU_22]                //
//                             + Jhc[_t][MUNU_11] + Jhc[_y][MUNU_11] - Jhd[_t][_y][MUNU_11]                //
//                             + Jhc[_t][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_t][_y][MUNU_33]                //
//                             - 4.0 * field[Site][MUNU_02] - field[Site][MUNU_11] - field[Site][MUNU_33]  //
//                             + 2.0 * (Jhc[_x][MUNU_02] + Jhc[_z][MUNU_02])                               //
//                             + field[Site][MUNU_01] - field.n(Site, _y)[MUNU_01] - field.p(Site, _x)[MUNU_01] +
//                             field.n(field.pI(Site, _x), _y)[MUNU_01]  //
//                             + field[Site][MUNU_03] - field.n(Site, _y)[MUNU_03] - field.p(Site, _z)[MUNU_03] +
//                             field.n(field.pI(Site, _z), _y)[MUNU_03]  //
//                             + field[Site][MUNU_12] - field.n(Site, _t)[MUNU_12] - field.p(Site, _x)[MUNU_12] +
//                             field.n(field.pI(Site, _x), _t)[MUNU_12]  //
//                             + field[Site][MUNU_23] - field.n(Site, _t)[MUNU_23] - field.p(Site, _z)[MUNU_23] +
//                             field.n(field.pI(Site, _z), _t)[MUNU_23];

//     force[MUNU_03] = Dhc[_t][MUNU_00] - Dhc[_z][MUNU_00] - Dhd[_t][_z][MUNU_00]                         //
//                             - Dhc[_t][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_t][_z][MUNU_33]                //
//                             + Jhc[_t][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_t][_z][MUNU_11]                //
//                             + Jhc[_t][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_t][_z][MUNU_22]                //
//                             - 4.0 * field[Site][MUNU_03] - field[Site][MUNU_11] - field[Site][MUNU_22]  //
//                             + 2.0 * (Jhc[_x][MUNU_03] + Jhc[_y][MUNU_03])                               //
//                             + field[Site][MUNU_01] - field.n(Site, _z)[MUNU_01] - field.p(Site, _x)[MUNU_01] +
//                             field.n(field.pI(Site, _x), _z)[MUNU_01]  //
//                             + field[Site][MUNU_02] - field.n(Site, _z)[MUNU_02] - field.p(Site, _y)[MUNU_02] +
//                             field.n(field.pI(Site, _y), _z)[MUNU_02]  //
//                             + field[Site][MUNU_13] - field.n(Site, _t)[MUNU_13] - field.p(Site, _y)[MUNU_13] +
//                             field.n(field.pI(Site, _y), _t)[MUNU_13]  //
//                             + field[Site][MUNU_23] - field.n(Site, _t)[MUNU_23] - field.p(Site, _y)[MUNU_23] +
//                             field.n(field.pI(Site, _y), _t)[MUNU_23];

//     // Space-Space Components
//     force[MUNU_12] = Dhc[_x][MUNU_11] - Dhc[_y][MUNU_11] - Dhd[_x][_y][MUNU_11]                         //
//                             - Dhc[_x][MUNU_22] + Dhc[_y][MUNU_22] + Dhd[_x][_y][MUNU_22]                //
//                             + Jhc[_x][MUNU_00] + Jhc[_y][MUNU_00] - Jhd[_x][_y][MUNU_00]                //
//                             + Jhc[_x][MUNU_33] + Jhc[_y][MUNU_33] - Jhd[_x][_y][MUNU_33]                //
//                             - 4.0 * field[Site][MUNU_12] - field[Site][MUNU_00] - field[Site][MUNU_33]  //
//                             + 2.0 * (Jhc[_t][MUNU_12] + Jhc[_z][MUNU_12])                               //
//                             + field[Site][MUNU_01] - field.n(Site, _y)[MUNU_01] - field.p(Site, _t)[MUNU_01] +
//                             field.n(field.pI(Site, _t), _y)[MUNU_01]  //
//                             + field[Site][MUNU_02] - field.n(Site, _x)[MUNU_02] - field.p(Site, _t)[MUNU_02] +
//                             field.n(field.pI(Site, _t), _x)[MUNU_02]  //
//                             + field[Site][MUNU_13] - field.n(Site, _y)[MUNU_13] - field.p(Site, _z)[MUNU_13] +
//                             field.n(field.pI(Site, _z), _y)[MUNU_13]  //
//                             + field[Site][MUNU_23] - field.n(Site, _x)[MUNU_23] - field.p(Site, _z)[MUNU_23] +
//                             field.n(field.pI(Site, _z), _x)[MUNU_23];

//     force[MUNU_13] = Dhc[_x][MUNU_11] - Dhc[_z][MUNU_11] - Dhd[_x][_z][MUNU_11]                  //
//                             - Dhc[_x][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_x][_z][MUNU_33]                //
//                             + Jhc[_x][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_x][_z][MUNU_00]                //
//                             + Jhc[_x][MUNU_22] + Jhc[_z][MUNU_22] - Jhd[_x][_z][MUNU_22]                //
//                             - 4.0 * field[Site][MUNU_13] - field[Site][MUNU_00] - field[Site][MUNU_22]  //
//                             + 2.0 * (Jhc[_t][MUNU_13] + Jhc[_y][MUNU_13])                               //
//                             + field[Site][MUNU_01] - field.n(Site, _z)[MUNU_01] - field.p(Site, _t)[MUNU_01] +
//                             field.n(field.pI(Site, _t), _z)[MUNU_01]  //
//                             + field[Site][MUNU_03] - field.n(Site, _x)[MUNU_03] - field.p(Site, _t)[MUNU_03] +
//                             field.n(field.pI(Site, _t), _x)[MUNU_03]  //
//                             + field[Site][MUNU_12] - field.n(Site, _z)[MUNU_12] - field.p(Site, _y)[MUNU_12] +
//                             field.n(field.pI(Site, _y), _z)[MUNU_12]  //
//                             + field[Site][MUNU_23] - field.n(Site, _x)[MUNU_23] - field.p(Site, _y)[MUNU_23] +
//                             field.n(field.pI(Site, _y), _x)[MUNU_23];

//     force[MUNU_23] = Dhc[_y][MUNU_22] - Dhc[_z][MUNU_22] - Dhd[_y][_z][MUNU_22]                  //
//                             - Dhc[_y][MUNU_33] + Dhc[_z][MUNU_33] + Dhd[_y][_z][MUNU_33]                //
//                             + Jhc[_y][MUNU_00] + Jhc[_z][MUNU_00] - Jhd[_y][_z][MUNU_00]                //
//                             + Jhc[_y][MUNU_11] + Jhc[_z][MUNU_11] - Jhd[_y][_z][MUNU_11]                //
//                             - 4.0 * field[Site][MUNU_23] - field[Site][MUNU_00] - field[Site][MUNU_11]  //
//                             + 2.0 * (Jhc[_t][MUNU_23] + Jhc[_x][MUNU_23])                               //
//                             + field[Site][MUNU_02] - field.n(Site, _z)[MUNU_02] - field.p(Site, _t)[MUNU_02] +
//                             field.n(field.pI(Site, _t), _z)[MUNU_02]  //
//                             + field[Site][MUNU_03] - field.n(Site, _y)[MUNU_03] - field.p(Site, _t)[MUNU_03] +
//                             field.n(field.pI(Site, _t), _y)[MUNU_03]  //
//                             + field[Site][MUNU_12] - field.n(Site, _z)[MUNU_12] - field.p(Site, _x)[MUNU_12] +
//                             field.n(field.pI(Site, _x), _z)[MUNU_12]  //
//                             + field[Site][MUNU_13] - field.n(Site, _y)[MUNU_13] - field.p(Site, _x)[MUNU_13] +
//                             field.n(field.pI(Site, _x), _y)[MUNU_13];
//     // clang-format on

//     // force *= kappa;
// }

}  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  HDF5 helper method Implementatin
--------------------------------------------------------------------------------------------------*/
namespace reticolo {
template <>
auto make_H5_Type<action::WeakFieldEuclideanGR<RealF>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::WeakFieldEuclideanGR<RealF>::Observables));
    H5Tinsert(DataTypeHid, "R", HOFFSET(action::WeakFieldEuclideanGR<RealF>::Observables, R), H5T_NATIVE_FLOAT);
    return DataTypeHid;
}
template <>
auto make_H5_Type<action::WeakFieldEuclideanGR<RealD>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::WeakFieldEuclideanGR<RealD>::Observables));
    H5Tinsert(DataTypeHid, "R", HOFFSET(action::WeakFieldEuclideanGR<RealD>::Observables, R), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}
}  // namespace reticolo

/*--------------------------------------------------------------------------------------------------
  MMonteCarlo::Metropolis::updateField() Specialization
--------------------------------------------------------------------------------------------------*/
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"  // IWYU pragma: keep

namespace reticolo {

template <>
void MMonteCarlo::Metropolis<action::WeakFieldEuclideanGR<RealF>>::updateField(
    lattice_type&                        field,   //
    action::WeakFieldEuclideanGR<RealF>& action,  //
    monte_carlo_data_type&               state,   //
    std::mt19937_64&                     rng)                         //
{
    impl_type u;   // Marsaglia polar method support variables
    impl_type v;   //
    impl_type s;   //
    impl_type fp;  //
    impl_type Scale = _ProposalWidth * action._LPFm / action._AA;

    size_type   Acc = 0;       // acceptance
    action_type SVarTot(0.0);  // cumulative action variation

    // for (size_type Site = 0; Site < field.getNsites(); Site++) {
    for (const auto& Site : action._ToUpdate) {
        // Generate a randomized local field variation
        // field_type FieldVar;  // local field variation
        field_type FieldOld = field[Site];

        // wiggle h
        for (int i = 0; i < 10; i++) {
            do {
                u = _Unifc(rng);
                v = _Unifc(rng);
                s = u * u + v * v;
            } while (s > 1 || s == 0);
            fp = std::sqrt(-2 * std::log(s) / s);
            field[Site][i] += v * fp * Scale;
            field[Site][++i] += u * fp * Scale;
        }

        // Compute the updated Lagrangian in the surrounding sites
        std::vector<action_type> LGRPost;             // Vector storing the new curvature values in the check sites
        action_type              ActionChange = 0.0;  // Cumulative curvature variation
        for (size_type& CheckSite : action._Checks[Site]) {
            impl_type Tmp = action.compute_LGR_loc(field, CheckSite);
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - action._LGR[CheckSite];
            } else {
                break;
            }
        }
        // Metropolis acceptance + positive action
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
void MMonteCarlo::Metropolis<action::WeakFieldEuclideanGR<RealD>, std::mt19937_64>::updateField(
    lattice_type&                        field,   //
    action::WeakFieldEuclideanGR<RealD>& action,  //
    monte_carlo_data_type&               state,   //
    std::mt19937_64&                     rng)                         //
{
    impl_type u;   // Marsaglia polar method support variables
    impl_type v;   //
    impl_type s;   //
    impl_type fp;  //
    impl_type Scale = _ProposalWidth * action._LPFm / action._AA;

    size_type   Acc = 0;       // acceptance
    action_type SVarTot(0.0);  // cumulative action variation

    field_type FieldOld;

    // for (size_type Site = 0; Site < field.getNsites(); Site++) {
    for (const auto& Site : action._ToUpdate) {
        // store old value of the field
        FieldOld = field[Site];

        // wiggle h
        for (int i = 0; i < 10; i++) {
            do {
                u = _Unifc(rng);
                v = _Unifc(rng);
                s = u * u + v * v;
            } while (s > 1 || s == 0);
            fp = std::sqrt(-2 * std::log(s) / s);
            field[Site][i] += v * fp * Scale;
            field[Site][++i] += u * fp * Scale;
        }

        // Compute the updated Lagrangian in the surrounding sites
        std::vector<action_type> LGRPost;             // Vector storing the new curvature values in the check sites
        action_type              ActionChange = 0.0;  // Cumulative curvature variation
        for (size_type& CheckSite : action._Checks[Site]) {
            impl_type Tmp = action.compute_LGR_loc(field, CheckSite);
            if (Tmp > 0.0) {
                LGRPost.push_back(Tmp);
                ActionChange += Tmp - action._LGR[CheckSite];
            } else {
                break;
            }
        }
        // Metropolis acceptance + positive action
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

// /*--------------------------------------------------------------------------------------------------
//   montecarlo::HMC::updateField() Specialization
// --------------------------------------------------------------------------------------------------*/
// #include "reticolo/montecarlo/HMC.hpp"

// template <>
// void reticolo::montecarlo::HMC<reticolo::action::WeakFieldEuclideanGR>::updateField() {
//     int        NSites = _Field.getNsites();
//     int        Acc = 0;
//     action_type SvarTot(0.0);
//     field_type  OldField;
//     field_type  Mom;
//     field_type  Force;

//     for (int Site = 0; Site < NSites; Site++) {
//         // save the old field configuration;
//         OldField = _Field[Site];
//         // Compute start Hamiltonian
//         randomize(Mom, 1.0, _Norm, _Rng);
//         impl_type OldK = 0.5 * Mom.dot();
//         // Compute Forces
//         _Action.compute_Force_loc(_Field, Force, Site);
//         // Momenta half step
//         Mom += 0.5 * _Stepsize * Force;
//         // Leapfrog algorithm
//         for (uint Step = 0; Step < _Steps; Step++) {
//             // Update field
//             _Field[Site] += _Stepsize * Mom;
//             // Compute new forces
//             _Action.compute_Force_loc(_Field, Force, Site);
//             // Update momenta
//             Mom -= _Stepsize * Force;
//         }
//         // Half step momenta roll-back
//         Mom += 0.5 * _Stepsize * Force;
//         // Compute end Hamiltonian
//         impl_type NewK = 0.5 * Mom.dot();
//         // Compute the updated Lagrangian in the surrounding sites
//         std::vector<action_type> LGRPost;             // Vector storing the new curvature values in the check sites
//         action_type              ActionChange = 0.0;  // Cumulative curvature variation
//         for (int& CheckSite : _Action._Checks[Site]) {
//             impl_type Tmp = _Action.compute_LGR_loc(_Field, CheckSite) / _Action.kappa;
//             if (Tmp > 0.0) {
//                 LGRPost.push_back(Tmp);
//                 ActionChange += Tmp - _Action._LGR[CheckSite];
//             } else {
//                 break;
//             }
//         }
//         // Metropolis acceptance + positive action
//         if (LGRPost.size() == 5 && exp(-(ActionChange + NewK - OldK)) > _Unif(_Rng)) {
//             Acc++;
//             SvarTot += ActionChange;
//             for (uint CheckSite = 0; CheckSite < 5; CheckSite++) {
//                 _Action._LGR[_Action._Checks[Site][CheckSite]] = LGRPost[CheckSite];
//             }
//         } else {
//             _Field[Site] = OldField;
//         }
//     }
//     _McStats.update(static_cast<impl_type>(Acc) / NSites, SvarTot);
// }
