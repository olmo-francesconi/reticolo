/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/montecarlo_data.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/real.hpp"

namespace reticolo::MMonteCarlo {
/* Generic template declaration - requires the action type to be either real or complex */
template <typename T>
    requires RealValue<T> || ComplexValue<T>
struct data;

/* Real-valued action specialization */
template <RealValue ActionType>
struct data<ActionType> {
    double _Acceptance;
    double _S;

    /* Constructors*/
    data() = default;  // Default
    data(double acceptance, ActionType action, ActionType action_change)
        : _Acceptance(acceptance), _S(double(action)){};  // Parameter

    /* Update values */
    void setS(ActionType SReal) { _S = SReal; }
    auto getS() -> ActionType { return _S; }
    void update(double acc, ActionType dS) {
        _Acceptance = acc;
        _S += double(dS);
    }
    void softReset() { _Acceptance = 0.0; }
    void hardReset() {
        _Acceptance = 0;
        _S = 0.0;
    }

    /* Operators overloading */
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance = rhs._Acceptance;
        _S = rhs._S;
        return *this;
    }

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance += rhs._Acceptance;
        _S += rhs._S;
        return *this;
    }

    auto operator-=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance -= rhs._Acceptance;
        _S -= rhs._S;
        return *this;
    }

    auto operator*=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance *= rhs._Acceptance;
        _S *= rhs._S;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        _Acceptance *= rhs;
        _S *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        _Acceptance /= rhs;
        _S /= rhs;
        return *this;
    }

    friend auto operator+(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs += rhs;
        return lhs;
    }

    friend auto operator-(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs -= rhs;
        return lhs;
    }

    friend auto operator*(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs *= rhs;
        return lhs;
    }

    friend auto operator/(data<ActionType> lhs, const double& rhs) -> data<ActionType> {
        lhs /= rhs;
        return lhs;
    }
};

/* Complex-valued action specialization */
template <ComplexValue ActionType>
struct data<ActionType> {
    double _Acceptance;
    double _SRe;
    double _SIm;

    /* Constructors*/
    data() = default;  // Default
    data(double acceptance, ActionType action, ActionType action_change)
        : _Acceptance(acceptance), _SRe(double(action.real())), _SIm(double(action.imag())){};  // Parameter

    /* Update values */
    void setS(ActionType SCmplx) {
        _SRe = SCmplx.real();
        _SIm = SCmplx.imag();
    }
    auto getS() -> ActionType { return ActionType(_SRe, _SIm); }
    void update(double acc, ActionType dSCmplx) {
        _Acceptance = acc;
        _SRe += dSCmplx.real();
        _SIm += dSCmplx.imag();
    }
    void softReset() { _Acceptance = 0.0; }
    void hardReset() {
        _Acceptance = 0;
        _SRe = 0.0;
        _SIm = 0.0;
    }

    /* Operators overloading */
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance = rhs._Acceptance;
        _SRe = rhs._SRe;
        _SIm = rhs._SIm;
        return *this;
    }

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance += rhs._Acceptance;
        _SRe += rhs._SRe;
        _SIm += rhs._SIm;
        return *this;
    }

    auto operator-=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance -= rhs._Acceptance;
        _SRe -= rhs._SRe;
        _SIm -= rhs._SIm;
        return *this;
    }

    auto operator*=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance *= rhs._Acceptance;
        _SRe *= rhs._SRe;
        _SIm *= rhs._SIm;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        _Acceptance *= rhs;
        _SRe *= rhs;
        _SIm *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        _Acceptance /= rhs;
        _SRe /= rhs;
        _SIm /= rhs;
        return *this;
    }

    friend auto operator+(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs += rhs;
        return lhs;
    }

    friend auto operator-(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs -= rhs;
        return lhs;
    }

    friend auto operator*(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs *= rhs;
        return lhs;
    }

    friend auto operator/(data<ActionType> lhs, const double& rhs) -> data<ActionType> {
        lhs /= rhs;
        return lhs;
    }
};

}  // namespace reticolo::MMonteCarlo

namespace reticolo {

template <>
auto make_H5_Type<MMonteCarlo::data<RealF>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<RealD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<RealD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(MMonteCarlo::data<RealD>, _S), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<MMonteCarlo::data<RealD>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<RealD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<RealD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(MMonteCarlo::data<RealD>, _S), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<MMonteCarlo::data<ComplexF>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<ComplexD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<ComplexD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(MMonteCarlo::data<ComplexD>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(MMonteCarlo::data<ComplexD>, _SIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<MMonteCarlo::data<ComplexD>>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(MMonteCarlo::data<ComplexD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(MMonteCarlo::data<ComplexD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(MMonteCarlo::data<ComplexD>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(MMonteCarlo::data<ComplexD>, _SIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};
}  // namespace reticolo
