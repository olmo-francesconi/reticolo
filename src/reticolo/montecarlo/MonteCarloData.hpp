/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/montecarlo_data.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Tpublic.h>

#include "H5Cpp.h"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::montecarlo {
/* Generic template declaration - requires the action type to be either real or complex */
template <typename T>
    requires RealValue<T> || ComplexValue<T>
struct data;

/* Real-valued action specialization */
template <RealValue ActionType>
struct data<ActionType> {
    double _Acceptance;
    double _S;
    double _DS;

    /* Constructors*/
    data() = default;  // Default
    data(double acceptance, ActionType action, ActionType action_change)
        : _Acceptance(acceptance), _S(double(action)), _DS(double(action_change)){};  // Parameter

    /* Update values */
    void setS(ActionType SReal) { _S = SReal; }
    auto getS() -> ActionType { return _S; }
    void update(double acc, ActionType dS) {
        _Acceptance = acc;
        _S += double(dS);
        _DS = double(dS);
    }

    /* Operators overloading */
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance = rhs._Acceptance;
        _S = rhs._S;
        _DS = rhs._DS;
        return *this;
    }

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance += rhs._Acceptance;
        _S += rhs._S;
        _DS += rhs._DS;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        _Acceptance *= rhs;
        _S *= rhs;
        _DS *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        _Acceptance /= rhs;
        _S /= rhs;
        _DS /= rhs;
        return *this;
    }

    friend auto operator+(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs += rhs;
        return lhs;
    }
};

/* Complex-valued action specialization */
template <ComplexValue ActionType>
struct data<ActionType> {
    double _Acceptance;
    double _SRe;
    double _SIm;
    double _DSRe;
    double _DSIm;

    /* Constructors*/
    data() = default;  // Default
    data(double acceptance, ActionType action, ActionType action_change)
        : _Acceptance(acceptance),
          _SRe(double(action.real())),
          _SIm(double(action.imag())),
          _DSRe(double(action_change.real())),
          _DSIm(double(action_change.imag())){};  // Parameter

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
        _DSRe = dSCmplx.real();
        _DSIm = dSCmplx.imag();
    }
    void softReset() {
        _Acceptance = 0.0;
        _DSRe = 0.0;
        _DSIm = 0.0;
    }
    void hardReset() {
        _Acceptance = 0;
        _SRe = 0.0;
        _SIm = 0.0;
        _DSRe = 0.0;
        _DSIm = 0.0;
    }

    /* Operators overloading */
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance = rhs._Acceptance;
        _SRe = rhs._SRe;
        _SIm = rhs._SIm;
        _DSRe = rhs._DSRe;
        _DSIm = rhs._DSIm;
        return *this;
    }

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        _Acceptance += rhs._Acceptance;
        _SRe += rhs._SRe;
        _SIm += rhs._SIm;
        _DSRe += rhs._DSRe;
        _DSIm += rhs._DSIm;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        _Acceptance *= rhs;
        _SRe *= rhs;
        _SIm *= rhs;
        _DSRe *= rhs;
        _DSIm *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        _Acceptance /= rhs;
        _SRe /= rhs;
        _SIm /= rhs;
        _DSRe /= rhs;
        _DSIm /= rhs;
        return *this;
    }

    friend auto operator+(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs += rhs;
        return lhs;
    }
};

}  // namespace reticolo::montecarlo

namespace reticolo {

template <>
auto make_H5_Type<montecarlo::data<RealF>>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(montecarlo::data<RealD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(montecarlo::data<RealD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(montecarlo::data<RealD>, _S), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS", HOFFSET(montecarlo::data<RealD>, _DS), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<montecarlo::data<RealD>>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(montecarlo::data<RealD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(montecarlo::data<RealD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S", HOFFSET(montecarlo::data<RealD>, _S), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS", HOFFSET(montecarlo::data<RealD>, _DS), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<montecarlo::data<ComplexF>>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(montecarlo::data<ComplexD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(montecarlo::data<ComplexD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(montecarlo::data<ComplexD>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(montecarlo::data<ComplexD>, _SIm), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS_re", HOFFSET(montecarlo::data<ComplexD>, _DSRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS_im", HOFFSET(montecarlo::data<ComplexD>, _DSIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};

template <>
auto make_H5_Type<montecarlo::data<ComplexD>>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(montecarlo::data<ComplexD>));
    H5Tinsert(DataTypeHid, "acceptance", HOFFSET(montecarlo::data<ComplexD>, _Acceptance), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_re", HOFFSET(montecarlo::data<ComplexD>, _SRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "S_im", HOFFSET(montecarlo::data<ComplexD>, _SIm), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS_re", HOFFSET(montecarlo::data<ComplexD>, _DSRe), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "dS_im", HOFFSET(montecarlo::data<ComplexD>, _DSIm), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
};
}  // namespace reticolo
