/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/montecarlo_data.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <format>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep

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
    data(double acc, ActionType action, ActionType action_change)
        : _Acceptance(acc), _S(double(action)), _DS(double(action_change)){};  // Parameter

    /* Update values */
    void setS(ActionType SReal) { _S = SReal; }
    void update(double acc, ActionType dS) {
        _Acceptance = acc;
        _S += double(dS);
        _DS = double(dS);
    }
    void softReset() {
        _Acceptance = 0;
        _DS = 0;
    }
    void hardReset() {
        _Acceptance = 0;
        _S = 0;
        _DS = 0;
    }

    /* Get/Set and Data/Str dump */
    [[nodiscard]] auto dump_str() -> std::string { return std::format("{:+8e},{:+8e},{:+8e}", _Acceptance, _S, _DS); }
    [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({_Acceptance, _S, _DS}); }

    /* Hdf5 CompType */
    static auto make_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(data<ActionType>));
        Type.insertMember("acceptance", HOFFSET(data<ActionType>, _Acceptance), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S", HOFFSET(data<ActionType>, _S), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS", HOFFSET(data<ActionType>, _DS), H5::PredType::NATIVE_DOUBLE);
        return Type;
    };

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
    data(double acc, ActionType action, ActionType action_change)
        : _Acceptance(acc),
          _SRe(double(action.real())),
          _SIm(double(action.imag())),
          _DSRe(double(action_change.real())),
          _DSIm(double(action_change.imag())){};  // Parameter

    /* Update values */
    void setS(ActionType SCmplx) {
        _SRe = SCmplx.real();
        _SIm = SCmplx.imag();
    }
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

    /* Get/Set and Data/Str dump */
    [[nodiscard]] auto dump_str() -> std::string {
        return std::format("{:+8e},{:+8e},{:+8e},{:+8e},{:+8e}", _Acceptance, _SRe, _SIm, _DSRe, _DSIm);
    }
    [[nodiscard]] auto dump_data() const -> std::vector<double> {
        return std::vector<double>({_Acceptance, _SRe, _SIm, _DSRe, _DSIm});
    }

    /* Hdf5 CompType */
    static auto make_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(data<ActionType>));
        Type.insertMember("acceptance", HOFFSET(data<ActionType>, _Acceptance), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S_re", HOFFSET(data<ActionType>, _SRe), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S_im", HOFFSET(data<ActionType>, _SIm), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS_re", HOFFSET(data<ActionType>, _DSRe), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS_im", HOFFSET(data<ActionType>, _DSIm), H5::PredType::NATIVE_DOUBLE);
        return Type;
    };

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
