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
#include "reticolo/types/core.hpp"

namespace reticolo::montecarlo {
/* Generic template declaration - requires the action type to be either real or complex */
template <typename T>
    requires RealValue<T> || ComplexValue<T>
class data;

/* Real-valued action specialization */
template <RealValue ActionType>
class data<ActionType> {
  private:
    double _Acceptance;
    double _S;
    double _DS;

  public:
    /* Constructors*/
    data() = default;                                                                                    // Default
    data(double acc, ActionType S, ActionType dS) : _Acceptance(acc), _S(double(S)), _DS(double(dS)){};  // Parameter
    data(data& other) = default;                                                                         // Copy
    data(data&& other) = default;                                                                        // Move

    /* Update values */
    void update(double acc, ActionType dS) {
        _Acceptance = acc;
        _S += double(dS);
        _DS = double(dS);
    }
    void setS(ActionType S) { _S = double(S); }

    /* Data output */
    [[nodiscard]] auto getAcceptance() -> double { return _Acceptance; }
    [[nodiscard]] auto getS() -> double { return _S; }
    [[nodiscard]] auto getDS() -> double { return _DS; }
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
class data<ActionType> {
  private:
    double _Acceptance;
    double _SRe;
    double _SIm;
    double _DSRe;
    double _DSIm;

  public:
    /* Constructors*/
    data() = default;  // Default
    data(double acc, ActionType S, ActionType dS)
        : _Acceptance(acc),
          _SRe(double(S.real())),
          _SIm(double(S.imag())),
          _DSRe(double(dS.real())),
          _DSIm(double(dS.imag())){};  // Parameter
    data(data& other) = default;       // Copy
    data(data&& other) = default;      // Move

    /* Update values */
    void update(double acc, ActionType dS) {
        _Acceptance = acc;
        _SRe += dS.real();
        _SIm += dS.imag();
        _DSRe = dS.real();
        _DSIm = dS.imag();
    }
    void setS(ActionType S) {
        _SRe = double(S.real());
        _SIm = double(S.imag());
    }

    /* Data output */
    [[nodiscard]] auto getAcceptance() -> double { return _Acceptance; }
    [[nodiscard]] auto getS() -> ComplexD { return {_SRe, _SIm}; }
    [[nodiscard]] auto getDS() -> ComplexD { return {_DSRe, _DSIm}; }
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
