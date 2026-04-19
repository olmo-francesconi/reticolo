/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/montecarlo_data.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

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
    data(const data&) = default;
    data(data&&) = default;
    data(double acceptance, ActionType action, ActionType /*unused*/)
        : _Acceptance(acceptance), _S(double(action)) {};  // Parameter

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
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& = default;
    auto operator=(data<ActionType>&&) -> data<ActionType>& = default;

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
    data(const data&) = default;
    data(data&&) = default;
    data(double acceptance, ActionType action, ActionType /*unused*/)
        : _Acceptance(acceptance), _SRe(double(action.real())), _SIm(double(action.imag())) {};  // Parameter

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
    auto operator=(const data<ActionType>& rhs) -> data<ActionType>& = default;
    auto operator=(data<ActionType>&&) -> data<ActionType>& = default;

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
