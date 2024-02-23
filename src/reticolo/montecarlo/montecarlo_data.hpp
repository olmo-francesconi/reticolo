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
    double acceptance;
    double S;
    double dS;

    [[nodiscard]] auto dump_str() -> std::string { return std::format("{:+8e},{:+8e},{:+8e}", acceptance, S, dS); }
    [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({acceptance, S, dS}); }

    static auto make_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(data<ActionType>));
        Type.insertMember("acceptance", HOFFSET(data<ActionType>, acceptance), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S", HOFFSET(data<ActionType>, S), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS", HOFFSET(data<ActionType>, dS), H5::PredType::NATIVE_DOUBLE);
        return Type;
    };

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        acceptance += rhs.acceptance;
        S += rhs.S;
        dS += rhs.dS;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        acceptance *= rhs;
        S *= rhs;
        dS *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        acceptance /= rhs;
        S /= rhs;
        dS /= rhs;
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
    double acceptance;
    double S_re;
    double S_im;
    double dS_re;
    double dS_im;

    [[nodiscard]] auto dump_str() -> std::string {
        return std::format("{:+8e},{:+8e},{:+8e},{:+8e},{:+8e}", acceptance, S_re, S_im, dS_re, dS_im);
    }
    [[nodiscard]] auto dump_data() const -> std::vector<double> {
        return std::vector<double>({acceptance, S_re, S_im, dS_re, dS_im});
    }

    static auto make_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(data<ActionType>));
        Type.insertMember("acceptance", HOFFSET(data<ActionType>, acceptance), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S_re", HOFFSET(data<ActionType>, S_re), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("S_im", HOFFSET(data<ActionType>, S_im), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS_re", HOFFSET(data<ActionType>, dS_re), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("dS_im", HOFFSET(data<ActionType>, dS_im), H5::PredType::NATIVE_DOUBLE);
        return Type;
    };

    auto operator+=(const data<ActionType>& rhs) -> data<ActionType>& {
        acceptance += rhs.acceptance;
        S_re += rhs.S_re;
        S_im += rhs.S_im;
        dS_re += rhs.dS_re;
        dS_im += rhs.dS_im;
        return *this;
    }

    auto operator*=(const double& rhs) -> data<ActionType>& {
        acceptance *= rhs;
        S_re *= rhs;
        S_im *= rhs;
        dS_re *= rhs;
        dS_im *= rhs;
        return *this;
    }

    auto operator/=(const double& rhs) -> data<ActionType>& {
        acceptance /= rhs;
        S_re /= rhs;
        S_im /= rhs;
        dS_re /= rhs;
        dS_im /= rhs;
        return *this;
    }

    friend auto operator+(data<ActionType> lhs, const data<ActionType>& rhs) -> data<ActionType> {
        lhs += rhs;
        return lhs;
    }
};

}  // namespace reticolo::montecarlo
