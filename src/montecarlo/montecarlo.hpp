/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/montecarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "tools/types.hpp"

#include "H5Cpp.h"

#include <string>
#include <format>

namespace reticolo
{
    namespace montecarlo
    {
        template <typename T>
            requires RealValue<T> || ComplexValue<T>
        struct data;

        template <RealValue ActionType>
        struct data<ActionType>
        {
            double acceptance;
            double S;
            double dS;

            std::string dump_str() { return std::format("{:+8e},{:+8e},{:+8e}", acceptance, S, dS); }
            std::vector<double> dump_data() const { return std::vector<double>({acceptance, S, dS}); }

            data<ActionType> &operator+=(const data<ActionType> &rhs)
            {
                acceptance += rhs.acceptance;
                S += rhs.S;
                dS += rhs.dS;
                return *this;
            }

            data<ActionType> &operator*=(const double &rhs)
            {
                acceptance *= rhs;
                S *= rhs;
                dS *= rhs;
                return *this;
            }

            data<ActionType> &operator/=(const double &rhs)
            {
                acceptance /= rhs;
                S /= rhs;
                dS /= rhs;
                return *this;
            }

            friend data<ActionType> operator+(data<ActionType> lhs, const data<ActionType> &rhs)
            {
                lhs += rhs;
                return lhs;
            }
        };

        template <ComplexValue ActionType>
        struct data<ActionType>
        {
            double acceptance;
            double S_re;
            double S_im;
            double dS_re;
            double dS_im;

            std::string dump_str() { return std::format("{:+8e},{:+8e},{:+8e},{:+8e},{:+8e}", acceptance, S_re, S_im, dS_re, dS_im); }
            std::vector<double> dump_data() const { return std::vector<double>({acceptance, S_re, S_im, dS_re, dS_im}); }

            data<ActionType> &operator+=(const data<ActionType> &rhs)
            {
                acceptance += rhs.acceptance;
                S_re += rhs.S_re;
                S_im += rhs.S_im;
                dS_re += rhs.dS_re;
                dS_im += rhs.dS_im;
                return *this;
            }

            data<ActionType> &operator*=(const double &rhs)
            {
                acceptance *= rhs;
                S_re *= rhs;
                S_im *= rhs;
                dS_re *= rhs;
                dS_im *= rhs;
                return *this;
            }

            data<ActionType> &operator/=(const double &rhs)
            {
                acceptance /= rhs;
                S_re /= rhs;
                S_im /= rhs;
                dS_re /= rhs;
                dS_im /= rhs;
                return *this;
            }

            friend data<ActionType> operator+(data<ActionType> lhs, const data<ActionType> &rhs)
            {
                lhs += rhs;
                return lhs;
            }
        };

        template <RealValue ActionType>
        H5::CompType make_mc_data_hdf5_CompType()
        {
            H5::CompType type(sizeof(data<ActionType>));
            type.insertMember("acceptance", HOFFSET(data<ActionType>, acceptance), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S", HOFFSET(data<ActionType>, S_re), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS", HOFFSET(data<ActionType>, dS_re), H5::PredType::NATIVE_DOUBLE);
            return type;
        };

        template <ComplexValue ActionType>
        H5::CompType make_mc_data_hdf5_CompType()
        {
            H5::CompType type(sizeof(data<ActionType>));
            type.insertMember("acceptance", HOFFSET(data<ActionType>, acceptance), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S_re", HOFFSET(data<ActionType>, S_re), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S_im", HOFFSET(data<ActionType>, S_im), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS_re", HOFFSET(data<ActionType>, dS_re), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS_im", HOFFSET(data<ActionType>, dS_im), H5::PredType::NATIVE_DOUBLE);
            return type;
        };

    }
}