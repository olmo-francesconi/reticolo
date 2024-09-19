/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

// #include <cmath>
#include <complex>
#include <concepts>
#include <sstream>
#include <string>

#include "reticolo/action/actionBase.hpp"
#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep
#include "reticolo/core/types/coord.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo {
namespace action {

/*--------------------------------------------------------------------------------------------------
  RelativisticBoseGas Class Declaration
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
class RelativisticBoseGas : public ActionBase<std::complex<TImpl>, std::complex<TImpl>, TImpl> {
  public:
    /* Types and public action metadata */
    using base_type = ActionBase<std::complex<TImpl>, std::complex<TImpl>, TImpl>;
    using action_type = base_type::action_type;
    using field_type = base_type::field_type;
    using size_type = base_type::size_type;
    using impl_type = TImpl;
    static constexpr int Dims = 4;     // Dimensions of the action
    static constexpr int Stencil = 2;  // Minimum step size for multi-thread safety

    /* Algorithm capabilities */
    static constexpr bool IsMetropolisCapable = true;
    static constexpr bool IsHmcCapable = true;
    static constexpr bool IsLLRCapable = true;

    /* Action parameters */
    struct Params {
        TImpl lambda;
        TImpl eta;
        TImpl mu;
        Params() : lambda(1.0), eta(9.0), mu(0) {};
        Params(TImpl lambda, TImpl eta, TImpl chem_mu) : lambda(lambda), eta(eta), mu(chem_mu) {};
    } p;

    /* Observables */
    struct Observables {
        TImpl phi2;
        TImpl density;
        void  reset() { phi2 = 0.0, density = 0.0; };

        auto operator+=(const Observables& rhs) -> Observables {
            phi2 += rhs.phi2;
            density += rhs.density;
            return *this;
        };
        auto operator/=(const double& rhs) -> Observables {
            phi2 /= rhs;
            density /= rhs;
            return *this;
        };
    };

    /* Constructors */
    RelativisticBoseGas() = default;

    /* setup */
    void setup(const YAML::Node& ActionParams) override;

    /* Sync with lattice */
    void lattice_sync(Lattice<field_type>& field) override;

    /* Gloabal and local action computations */

    auto compute_S(Lattice<field_type>& field) -> action_type override;
    auto compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type override;
    auto compute_dS_loc(Lattice<field_type>& field, const field_type& dphi, size_type site) -> action_type override;

    /* HMC methods */
    void compute_Forces(Lattice<field_type>& field, Lattice<field_type>& forces) override;
    void compute_LLRForces(Lattice<field_type>& field, Lattice<field_type>& forces, TImpl Sk, TImpl width,
                           TImpl ak) override;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(Lattice<field_type>& field) -> Observables;

    /* Log stuff */
    auto GetName() -> std::string override;
    auto GetParameters() -> std::string override;
};

/*--------------------------------------------------------------------------------------------------
  Private methods Implementation
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::GetName() -> std::string {
    std::string Res = "Relativistic Bose Gas";
    Res += std::same_as<impl_type, RealF> ? " [float]" : " [double]";
    return Res;
};

template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::GetParameters() -> std::string {
    std::stringstream Res;
    Res << "[ lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta)
        << ", mu : " << std::format("{:4.1f}", p.mu) << " ]";
    return Res.str();
};

/*--------------------------------------------------------------------------------------------------
  Public methods Implementation
--------------------------------------------------------------------------------------------------*/
template <RealValue TImpl>
inline void RelativisticBoseGas<TImpl>::setup(const YAML::Node& ActionParams) {
    p.lambda = ActionParams["lambda"].as<TImpl>();
    p.eta = ActionParams["eta"].as<TImpl>();
    p.mu = ActionParams["mu"].as<TImpl>();
}

template <RealValue TImpl>
inline void RelativisticBoseGas<TImpl>::lattice_sync(Lattice<field_type>& field) {
    // Maybe do some checks here
}

template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::compute_S(Lattice<field_type>& field) -> action_type {
    TImpl      Real = 0.0;
    TImpl      Imag = 0.0;
    field_type Phi;
    TImpl      Phi2;

    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];

        Phi2 = dot(Phi);

        Real += 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                (Phi.real() * field.n(Site, _x).real() + Phi.imag() * field.n(Site, _x).imag()) -
                (Phi.real() * field.n(Site, _y).real() + Phi.imag() * field.n(Site, _y).imag()) -
                (Phi.real() * field.n(Site, _z).real() + Phi.imag() * field.n(Site, _z).imag()) -
                cosh(p.mu) * (Phi.real() * field.n(Site, _t).real() + Phi.imag() * field.n(Site, _t).imag());

        Imag += Phi.real() * field.n(Site, _t).imag() - Phi.imag() * field.n(Site, _t).real();
    }

    return {Real, Imag};
}

template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type {
    field_type Phi = field[site];
    field_type PhiNt = field.n(site, _t);
    field_type PhiPt = field.p(site, _t);

    TImpl Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

    TImpl Real = 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                 Phi.real() * (field.n(site, _x).real() + field.p(site, _x).real() +  //
                               field.n(site, _y).real() + field.p(site, _y).real() +  //
                               field.n(site, _z).real() + field.p(site, _z).real() +  //
                               cosh(p.mu) * PhiNt.real() + cosh(p.mu) * PhiPt.real()) -
                 Phi.imag() * (field.n(site, _x).imag() + field.p(site, _x).imag() +  //
                               field.n(site, _y).imag() + field.p(site, _y).imag() +  //
                               field.n(site, _z).imag() + field.p(site, _z).imag() +  //
                               cosh(p.mu) * PhiNt.imag() + cosh(p.mu) * PhiPt.imag());

    TImpl Imag =
        Phi.real() * PhiNt.imag() - Phi.imag() * PhiNt.real() + PhiPt.real() * Phi.imag() - PhiPt.imag() * Phi.real();

    return {Real, Imag};
}

template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::compute_dS_loc(Lattice<field_type>& field, const field_type& dphi,
                                                       size_type site) -> action_type {
    field_type PhiOld = field[site];
    field_type PhiNew = PhiOld + dphi;
    field_type PhiNt = field.n(site, _t);
    field_type PhiPt = field.p(site, _t);

    TImpl Phi2Old = dot(PhiOld);
    TImpl Phi2New = dot(PhiNew);
    TImpl Phi2Var = Phi2New - Phi2Old;
    TImpl Phi4Var = Phi2New * Phi2New - Phi2Old * Phi2Old;

    field_type NeighborsSum = field.n(site, _x) + field.p(site, _x) +  //
                              field.n(site, _y) + field.p(site, _y) +  //
                              field.n(site, _z) + field.p(site, _z) +  //
                              cosh(p.mu) * (PhiNt + PhiPt);

    TImpl Real = 0.5 * p.eta * Phi2Var + 0.25 * p.lambda * Phi4Var - dphi.real() * NeighborsSum.real() -
                 dphi.imag() * NeighborsSum.imag();

    TImpl Imag = (dphi.real() * PhiNt.imag() - dphi.imag() * PhiNt.real()) +
                 (PhiPt.real() * dphi.imag() - PhiPt.imag() * dphi.real());

    return {Real, Imag};
}

template <RealValue TImpl>
inline void RelativisticBoseGas<TImpl>::compute_Forces(Lattice<field_type>& field, Lattice<field_type>& forces) {
    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        field_type Phi = field[Site];
        field_type Phi2 = {Phi.real() * Phi.real(), Phi.imag() * Phi.imag()};
        field_type Phi3 = {Phi2.real() * Phi.real(), Phi2.imag() * Phi.imag()};

        field_type NeighborsSum = field.n(Site, _x) + field.p(Site, _x) +                //
                                  field.n(Site, _y) + field.p(Site, _y) +                //
                                  field.n(Site, _z) + field.p(Site, _z) +                //
                                  cosh(p.mu) * (field.n(Site, _t) + field.p(Site, _t));  //

        TImpl Real = p.eta * Phi.real() + p.lambda * (Phi3.real() + Phi.real() * Phi2.imag()) - NeighborsSum.real();
        TImpl Imag = p.eta * Phi.imag() + p.lambda * (Phi3.imag() + Phi.imag() * Phi2.real()) - NeighborsSum.imag();
        forces[Site] = {Real, Imag};
    }
}

template <RealValue TImpl>
inline void RelativisticBoseGas<TImpl>::compute_LLRForces(Lattice<field_type>& field, Lattice<field_type>& forces,
                                                          TImpl Sk, TImpl width, TImpl ak) {
    field_type Phi;
    field_type Phi2;
    field_type Phi3;
    field_type NeighborsSum;
    TImpl      ForcePhiRe;
    TImpl      ForceLLRPhiRe;
    TImpl      ForcePhiIm;
    TImpl      ForceLLRPhiIm;

    // compute current value of the imaginary action
    TImpl SIm = 0.0;
    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        SIm += Phi.real() * field.n(Site, _t).imag() - Phi.imag() * field.n(Site, _t).real();
    }
    TImpl LLRForcePref = ((SIm - Sk) / (width * width) + ak);

    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        Phi2 = {Phi.real() * Phi.real(), Phi.imag() * Phi.imag()};
        Phi3 = {Phi2.real() * Phi.real(), Phi2.imag() * Phi.imag()};
        NeighborsSum = field.n(Site, _x) + field.p(Site, _x) +                //
                       field.n(Site, _y) + field.p(Site, _y) +                //
                       field.n(Site, _z) + field.p(Site, _z) +                //
                       cosh(p.mu) * (field.n(Site, _t) + field.p(Site, _t));  //
        // Standard forces
        ForcePhiRe = p.eta * Phi.real() + p.lambda * (Phi3.real() + Phi.real() * Phi2.imag()) - NeighborsSum.real();
        ForcePhiIm = p.eta * Phi.imag() + p.lambda * (Phi3.imag() + Phi.imag() * Phi2.real()) - NeighborsSum.imag();
        // LLR Forces
        ForceLLRPhiRe = LLRForcePref * (field.n(Site, _t).imag() - field.p(Site, _t).imag());
        ForceLLRPhiIm = LLRForcePref * (field.p(Site, _t).real() - field.n(Site, _t).real());

        forces[Site] = {ForcePhiRe + ForceLLRPhiRe, ForcePhiIm + ForceLLRPhiIm};
    }
}

template <RealValue TImpl>
inline auto RelativisticBoseGas<TImpl>::Measure(Lattice<field_type>& field) -> RelativisticBoseGas::Observables {
    field_type Phi;
    TImpl      Phi2 = 0.0;
    auto       Norm = static_cast<TImpl>(field.getNsites());

    for (size_type Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        Phi2 += static_cast<TImpl>(dot(Phi));
    }

    return {static_cast<impl_type>(0.5 * Phi2 / Norm), static_cast<impl_type>(0.0)};
}

}  // namespace action

/*--------------------------------------------------------------------------------------------------
  HDF5 helper method instantiations
--------------------------------------------------------------------------------------------------*/
template <>
auto make_H5_Type<action::RelativisticBoseGas<RealF>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::RelativisticBoseGas<RealF>::Observables));
    H5Tinsert(DataTypeHid, "phi2", HOFFSET(action::RelativisticBoseGas<RealF>::Observables, phi2), H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "density", HOFFSET(action::RelativisticBoseGas<RealF>::Observables, density),
              H5T_NATIVE_FLOAT);
    return DataTypeHid;
}
template <>
auto make_H5_Type<action::RelativisticBoseGas<RealD>::Observables>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::RelativisticBoseGas<RealD>::Observables));
    H5Tinsert(DataTypeHid, "phi2", HOFFSET(action::RelativisticBoseGas<RealD>::Observables, phi2), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "density", HOFFSET(action::RelativisticBoseGas<RealD>::Observables, density),
              H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo
