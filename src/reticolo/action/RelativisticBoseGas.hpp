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

#include <cmath>
#include <format>
#include <sstream>
#include <string>

#include "reticolo/action/action_base.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  RelativisticBoseGas Class Declaration
--------------------------------------------------------------------------------------------------*/
class RelativisticBoseGas : public ActionBase<ComplexD, ComplexD, 4> {
  public:
    /* Types and public action metadata */
    using FieldType = ComplexD;    // Type of the field variables
    using ActionType = ComplexD;   // Return type fo the action
    const static int Dims = 4;     // Dimensions of the action
    const static int Stencil = 2;  // Minimum step size for multi-thread safety

    /* Algorithm capabilities */
    const static bool IsMetropolisCapable = true;
    const static bool IsHmcCapable = true;
    const static bool IsLLRCapable = true;

    /* Action parameters */
    struct Params {
        double lambda;
        double eta;
        double mu;
        Params() : lambda(1.0), eta(9.0), mu(0){};
        Params(double lambda, double eta, double chem_mu) : lambda(lambda), eta(eta), mu(chem_mu){};
    } p;

    /* Observables */
    struct Observables {
        double phi2;
        double density;
        void   reset() { phi2 = 0.0, density = 0.0; };
        auto   operator+=(const Observables& rhs) -> Observables {
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
    friend auto make_H5_Type<Observables>();
    //  {
    //     hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(Observables));
    //     H5Tinsert(DataTypeHid, "phi2", HOFFSET(Observables, phi2), H5T_NATIVE_DOUBLE);
    //     H5Tinsert(DataTypeHid, "density", HOFFSET(Observables, density), H5T_NATIVE_DOUBLE);
    //     return DataTypeHid;
    // }

    /* Constructors */
    RelativisticBoseGas(Lattice<FieldType, 4>& field, double lambda, double eta, double chem_mu)
        : p(lambda, eta, chem_mu) {
        lattice_sync(field);
    };
    RelativisticBoseGas(Lattice<FieldType, 4>& field, Params par) : p(par) { lattice_sync(field); };

    /* Sync with lattice */
    void lattice_sync(const Lattice<FieldType, 4>& field) override;  // Nothing to do here

    /* Gloabal and local action computations */
    auto compute_S(const Lattice<FieldType, 4>& field) -> ActionType override;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, int site) -> ActionType override;
    auto compute_dS_loc(const Lattice<FieldType, 4>& field, const FieldType& dphi, int site) -> ActionType override;

    /* HMC methods */
    void compute_Forces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& forces) override;
    void compute_LLRForces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& forces, double Sk, double width,
                           double ak) const;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables;

    /* Log stuff */
    auto name() -> std::string override { return "Relativistic Bose Gas"; };
    auto name_short() -> std::string override { return "Phi^4"; };

    auto parameters() -> std::string override {
        std::stringstream Res;
        Res << "[ lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta)
            << ", mu : " << std::format("{:4.1f}", p.mu) << " ]";
        return Res.str();
    }
};

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/

inline void RelativisticBoseGas::lattice_sync(const Lattice<FieldType, 4>& field) {
    // Maybe do some checks here
}

inline auto RelativisticBoseGas::compute_S(const Lattice<FieldType, 4>& field) -> ActionType {
    RealD    Real = 0.0;
    RealD    Imag = 0.0;
    ComplexD Phi;
    RealD    Phi2;

    for (int Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];

        Phi2 = dot(Phi);

        Real += 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                (Phi.real() * field.next(Site, _x).real() + Phi.imag() * field.next(Site, _x).imag()) -
                (Phi.real() * field.next(Site, _y).real() + Phi.imag() * field.next(Site, _y).imag()) -
                (Phi.real() * field.next(Site, _z).real() + Phi.imag() * field.next(Site, _z).imag()) -
                cosh(p.mu) * (Phi.real() * field.next(Site, _t).real() + Phi.imag() * field.next(Site, _t).imag());

        Imag += Phi.real() * field.next(Site, _t).imag() - Phi.imag() * field.next(Site, _t).real();
    }

    return {Real, Imag};
}

inline auto RelativisticBoseGas::compute_S_loc(const Lattice<FieldType, 4>& field, int site) -> ActionType {
    ComplexD Phi = field[site];
    ComplexD PhiNt = field.next(site, _t);
    ComplexD PhiPt = field.prev(site, _t);

    RealD Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

    RealD Real = 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                 Phi.real() * (field.next(site, _x).real() + field.prev(site, _x).real() +  //
                               field.next(site, _y).real() + field.prev(site, _y).real() +  //
                               field.next(site, _z).real() + field.prev(site, _z).real() +  //
                               cosh(p.mu) * PhiNt.real() + cosh(p.mu) * PhiPt.real()) -
                 Phi.imag() * (field.next(site, _x).imag() + field.prev(site, _x).imag() +  //
                               field.next(site, _y).imag() + field.prev(site, _y).imag() +  //
                               field.next(site, _z).imag() + field.prev(site, _z).imag() +  //
                               cosh(p.mu) * PhiNt.imag() + cosh(p.mu) * PhiPt.imag());

    RealD Imag =
        Phi.real() * PhiNt.imag() - Phi.imag() * PhiNt.real() + PhiPt.real() * Phi.imag() - PhiPt.imag() * Phi.real();

    return {Real, Imag};
}

inline auto RelativisticBoseGas::compute_dS_loc(const Lattice<FieldType, 4>& field, const FieldType& dphi, int site)
    -> ActionType {
    FieldType PhiOld = field[site];
    FieldType PhiNew = PhiOld + dphi;
    FieldType PhiNt = field.next(site, _t);
    FieldType PhiPt = field.prev(site, _t);

    RealD Phi2Old = dot(PhiOld);
    RealD Phi2New = dot(PhiNew);
    RealD Phi2Var = Phi2New - Phi2Old;
    RealD Phi4Var = Phi2New * Phi2New - Phi2Old * Phi2Old;

    FieldType NeighborsSum = field.next(site, _x) + field.prev(site, _x) +  //
                             field.next(site, _y) + field.prev(site, _y) +  //
                             field.next(site, _z) + field.prev(site, _z) +  //
                             cosh(p.mu) * (PhiNt + PhiPt);

    RealD Real = 0.5 * p.eta * Phi2Var + 0.25 * p.lambda * Phi4Var - dphi.real() * NeighborsSum.real() -
                 dphi.imag() * NeighborsSum.imag();

    RealD Imag = (dphi.real() * PhiNt.imag() - dphi.imag() * PhiNt.real()) +
                 (PhiPt.real() * dphi.imag() - PhiPt.imag() * dphi.real());

    return {Real, Imag};
}

inline void RelativisticBoseGas::compute_Forces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& forces) {
    for (int Site = 0; Site < field.getNsites(); Site++) {
        FieldType Phi = field[Site];
        FieldType Phi2 = {Phi.real() * Phi.real(), Phi.imag() * Phi.imag()};
        FieldType Phi3 = {Phi2.real() * Phi.real(), Phi2.imag() * Phi.imag()};

        ComplexD NeighborsSum = field.next(Site, _x) + field.prev(Site, _x) +                //
                                field.next(Site, _y) + field.prev(Site, _y) +                //
                                field.next(Site, _z) + field.prev(Site, _z) +                //
                                cosh(p.mu) * (field.next(Site, _t) + field.prev(Site, _t));  //

        RealD Real = p.eta * Phi.real() + p.lambda * (Phi3.real() + Phi.real() * Phi2.imag()) - NeighborsSum.real();
        RealD Imag = p.eta * Phi.imag() + p.lambda * (Phi3.imag() + Phi.imag() * Phi2.real()) - NeighborsSum.imag();
        forces[Site] = {Real, Imag};
    }
}

inline void RelativisticBoseGas::compute_LLRForces(const Lattice<FieldType, 4>& field, Lattice<FieldType, 4>& forces,
                                                   double Sk, double width, double ak) const {
    FieldType Phi;
    FieldType Phi2;
    FieldType Phi3;
    ComplexD  NeighborsSum;
    RealD     ForcePhiRe;
    RealD     ForceLLRPhiRe;
    RealD     ForcePhiIm;
    RealD     ForceLLRPhiIm;

    // compute current value of the imaginary action
    RealD SIm = 0.0;
    for (int Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        SIm += Phi.real() * field.next(Site, _t).imag() - Phi.imag() * field.next(Site, _t).real();
    }
    RealD LLRForcePref = ((SIm - Sk) / (width * width) + ak);

    for (int Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        Phi2 = {Phi.real() * Phi.real(), Phi.imag() * Phi.imag()};
        Phi3 = {Phi2.real() * Phi.real(), Phi2.imag() * Phi.imag()};
        NeighborsSum = field.next(Site, _x) + field.prev(Site, _x) +                //
                       field.next(Site, _y) + field.prev(Site, _y) +                //
                       field.next(Site, _z) + field.prev(Site, _z) +                //
                       cosh(p.mu) * (field.next(Site, _t) + field.prev(Site, _t));  //
        // Standard forces
        ForcePhiRe = p.eta * Phi.real() + p.lambda * (Phi3.real() + Phi.real() * Phi2.imag()) - NeighborsSum.real();
        ForcePhiIm = p.eta * Phi.imag() + p.lambda * (Phi3.imag() + Phi.imag() * Phi2.real()) - NeighborsSum.imag();
        // LLR Forces
        ForceLLRPhiRe = LLRForcePref * (field.next(Site, _t).imag() - field.prev(Site, _t).imag());
        ForceLLRPhiIm = LLRForcePref * (field.prev(Site, _t).real() - field.next(Site, _t).real());

        forces[Site] = {ForcePhiRe + ForceLLRPhiRe, ForcePhiIm + ForceLLRPhiIm};
    }
}

inline auto RelativisticBoseGas::Measure(const Lattice<FieldType, 4>& field) -> RelativisticBoseGas::Observables {
    FieldType Phi;
    RealD     Phi2 = 0.0;

    for (int Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        Phi2 += static_cast<RealD>(dot(Phi));
    }

    return {0.5 * Phi2 / static_cast<RealD>(field.getNsites()), 0.0};
}

}  // namespace reticolo::action

/*--------------------------------------------------------------------------------------------------
  HDF5 helper method Implementatin
--------------------------------------------------------------------------------------------------*/

namespace reticolo {
template <>
auto make_H5_Type<action::RelativisticBoseGas::Observables>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::RelativisticBoseGas::Observables));
    H5Tinsert(DataTypeHid, "phi2", HOFFSET(action::RelativisticBoseGas::Observables, phi2), H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "density", HOFFSET(action::RelativisticBoseGas::Observables, density), H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}
}  // namespace reticolo