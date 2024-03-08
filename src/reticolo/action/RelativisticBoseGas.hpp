/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cmath>
#include <format>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/action/action_base.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  RelativisticBoseGas Class Declaration
--------------------------------------------------------------------------------------------------*/
class RelativisticBoseGas : ActionBase<ComplexD, ComplexD, 4> {
  public:
    /* Types and public action metadata */
    using FieldType = ComplexD;    // Type of the field variables
    using ActionType = ComplexD;   // Return type fo the action
    const static int Dims = 4;     // Dimensions of the action
    const static int Stencil = 2;  // Minimum step size for multi-thread safety

    /* Algorithm capabilities */
    const static bool IsMetropolisCapable = true;
    const static bool IsHmcCapable = true;
    const static bool IsLLRCapable = false;

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
        double             phi2;
        double             density;
        [[nodiscard]] auto dump_str() -> std::string { return std::format("{:+8e},{:+8e}", phi2, density); }
        [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({phi2, density}); }
    };
    static auto make_obs_hdf5_CompType() -> H5::CompType {
        H5::CompType Type(sizeof(Observables));
        Type.insertMember("phi2", HOFFSET(Observables, phi2), H5::PredType::NATIVE_DOUBLE);
        Type.insertMember("density", HOFFSET(Observables, density), H5::PredType::NATIVE_DOUBLE);
        return Type;
    }

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

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables;

    /* Log stuff */
    auto action_name() -> std::string override { return "Relativistic Bose Gas (phi^4)"; };
    auto action_parameters() -> std::string override {
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

        Imag += sinh(p.mu) * (Phi.real() * field.next(Site, _t).imag() - Phi.imag() * field.next(Site, _t).real());
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

    RealD Imag = sinh(p.mu) * (Phi.real() * PhiNt.imag() - Phi.imag() * PhiNt.real()) +
                 sinh(p.mu) * (PhiPt.real() * Phi.imag() - PhiPt.imag() * Phi.real());

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