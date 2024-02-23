/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/RelativisticBoseGas.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <format>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::action {

/*--------------------------------------------------------------------------------------------------
  RelativisticBoseGas Class Declaration
--------------------------------------------------------------------------------------------------*/
template <ComplexValue TField, ComplexValue TAction>
class RelativisticBoseGas {
  public:
    /* Types and public action metadata */
    using FieldType = TField;      // Type of the field variables
    using ActionType = TAction;    // Return type fo the action
    const static int Dims = 4;     // Dimensions of the action
    const static int Stencil = 2;  // Minimum step size for multi-thread safety

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
    RelativisticBoseGas() = default;                             // Default
    RelativisticBoseGas(RelativisticBoseGas&& other) = default;  // Move

    /* Initializer Construtors */
    RelativisticBoseGas(double lambda, double eta, double chem_mu) : p(lambda, eta, chem_mu){};  // Parameter List
    RelativisticBoseGas(Params par) : p(par){};                                                  // Parameter struct

    /* Destructor*/
    ~RelativisticBoseGas() = default;

    /* Sync with lattice */
    void lattice_sync(const Lattice<FieldType, 4>& field){};  // Nothing to do here

    /* Gloabal and local action computations */
    auto compute_S(const Lattice<FieldType, 4>& field) const -> ActionType;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, uint site) const -> ActionType;
    auto compute_dS_loc(const Lattice<TField, 4>& field, const TField& dphi, uint site) const -> ActionType;

    /* Perform the measurements or returns updated Observable values*/
    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables;

    /* Log stuff*/
    auto action_name() -> std::string { return "Relativistic Bose Gas (phi^4)"; };
    auto action_parameters() -> std::string {
        std::stringstream Res;
        Res << "[ lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta)
            << ", mu : " << std::format("{:4.1f}", p.mu) << " ]";
        return Res.str();
    }
};

/*--------------------------------------------------------------------------------------------------
  Public methods Implementatin
--------------------------------------------------------------------------------------------------*/
template <ComplexValue TField, ComplexValue TAction>
inline auto RelativisticBoseGas<TField, TAction>::compute_S(const Lattice<TField, 4>& Field) const -> TAction {
    double  Real = 0.0;
    double  Imag = 0.0;
    TAction Phi;

    for (uint Site = 0; Site < Field.getNsites(); Site++) {
        Phi = Field[Site];

        double Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

        Real += 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                (Phi.real() * Field.next(Site, _x).real() + Phi.imag() * Field.next(Site, _x).imag()) -
                (Phi.real() * Field.next(Site, _y).real() + Phi.imag() * Field.next(Site, _y).imag()) -
                (Phi.real() * Field.next(Site, _z).real() + Phi.imag() * Field.next(Site, _z).imag()) -
                cosh(p.mu) * (Phi.real() * Field.next(Site, _t).real() + Phi.imag() * Field.next(Site, _t).imag());

        Imag += sinh(p.mu) * (Phi.real() * Field.next(Site, _t).imag() - Phi.imag() * Field.next(Site, _t).real());
    }

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto RelativisticBoseGas<TField, TAction>::compute_S_loc(const Lattice<TField, 4>& Field, uint Site) const
    -> TAction {
    TAction Phi = Field[Site];
    TAction PhiNt = Field.next(Site, _t);
    TAction PhiPt = Field.prev(Site, _t);

    double Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

    double Real = 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                  Phi.real() * (Field.next(Site, _x).real() + Field.prev(Site, _x).real() +  //
                                Field.next(Site, _y).real() + Field.prev(Site, _y).real() +  //
                                Field.next(Site, _z).real() + Field.prev(Site, _z).real() +  //
                                cosh(p.mu) * PhiNt.real() + cosh(p.mu) * PhiPt.real()) -
                  Phi.imag() * (Field.next(Site, _x).imag() + Field.prev(Site, _x).imag() +  //
                                Field.next(Site, _y).imag() + Field.prev(Site, _y).imag() +  //
                                Field.next(Site, _z).imag() + Field.prev(Site, _z).imag() +  //
                                cosh(p.mu) * PhiNt.imag() + cosh(p.mu) * PhiPt.imag());

    double Imag = sinh(p.mu) * (Phi.real() * PhiNt.imag() - Phi.imag() * PhiNt.real()) +
                  sinh(p.mu) * (PhiPt.real() * Phi.imag() - PhiPt.imag() * Phi.real());

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto RelativisticBoseGas<TField, TAction>::compute_dS_loc(const Lattice<TField, 4>& Field, const TField& dPhi,
                                                                 uint Site) const -> TAction {
    TField PhiOld = Field[Site];
    TField PhiNew = PhiOld + dPhi;
    TField PhiNt = Field.next(Site, _t);
    TField PhiPt = Field.prev(Site, _t);

    double Phi2Old = PhiOld.real() * PhiOld.real() + PhiOld.imag() * PhiOld.imag();
    double Phi2New = PhiNew.real() * PhiNew.real() + PhiNew.imag() * PhiNew.imag();

    double Phi2Var = Phi2New - Phi2Old;
    double Phi4Var = Phi2New * Phi2New - Phi2Old * Phi2Old;

    TField NeighborsSum = Field.next(Site, _x) + Field.prev(Site, _x) +  //
                          Field.next(Site, _y) + Field.prev(Site, _y) +  //
                          Field.next(Site, _z) + Field.prev(Site, _z) +  //
                          cosh(p.mu) * (PhiNt + PhiPt);

    double Real = 0.5 * p.eta * Phi2Var + 0.25 * p.lambda * Phi4Var - dPhi.real() * NeighborsSum.real() -
                  dPhi.imag() * NeighborsSum.imag();

    double Imag = (dPhi.real() * PhiNt.imag() - dPhi.imag() * PhiNt.real()) +
                  (PhiPt.real() * dPhi.imag() - PhiPt.imag() * dPhi.real());

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto RelativisticBoseGas<TField, TAction>::Measure(const Lattice<TField, 4>& field)
    -> RelativisticBoseGas::Observables {
    TField Phi;
    double Phi2 = 0.0;

    for (uint Site = 0; Site < field.getNsites(); Site++) {
        Phi = field[Site];
        Phi2 += Phi.real() * Phi.real() + Phi.imag() * Phi.imag();
    }

    return {0.5 * Phi2 / double(field.getNsites()), 0.0};
};

}  // namespace reticolo::action
