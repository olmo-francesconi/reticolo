/******************************************************************************

 - reticolo
 (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/phi4.hpp

 - Author: Olmo Francesconi
 <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <algorithm>
#include <format>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/action/action_base.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::action {

template <ComplexValue TField, ComplexValue TAction>
class BoseGas : ActionBase<TField, TAction, 4> {
  public:
    using FieldType = TField;
    using ActionType = TAction;
    const static int Dims = 4;

    struct Params {
        double lambda;
        double eta;
        double mu;

        Params() : lambda(1.0), eta(9.0), mu(0){};
        Params(double lambda, double eta, double chem_mu) : lambda(lambda), eta(eta), mu(chem_mu){};
    } p;

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

    BoseGas() = default;
    BoseGas(double lambda, double eta, double chem_mu) : p(lambda, eta, chem_mu){};
    BoseGas(Params par) : p(par){};
    ~BoseGas(){};

    auto compute_S(const Lattice<FieldType, 4>& field) const -> ActionType override;
    auto compute_S_loc(const Lattice<FieldType, 4>& field, const uintvect<4>& coord) const -> ActionType override;
    auto compute_dS_loc(const Lattice<TField, 4>& field, const TField& dphi, const uintvect<4>& coord) const
        -> ActionType override;

    static auto Measure(const Lattice<FieldType, 4>& field) -> Observables;

    auto action_name() -> std::string override { return "Relativistic Bose Gas (phi^4)"; };
    auto action_parameters() -> std::string override {
        std::stringstream Res;
        Res << "[ lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta)
            << ", mu : " << std::format("{:4.1f}", p.mu) << " ]";
        return Res.str();
    }
};

template <ComplexValue TField, ComplexValue TAction>
inline auto BoseGas<TField, TAction>::compute_S(const Lattice<TField, 4>& Field) const -> TAction {
    double  Real = 0.0;
    double  Imag = 0.0;
    TAction Phi;

    uintvect<4> Sizes = Field.getSizes();
    uintvect<4> Coord;
    std::fill(Coord.begin(), Coord.end(), 0);
    for (uint Site = 0; Site < Field.getNsites(); advance_coord(Sizes, Coord), Site++) {
        Phi = Field[Coord];

        double Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

        Real +=
            0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
            (Phi.real() * Field[next(Field, Coord, _x)].real() + Phi.imag() * Field[next(Field, Coord, _x)].imag()) -
            (Phi.real() * Field[next(Field, Coord, _y)].real() + Phi.imag() * Field[next(Field, Coord, _y)].imag()) -
            (Phi.real() * Field[next(Field, Coord, _z)].real() + Phi.imag() * Field[next(Field, Coord, _z)].imag()) -
            cosh(p.mu) *
                (Phi.real() * Field[next(Field, Coord, _t)].real() + Phi.imag() * Field[next(Field, Coord, _t)].imag());

        Imag += sinh(p.mu) *
                (Phi.real() * Field[next(Field, Coord, _t)].imag() - Phi.imag() * Field[next(Field, Coord, _t)].real());
    }

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto BoseGas<TField, TAction>::compute_S_loc(const Lattice<TField, 4>& Field, const uintvect<4>& Coord) const
    -> TAction {
    TAction Phi = Field[Coord];
    TAction PhiNt = Field[next(Field, Coord, _t)];
    TAction PhiPt = Field[prev(Field, Coord, _t)];

    double Phi2 = Phi.real() * Phi.real() + Phi.imag() * Phi.imag();

    double Real = 0.5 * p.eta * Phi2 + 0.25 * p.lambda * Phi2 * Phi2 -
                  Phi.real() * (Field[next(Field, Coord, _x)].real() + Field[prev(Field, Coord, _x)].real() +
                                Field[next(Field, Coord, _y)].real() + Field[prev(Field, Coord, _y)].real() +
                                Field[next(Field, Coord, _z)].real() + Field[prev(Field, Coord, _z)].real() +
                                cosh(p.mu) * PhiNt.real() + cosh(p.mu) * PhiPt.real()) -
                  Phi.imag() * (Field[next(Field, Coord, _x)].imag() + Field[prev(Field, Coord, _x)].imag() +
                                Field[next(Field, Coord, _y)].imag() + Field[prev(Field, Coord, _y)].imag() +
                                Field[next(Field, Coord, _z)].imag() + Field[prev(Field, Coord, _z)].imag() +
                                cosh(p.mu) * PhiNt.imag() + cosh(p.mu) * PhiPt.imag());

    double Imag = sinh(p.mu) * (Phi.real() * PhiNt.imag() - Phi.imag() * PhiNt.real()) +
                  sinh(p.mu) * (PhiPt.real() * Phi.imag() - PhiPt.imag() * Phi.real());

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto BoseGas<TField, TAction>::compute_dS_loc(const Lattice<TField, 4>& Field, const TField& dPhi,
                                                     const uintvect<4>& Coord) const -> TAction {
    TField PhiOld = Field[Coord];
    TField PhiNew = PhiOld + dPhi;
    TField PhiNt = Field[next(Field, Coord, _t)];
    TField PhiPt = Field[prev(Field, Coord, _t)];

    double Phi2Old = PhiOld.real() * PhiOld.real() + PhiOld.imag() * PhiOld.imag();
    double Phi2New = PhiNew.real() * PhiNew.real() + PhiNew.imag() * PhiNew.imag();

    double Phi2Var = Phi2New - Phi2Old;
    double Phi4Var = Phi2New * Phi2New - Phi2Old * Phi2Old;

    TField NeighborsSum = Field[next(Field, Coord, _x)] + Field[prev(Field, Coord, _x)] +
                          Field[next(Field, Coord, _y)] + Field[prev(Field, Coord, _y)] +
                          Field[next(Field, Coord, _z)] + Field[prev(Field, Coord, _z)] + cosh(p.mu) * (PhiNt + PhiPt);

    double Real = 0.5 * p.eta * Phi2Var + 0.25 * p.lambda * Phi4Var - dPhi.real() * NeighborsSum.real() -
                  dPhi.imag() * NeighborsSum.imag();

    double Imag = (dPhi.real() * PhiNt.imag() - dPhi.imag() * PhiNt.real()) +
                  (PhiPt.real() * dPhi.imag() - PhiPt.imag() * dPhi.real());

    return {Real, Imag};
};

template <ComplexValue TField, ComplexValue TAction>
inline auto BoseGas<TField, TAction>::Measure(const Lattice<TField, 4>& field) -> BoseGas::Observables {
    TField Phi;
    double Phi2 = 0.0;

    uintvect<4> Sizes = field.getSizes();
    uintvect<4> Coord;
    std::fill(Coord.begin(), Coord.end(), 0);

    for (uint Site = 0; Site < field.getNsites(); advance_coord(Sizes, Coord), Site++) {
        Phi = field[Coord];
        Phi2 += Phi.real() * Phi.real() + Phi.imag() * Phi.imag();
    }

    return Observables(0.5 * Phi2 / (double)field.getNsites(), 0.0);
};

}  // namespace reticolo::action
