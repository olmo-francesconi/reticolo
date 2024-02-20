/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/gr.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <format>
#include <string>
#include <vector>

#include "H5cpp.h"
#include "reticolo/action/action_base.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/tools/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/tools/types/core.hpp"
#include "reticolo/tools/types/hfield.hpp"

namespace reticolo::action {

template <RealValue TField, RealValue TAction>
class WeakFieldEuclideanGR : ActionBase<HField<TField>, TAction, 4> {
  public:
    struct Params {
        double lambda;

        // params() : lambda(1.0), eta(9.0), mu(0){};
        // params(double lambda, double eta, double mu) : lambda(lambda), eta(eta), mu(mu){};
    };

    struct Observables {
        double R;

        [[nodiscard]] auto dump_str() -> std::string { return std::format("{:+8e}", R); }
        [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({R}); }
    };
    static void make_hdf5_CompType(H5::CompType& type) {
        type.insertMember("R", HOFFSET(Observables, R), H5::PredType::NATIVE_DOUBLE);
    }

    [[nodiscard]] auto compute_S(const Lattice<HField<TField>, 4>& field) const -> TAction override;
    [[nodiscard]] auto compute_S_loc(const Lattice<HField<TField>, 4>& field, const uintvect<4>& coord) const
        -> TAction override;
    [[nodiscard]] auto compute_dS_loc(const Lattice<HField<TField>, 4>& field, const HField<TField>& dphi,
                                      const uintvect<4>& coord) const -> TAction override;

    virtual auto Measure(const Lattice<HField<TField>, 4>& field) -> WeakFieldEuclideanGR::Observables;

    // Log action information
    auto action_name() -> std::string override = 0;        // return the action name
    auto action_parameters() -> std::string override = 0;  // prints action parameters
};
}  // namespace reticolo::action
