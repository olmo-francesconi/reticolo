/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/gr.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <format>
#include <string>

#include "H5cpp.h"
#include "reticolo/action/action_base.hpp"
#include "reticolo/tools/types/hfield.hpp"

namespace reticolo::action {

class WeakFieldEuclideanGR : action_base<HField, RealD, 4> {
 public:
  struct params {
    double lambda;
    double eta;
    double mu;

    // params() : lambda(1.0), eta(9.0), mu(0){};
    // params(double lambda, double eta, double mu) : lambda(lambda), eta(eta), mu(mu){};
  };

  struct observables {
    double R;

    auto dump_str() -> std::string { return std::format("{:+8e}", R); }
    [[nodiscard]] auto dump_data() const -> std::vector<double> { return std::vector<double>({R}); }
  };
  static void make_hdf5_CompType(H5::CompType& type) {
    type.insertMember("R", HOFFSET(observables, R), H5::PredType::NATIVE_DOUBLE);
  }

  [[nodiscard]] auto compute_S(const lattice<HField, 4>& field) const -> RealD override;
  [[nodiscard]] auto compute_S_loc(const lattice<HField, 4>& field, const uintvect<4>& coord) const -> RealD override;
  [[nodiscard]] auto compute_dS_loc(const lattice<HField, 4>& field, const HField& dphi, const uintvect<4>& coord) const
      -> RealD override;

  virtual void Measure(const lattice<HField, 4>& field);

  // Log action information
  auto action_name() -> std::string override = 0;        // return the action name
  auto action_parameters() -> std::string override = 0;  // prints action parameters
};
}  // namespace reticolo::action
