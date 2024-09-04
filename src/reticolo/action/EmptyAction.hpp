// /******************************************************************************

//  - reticolo
//  (www.github.com/olmo-francesconi/reticolo.git)

//  - SourceFile: action/RelativisticBoseGas.hpp

//  - Author: Olmo Francesconi
//  <olmo.francesconi@glasgow.ac.uk>

//  ******************************************************************************/

// #pragma once

// #include <H5Ipublic.h>
// #include <H5Tpublic.h>

// #include <cstddef>
// #include <format>
// #include <string>

// #include "reticolo/action/action_base.hpp"
// #include "reticolo/lattice/Lattice.hpp"
// #include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
// #include "reticolo/types/core.hpp"

// namespace reticolo::action {

// /*--------------------------------------------------------------------------------------------------
//   EmptyAction Class Declaration
// --------------------------------------------------------------------------------------------------*/
// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// class EmptyAction : ActionBase<TField, TAction, dim> {
//   public:
//     /* Types and public action metadata */
//     using FieldType = TField;      // Type of the field variables
//     using ActionType = TAction;    // Return type fo the action
//     const static int Dims = dim;   // Dimensions of the action
//     const static int Stencil = 2;  // Minimum step size for multi-thread safety

//     /* Algorithm capabilities */
//     const static bool IsMetropolisCapable = false;
//     const static bool IsHmcCapable = false;
//     const static bool IsLLRCapable = false;

//     /* Action parameters */
//     struct Params {
//         double _param;
//         Params() : _param(1.0){};
//         Params(double param) : _param(param){};
//     } p;

//     /* Observables */
//     struct Observables {
//         double obs;
//     };
//     friend auto make_H5_Type<Observables>() {
//         hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(Observables));
//         H5Tinsert(DataTypeHid, "obsName", HOFFSET(Observables, obs), H5T_NATIVE_DOUBLE);
//         return DataTypeHid;
//     }

//     /* Construtors */
//     EmptyAction(double param) : p(param){};    // Parameter List
//     EmptyAction(Params params) : p(params){};  // Parameter struct

//     /* Destructor*/
//     ~EmptyAction() = default;

//     /* Sync with lattice */
//     void lattice_sync(const Lattice<FieldType, dim>& field) override;

//     /* Gloabal and local action computations */
//     auto compute_S(const Lattice<FieldType, dim>& field) -> ActionType override;
//     auto compute_S_loc(const Lattice<FieldType, dim>& field, uint site) -> ActionType override;
//     auto compute_dS_loc(const Lattice<TField, dim>& field, const TField& dphi, uint site) -> ActionType override;

//     /* HMC methods */
//     void compute_Forces(const Lattice<FieldType, dim>& field, Lattice<FieldType, dim> Forces) override;

//     /* Perform the measurements or returns updated Observable values*/
//     static auto Measure(const Lattice<FieldType, dim>& field) -> Observables;

//     /* Log stuff */
//     auto action_name() -> std::string override { return "EmptyAction"; };
//     auto action_parameters() -> std::string override {
//         std::string Res = std::format("[param : {:4.1f}]", p.lambda);
//         return Res;
//     }
// };

// /*--------------------------------------------------------------------------------------------------
//   Public methods Implementatin
// --------------------------------------------------------------------------------------------------*/
// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// inline void EmptyAction<TField, TAction, dim>::lattice_sync(const Lattice<TField, dim>& field) {}

// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// inline auto EmptyAction<TField, TAction, dim>::compute_S(const Lattice<TField, dim>& Field) -> TAction {}

// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// inline auto EmptyAction<TField, TAction, dim>::compute_S_loc(const Lattice<TField, dim>& Field, uint Site) -> TAction
// {}

// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// inline auto EmptyAction<TField, TAction, dim>::compute_dS_loc(const Lattice<TField, dim>& Field, const TField& dPhi,
//                                                               uint Site) -> TAction {}

// template <ComplexValue TField, ComplexValue TAction, size_t dim>
// inline auto EmptyAction<TField, TAction, dim>::Measure(const Lattice<TField, dim>& field) -> EmptyAction::Observables
// {}

// }  // namespace reticolo::action
