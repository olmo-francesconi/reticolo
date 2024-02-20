/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/basic.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "reticolo/tools/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/tools/types/hfield.hpp"

namespace reticolo {
/* Randomize real-valued variable */
template <RealValue ValType, RNGDist DistType, RNGGen GenType>
inline void randomize(ValType& Val, const ValType& Scale, DistType& Dist, GenType& Gen)
    requires std::same_as<ValType, typename DistType::result_type>
{
    Val = Scale * Dist(Gen);
}

/* Randomize complex-valued variable */
template <ComplexValue ValType, RNGDist DistType, RNGGen GenType>
inline void randomize(ValType& Val, const typename ValType::value_type& Scale, DistType& Dist, GenType& Gen)
    requires std::same_as<typename ValType::value_type, typename DistType::result_type>
{
    Val = Scale * ValType(Dist(Gen), Dist(Gen));
}

/* Randomize Hfield variable (GR metric perturbation) */
template <RealValue T, RNGDist DistType, RNGGen GenType>
inline void randomize(HField<T>& Val, T& Scale, DistType& Dist, GenType& Gen)
    requires std::same_as<T, typename DistType::result_type>
{
    for (int Comp = 0; Comp < HfieldNumComp; Comp++) {
        Val[Comp] = Scale * Dist(Gen);
    }
}

}  // namespace reticolo