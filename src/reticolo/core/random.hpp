/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/basic.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <concepts>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

namespace reticolo {
/* Randomize real-valued variable */
template <RealValue T, RNGDist TDist, RNGGen TGen>
inline void randomize(T& Val, const T& Scale, TDist& Dist, TGen& Gen)
    requires std::same_as<T, typename TDist::result_type>
{
    Val = Scale * Dist(Gen);
}

/* Randomize complex-valued variable */
template <ComplexValue T, RNGDist TDist, RNGGen TGen>
inline void randomize(T& Val, const typename T::value_type& Scale, TDist& Dist, TGen& Gen)
    requires std::same_as<typename T::value_type, typename TDist::result_type>
{
    Val = Scale * T(Dist(Gen), Dist(Gen));
}

/* Randomize Hfield variable (GR metric perturbation) */
template <RealValue T, RNGDist TDist, RNGGen TGen>
inline void randomize(HField<T>& Val, T Scale, TDist& Dist, TGen& Gen)
    requires std::same_as<T, typename TDist::result_type>
{
    for (uint Comp = 0; Comp < HField<T>::NumComp; Comp++) {
        Val[Comp] = Scale * Dist(Gen);
    }
}

}  // namespace reticolo
