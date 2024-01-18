/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/basic.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "tools/types/basic.hpp"
#include "tools/types/hfield.hpp"
#include "tools/types/concepts.hpp"

namespace reticolo
{
    template <RealValue ValType, RNGDist DistType, RNGGen GenType>
    ValType random(DistType &dist, GenType &gen)
    {
        return dist(gen);
    }

    template <ComplexValue ValType, RNGDist DistType, RNGGen GenType>
    ValType random(DistType &dist, GenType &gen)
    {
        return ValType(dist(gen), dist(gen));
    }

    template <typename ValType = HField, RNGDist DistType, RNGGen GenType>
    ValType random(DistType &dist, GenType &gen)
    {
        HField res;
        for (int i = 0; i < 10; i++)
            res[i] = dist(gen);
        return res;
    }

} // namespace reticolo