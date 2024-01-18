/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/concepts.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "tools/types/basic.hpp"

#include <random>

namespace reticolo
{
    // Template concepts
    template <typename T>
    concept RealValue = std::same_as<T, RealD> || std::same_as<T, RealF>;

    template <typename T>
    concept ComplexValue = std::same_as<T, ComplexD> || std::same_as<T, ComplexF>;

    template <typename T>
    concept RNGDist = std::same_as<T, std::normal_distribution<>> || std::same_as<T, std::uniform_real_distribution<>>;

    template <typename T>
    concept RNGGen = std::same_as<T, std::mt19937> || std::same_as<T, std::mt19937_64> || std::same_as<T, std::ranlux24> || std::same_as<T, std::ranlux48>;

} // namespace reticolo