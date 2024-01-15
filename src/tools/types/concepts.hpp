#pragma once

#include "tools/types/basic.hpp"

namespace reticolo
{
    // Template concepts
    template <typename T>
    concept RealValue = std::same_as<T, RealD> || std::same_as<T, RealF>;

    template <typename T>
    concept ComplexValue = std::same_as<T, ComplexD> || std::same_as<T, ComplexF>;
} // namespace reticolo