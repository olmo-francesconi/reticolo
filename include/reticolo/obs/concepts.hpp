#pragma once

#include <reticolo/core/lattice.hpp>

#include <concepts>

namespace reticolo::obs {

// An observer is anything callable on a `Lattice<T> const&` returning a
// scalar. The catalog ships free function templates that satisfy this;
// apps that want to roll their own ad-hoc observer can either pass a lambda
// straight into a Series<double>::append or use this concept to constrain
// generic measurement helpers.
template <class O, class T>
concept Observer = requires(O const& o, Lattice<T> const& l) {
    { o(l) } -> std::convertible_to<double>;
};

}  // namespace reticolo::obs
