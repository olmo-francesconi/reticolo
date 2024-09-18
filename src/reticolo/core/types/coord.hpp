#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

namespace reticolo {

// enumerator for lattice dimensions
enum { _t = 0, _x = 1, _y = 2, _z = 3 };

// enumerator for lattice dimensions
enum {
    _x0 = 0,
    _x1 = 1,
    _x2 = 2,
    _x3 = 3,
    _x4 = 4,
    _x5 = 5,
    _x6 = 6,
    _x7 = 7,
    _x8 = 8,
    _x9 = 9,
};

template <typename T>
inline auto dot(const std::vector<T>& vect) -> T {
    return std::accumulate(vect.begin(), vect.end(), 0, [](int sum, int value) { return sum + value * value; });
}

template <typename T>
inline auto dot(const std::vector<T>& vec1, const std::vector<T>& vec2) -> T {
    return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);
}

template <typename T>
inline void advance_coord(const std::vector<T>& sizes, std::vector<T>& coord) {
    assert(sizes.size() == coord.size());
    coord.back()++;
    for (std::size_t Dir = sizes.size() - 1; Dir > 0; Dir--) {
        if (coord[Dir] == sizes[Dir]) {
            coord[Dir] = 0;
            coord[Dir - 1]++;
        } else {
            return;
        }
    }
    if (coord[0] == sizes[0]) {
        coord[0] = 0;
    }
}

}  // namespace reticolo