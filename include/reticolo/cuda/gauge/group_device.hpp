#pragma once

// Maps a host gauge group (math::group::SU2 / SU3) to its device matrix-ops
// traits (SU2Device / SU3Device), declared once. Each device group header
// specializes it; the generic gauge kernels (gauge_sun.cuh) and the matrix
// drift atom resolve GD = group_device<G>::type from the field's group.

namespace reticolo::cuda {

template <class G>
struct group_device;

}  // namespace reticolo::cuda
