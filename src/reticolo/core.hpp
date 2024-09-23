/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: core.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

/* types */
#include "reticolo/core/types/SO(N).hpp"    // IWYU pragma: keep
#include "reticolo/core/types/SU(N).hpp"    // IWYU pragma: keep
#include "reticolo/core/types/complex.hpp"  // IWYU pragma: keep
#include "reticolo/core/types/coord.hpp"    // IWYU pragma: keep
#include "reticolo/core/types/hfield.hpp"   // IWYU pragma: keep
#include "reticolo/core/types/real.hpp"     // IWYU pragma: keep

/* tools */
#include "reticolo/core/tools/Hdf5Handler.hpp"   // IWYU pragma: keep
#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep
#include "reticolo/core/tools/io_utils.hpp"      // IWYU pragma: keep
#include "reticolo/core/tools/logger.hpp"        // IWYU pragma: keep
#include "reticolo/core/tools/timer.hpp"         // IWYU pragma: keep

/* physical constants */
#include "reticolo/core/physics/constants.hpp"  // IWYU pragma: keep