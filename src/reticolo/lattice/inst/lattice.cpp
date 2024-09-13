/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/inst/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "reticolo/lattice/lattice.hpp"

#include "reticolo/lattice/impl/lattice.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

namespace reticolo {

template class Lattice<RealF>;
template class Lattice<RealD>;
template class Lattice<ComplexF>;
template class Lattice<ComplexD>;
template class Lattice<HField<RealF>>;
template class Lattice<HField<RealD>>;

}  // namespace reticolo