/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tests/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <format>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/core/tools/Hdf5Handler.hpp"
#include "reticolo/core/tools/timer.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"

using namespace reticolo;

using i_t = Indexing::size_type;

auto main(int argc, char* argv[]) -> int {
    const std::vector<i_t> Size({4, 4, 4, 4});
    Lattice<RealD>         Field(Size);
    std::cout << "Lattice object initialized\n\n";

    std::cout << "Randomizing the lattice.. ";
    std::mt19937_64                 Rng;
    std::normal_distribution<RealD> Norm(0.0, 1.0);
    Rng.seed(0);
    for (auto& Site : Field) {
        randomize(Site, 1.0, Norm, Rng);
    }
    std::cout << "done\n\n";

    std::cout << "Saving the field to out.hdf5.. ";
    std::stringstream RngState;
    RngState << Rng;
    GlobalHdf5Handler.saveLattice("./out.hdf5", "rw_test", Field, RngState);
    std::cout << "done\n\n";

    std::cout << "Reading in the file.. ";
    GlobalHdf5Handler.readLattice("./out.hdf5", "rw_test", Field, RngState);
    std::cout << "done\n\n";

    std::cout << "Saving the field to out2.hdf5.. ";
    RngState << Rng;
    GlobalHdf5Handler.saveLattice("./out2.hdf5", "rw_test", Field, RngState);
    std::cout << "done\n\n";
}
