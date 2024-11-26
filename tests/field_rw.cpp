/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tests/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/core/tools/Hdf5Handler.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"

using namespace reticolo;

using i_t = Indexing::size_type;

auto main(int argc, char* argv[]) -> int {
    const std::vector<i_t> SizeOut({4, 4, 4, 4});
    Lattice<ComplexD>      FieldOut(SizeOut);
    std::cout << "Lattice object initialized\n\n";

    std::cout << "Randomizing the lattice.. ";
    std::mt19937_64                 Rng;
    std::normal_distribution<RealD> Norm(0.0, 1.0);
    Rng.seed(0);
    for (auto& Site : FieldOut) {
        randomize(Site, 1.0, Norm, Rng);
    }
    std::cout << "done\n\n";

    std::cout << "Saving the field to out.hdf5.. ";
    std::stringstream RngState;
    RngState << Rng;
    GlobalHdf5Handler.saveLattice("./out.hdf5", "rw_test", FieldOut, RngState);
    std::cout << "done\n";
    std::cout << "Field sum: " << std::reduce(FieldOut.begin(), FieldOut.end()) << "\n";
    std::cout << "Next randon munber: " << Rng() << "\n\n";

    const std::vector<i_t> SizeIn({4, 4, 4, 4});
    Lattice<ComplexD>      FieldIn(SizeIn);
    std::cout << "Reading in the file.. ";
    try {
        GlobalHdf5Handler.readLattice("./out.hdf5", "rw_test", FieldIn, RngState);
    } catch (std::exception& e) {
        std::cout << "\n" << e.what() << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "done\n";
    std::cout << "Field sum: " << std::reduce(FieldIn.begin(), FieldIn.end()) << "\n";
    RngState >> Rng;
    std::cout << "Next randon munber: " << Rng() << "\n\n";

    std::cout << "All done!\n";

    return EXIT_SUCCESS;
}
