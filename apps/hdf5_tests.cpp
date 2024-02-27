/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/LLRBoseGas.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <H5Cpp.h>

#include <cstdlib>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Vol1 = {4, 4, 4, 4};

    Lattice<ComplexD, 4> Latt1;
    Latt1.init(Vol1);
    for (uint i = 0; i < Latt1.getNsites(); i++) {
        Latt1[i] = {double(i), double(i)};
    }
    Latt1.save_Configuration("./test.h5");

    uintvect<4> Vol2 = {6, 6, 6, 6};

    Lattice<ComplexD, 4> Latt2;
    Latt2.init(Vol2);
    Latt2.read_Configuration("./test.h5");

    return EXIT_SUCCESS;
}