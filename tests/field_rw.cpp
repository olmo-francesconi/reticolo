/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tests/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/core/storage/StorageFacade.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"

using namespace reticolo;

using i_t = Indexing::size_type;

static auto make_tmp_h5_path() -> std::filesystem::path {
    static unsigned long long counter = 0;
    std::random_device        rd;
    return std::filesystem::temp_directory_path() /
           ("reticolo_field_rw_" + std::to_string(rd()) + "_" + std::to_string(counter++) + ".h5");
}

auto main(int /*unused*/, char* /*unused*/[]) -> int {
    const auto             output_file = make_tmp_h5_path();
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

    std::cout << "Saving the field to " << output_file << ".. ";
    std::stringstream RngState;
    RngState << Rng;
    storage::GlobalStorage.save_lattice(output_file, storage::schema::lattice::field("rw_test"), FieldOut, RngState);
    std::cout << "done\n";
    std::cout << "Field sum: " << std::reduce(FieldOut.begin(), FieldOut.end()) << "\n";
    std::cout << "Next randon munber: " << Rng() << "\n\n";

    const std::vector<i_t> SizeIn({4, 4, 4, 4});
    Lattice<ComplexD>      FieldIn(SizeIn);
    std::cout << "Reading in the file.. ";
    try {
        storage::GlobalStorage.load_lattice(output_file, storage::schema::lattice::field("rw_test"), FieldIn, RngState);
    } catch (std::exception& e) {
        std::cout << "\n" << e.what() << "\n";
        std::filesystem::remove(output_file);
        return EXIT_FAILURE;
    }
    std::cout << "done\n";
    std::cout << "Field sum: " << std::reduce(FieldIn.begin(), FieldIn.end()) << "\n";
    RngState >> Rng;
    std::cout << "Next randon munber: " << Rng() << "\n\n";

    std::cout << "All done!\n";
    std::filesystem::remove(output_file);

    return EXIT_SUCCESS;
}
