#include "reticolo.hpp"

#include <iostream>
#include <string>
#include <format>
#include <vector>
#include <complex>

using namespace reticolo;

int main(int argc, char *argv[])
{
    std::cout << IO::pretty_welcome() << std::endl;

    // // std::cout << std::is_base_of<action::action_base<RealD, ComplexD>, action::phi4>::value << std::endl;

    std::vector<vect<4>> volumes = {
        {4, 4, 4, 4},
        // {6, 6, 6, 6},
        // {8, 8, 8, 8},
        // {10, 10, 10, 10},
        // {12, 12, 12, 12},
        // {14, 14, 14, 14},
        // {16, 16, 16, 16}
    };

    for (const auto &volume : volumes)
    {
        std::cout << std::format("Lattice size: {} x {} x {} x {}", volume[_t], volume[_x], volume[_y], volume[_z]) << std::endl;

        std::string out_path = std::format("./{}_{}_{}_{}", volume[_t], volume[_x], volume[_y], volume[_z]);

        LLR::controller<action::phi4, ComplexD, ComplexD, 4> cont;

        action::phi4<ComplexD, ComplexD>::params par(1.0, 9.0, 1.0);

        cont.init(out_path, volume, 1, par);

        cont.run(1, 0, 1);
    }

    return EXIT_SUCCESS;
}