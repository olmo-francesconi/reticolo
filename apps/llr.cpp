#include <cstdlib>
#include <format>
#include <string>
#include <vector>

#include "reticolo/action/phi4.hpp"
#include "reticolo/reticolo.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    std::vector<uintvect<4>> Volumes = {
        {4, 4, 4, 4},
        // {6, 6, 6, 6},
        // {8, 8, 8, 8},
        // {10, 10, 10, 10},
        // {12, 12, 12, 12},
        // {14, 14, 14, 14},
        // {16, 16, 16, 16},
        // {64, 64, 64, 64},
    };

    for (const auto& Volume : Volumes) {
        // std::cout << std::format("Lattice size: {} x {} x {} x {}", volume[_t],
        // volume[_x], volume[_y], volume[_z]) << std::endl;

        for (int DMuI = 0; DMuI < 10; DMuI++) {
            double Mu = 0.1 * DMuI;

            action::BoseGas<ComplexD, ComplexD> Action(1.0, 9.0, Mu);

            LLR::LLRController Cont(Action);

            std::string OutPath =
                std::format("./{}_{}_{}_{}/{:3.2}", Volume[_t], Volume[_x], Volume[_y], Volume[_z], Mu);

            Cont.init(OutPath, Volume, 1, 0.0025, LOG_mode::silent);

            Cont.run(50, 50, 1000, "run", 1);
        }
    }

    return EXIT_SUCCESS;
}
