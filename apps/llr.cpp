#include "reticolo.hpp"

#include <format>
#include <iostream>
#include <string>
#include <vector>

using namespace reticolo;

int main(int argc, char *argv[]) {
  std::cout << IO::pretty_welcome() << '\n';

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

  for (const auto &Volume : Volumes) {
    // std::cout << std::format("Lattice size: {} x {} x {} x {}", volume[_t],
    // volume[_x], volume[_y], volume[_z]) << std::endl;

    std::string OutPath = std::format("./{}_{}_{}_{}", Volume[_t], Volume[_x],
                                      Volume[_y], Volume[_z]);

    action::phi4 Action(1.0, 9.0, 1.0);

    LLR::controller Cont(Action);

    Cont.init(OutPath, Volume, 12, 0.0025, IO::LOG_mode::silent);

    Cont.run(50, 2000, 1000, "run", 10);
  }

  return EXIT_SUCCESS;
}
