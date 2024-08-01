/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tests/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include "reticolo/lattice/Lattice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto get_stats(std::vector<RealD> Vec) -> std::array<RealD, 2> {
    RealD Sum = std::accumulate(std::begin(Vec), std::end(Vec), 0.0);
    RealD Mean = Sum / Vec.size();
    RealD Accum = 0.0;
    std::for_each(std::begin(Vec), std::end(Vec), [&](const double val) { Accum += (val - Mean) * (val - Mean); });
    RealD Stdev = sqrt(Accum / (Vec.size() - 1));
    return {Mean, Stdev};
}

void printHistogram(const std::vector<double>& data, int binCount = 10, int maxBinLength = 50) {
    if (data.empty()) {
        std::cout << "No data to display." << std::endl;
        return;
    }

    double minValue = *std::min_element(data.begin(), data.end());
    double maxValue = *std::max_element(data.begin(), data.end());
    double range = maxValue - minValue;
    double binSize = range / binCount;

    std::vector<int> bins(binCount, 0);

    // Fill the bins
    for (const double& value : data) {
        int binIndex = std::min(static_cast<int>((value - minValue) / binSize), binCount - 1);
        bins[binIndex]++;
    }

    // Find the maximum bin count
    int maxBinCount = *std::max_element(bins.begin(), bins.end());

    // Print the histogram
    std::cout << "Histogram:" << std::endl;
    for (int i = 0; i < binCount; ++i) {
        double binStart = minValue + i * binSize;
        double binEnd = binStart + binSize;
        std::cout << std::fixed << std::setprecision(2) << "[" << binStart << ", " << binEnd << "): ";

        int barLength = static_cast<int>((static_cast<double>(bins[i]) / maxBinCount) * maxBinLength);
        for (int j = 0; j < barLength; ++j) {
            std::cout << '*';
        }
        std::cout << " (" << bins[i] << ")" << std::endl;
    }
}

auto main(int argc, char* argv[]) -> int {
    const std::vector<int> Size({2, 2, 2, 2});

    Lattice<RealD> Lat(Size);

    for (auto Site : Lat) {
        std::cout << Site << "\n";
    }
}

// auto main(int argc, char* argv[]) -> int {
//     const std::vector<int> Size({4, 4});
//     const int              Iterations = 100;

//     std::vector<RealD> Times;
//     Times.resize(Iterations);

//     {
//         Timer Timer;
//         Timer.reset();
//         RealD Tot = 0.0;

//         std::cout << "\n\n### Default Lattice ###\n";

//         Lattice<RealD> Lat(Size);
//         std::cout << "Initialization done in " << Timer.elapsed_ms() << " ms\n";

//         for (int Iter = 0; Iter < Iterations; Iter++) {
//             Timer.reset();

//             for (size_t Site = 0; Site < Lat.getNsites(); Site++) {
//                 Lat[Site] = RealD(Site);
//                 for (int Dir = 0; Dir < Lat.getDim(); Dir++) {
//                     Tot += Lat.next(Site, Dir);
//                     Tot -= Lat.prev(Site, Dir);
//                 }
//             }
//             Times[Iter] = Timer.elapsed_ms();
//         }

//         std::array<RealD, 2> Stats = get_stats(Times);
//         printHistogram(Times);
//         std::cout << std::format("{} iterations done [ {:.2e} ± {:.2e} ms] {}\n", Iterations, Stats[0], Stats[1],
//         Tot);
//     }
// }
