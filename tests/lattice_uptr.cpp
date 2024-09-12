/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tests/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
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
        std::cout << "No data to display.\n";
        return;
    }

    double MinValue = *std::min_element(data.begin(), data.end());
    double MaxValue = *std::max_element(data.begin(), data.end());
    double Range = MaxValue - MinValue;
    double BinSize = Range / binCount;

    std::vector<int> Bins(binCount, 0);

    // Fill the bins
    for (const double& Value : data) {
        int BinIndex = std::min(static_cast<int>((Value - MinValue) / BinSize), binCount - 1);
        Bins[BinIndex]++;
    }

    // Find the maximum bin count
    int MaxBinCount = *std::max_element(Bins.begin(), Bins.end());

    // Print the histogram

    for (int i = 0; i < binCount; ++i) {
        double BinStart = MinValue + i * BinSize;
        double BinEnd = BinStart + BinSize;
        std::cout << std::fixed << std::setprecision(4) << "[" << BinStart << ", " << BinEnd << "): ";

        int BarLength = static_cast<int>((static_cast<double>(Bins[i]) / MaxBinCount) * maxBinLength);
        for (int j = 0; j < BarLength; ++j) {
            std::cout << '*';
        }
        std::cout << " (" << Bins[i] << ")\n";
    }
}

auto main(int argc, char* argv[]) -> int {
    Timer Time;
    uint  Len = std::stoul(argv[1]);
    uint  Rep = std::stoul(argv[2]);

    const std::vector<uint> Size({Len, Len, Len, Len});
    Time.reset();
    auto LatA = std::make_unique<Lattice<ComplexD>>(Size);
    std::cout << "unique_pointer to Lattice object initialized in " << Time.elapsed_ms() << " ms\n";

    std::mt19937_64                        Rng;    // Random Number Generator
    std::uniform_real_distribution<double> Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )
    Rng.seed(0);

    // Lattice<RealD> LatB(LatA);
    // Lattice<RealD> LatC(LatA);

    // RealD Tot;
    // for (auto& Site : LatB) {
    //     Site = 1.0;
    // }

    // Tot = 0.0;
    // for (auto& Site : LatB) {
    //     Tot += Site;
    // }
    // std::cout << "LatB " << Tot / LatB.getNsites() << "\n";

    // for (auto& Site : LatC) {
    //     Site = 2.0;
    // }

    // Tot = 0.0;
    // for (auto& Site : LatC) {
    //     Tot += Site;
    // }
    // std::cout << "LatC " << Tot / LatB.getNsites() << "\n";

    // std::cout << "Copy C into B\n";
    // LatB = LatC;

    // Tot = 0.0;
    // for (auto& Site : LatB) {
    //     Tot += Site;
    // }
    // std::cout << "LatB " << Tot / LatB.getNsites() << "\n";

    // Tot = 0.0;
    // for (auto& Site : LatC) {
    //     Tot += Site;
    // }
    // std::cout << "LatC " << Tot / LatB.getNsites() << "\n";

    /* support variables */
    std::vector<double> Times;
    Times.resize(Rep);
    ComplexD Tot = 0.0;

    std::cout << "\n### fill lattice with value (for loop) ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d}/{:<4d}]", Idx, Rep) << std::flush;
        Time.reset();
        for (auto& Site : *LatA) {
            Site = 1.0;
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d}/{:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);

    std::cout << "\n### accumulate lattice ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d}/{:<4d}]", Idx, Rep) << std::flush;

        Time.reset();
        for (auto& Site : *LatA) {
            Tot += Site;
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d} / {:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);
    std::cout << "garbage -> " << Tot << "\n";

    std::cout << "\n### sum over all next neighbours ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d} / {:<4d}]", Idx, Rep) << std::flush;
        Time.reset();
        for (uint Site = 0; Site < (*LatA).getNsites(); Site++) {
            for (uint Dir = 0; Dir < (*LatA).getDim(); Dir++) {
                Tot += (*LatA).n(Site, Dir);
            }
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d} / {:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);
    std::cout << "garbage -> " << Tot << "\n";

    std::cout << "\n### sum over all prev neighbours ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d} / {:<4d}]", Idx, Rep) << std::flush;
        Time.reset();
        for (uint Site = 0; Site < (*LatA).getNsites(); Site++) {
            for (uint Dir = 0; Dir < (*LatA).getDim(); Dir++) {
                Tot += (*LatA).p(Site, Dir);
            }
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d} / {:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);
    std::cout << "garbage -> " << Tot << "\n";

    std::cout << "\n### sum over all next-to-next neighbours ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d} / {:<4d}]", Idx, Rep) << std::flush;
        Time.reset();
        for (uint Site = 0; Site < (*LatA).getNsites(); Site++) {
            for (uint Dir1 = 0; Dir1 < (*LatA).getDim(); Dir1++) {
                for (uint Dir2 = 0; Dir2 < (*LatA).getDim(); Dir2++) {
                    Tot += (*LatA).nn(Site, {Dir1, Dir2});
                }
            }
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d} / {:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);
    std::cout << "garbage -> " << Tot << "\n";

    std::cout << "\n### sum over all prev-to-prev neighbours ###\n";
    for (uint Idx = 0; Idx < Rep; Idx++) {
        Tot = 0.0;
        // std::cout << std::format("\r[{:>4d} / {:<4d}]", Idx, Rep) << std::flush;
        Time.reset();
        uint Dims = (*LatA).getDim();
        uint Sites = (*LatA).getNsites();
        for (uint Site = 0; Site < Sites; Site++) {
            for (uint Dir1 = 0; Dir1 < Dims; Dir1++) {
                for (uint Dir2 = 0; Dir2 < Dims; Dir2++) {
                    Tot += (*LatA).pp(Site, {Dir1, Dir2});
                }
            }
        }
        Times[Idx] = Time.elapsed_ms();
    }
    std::cout << std::format("\r[{:>4d} / {:<4d}] Done!\n", Rep, Rep);
    printHistogram(Times);
    std::cout << "garbage -> " << Tot << "\n";
}
