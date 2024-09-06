/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/controller.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/llr/LLRHMCWorker.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace fs = std::filesystem;

namespace reticolo::MLLR {

/*--------------------------------------------------------------------------------------------------
    LLRController Class declaration
--------------------------------------------------------------------------------------------------*/
template <class Action>
class LLRController : ModuleBase {
  private:
    /* Private types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using McDataType = MMonteCarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* Main Workspace folder */
    fs::path _WorkspacePath;
    fs::path _HDF5OutputFile;

    /* LLR workers vector */
    std::vector<std::unique_ptr<LLRHMCWorker<Action>>> _Workers;

    /* Vector of Lattices */
    std::vector<std::unique_ptr<LatticeType>> _Fields;

    /* Vector of actions */
    std::vector<std::unique_ptr<Action>> _Actions;

    /* Main random device*/
    std::random_device _Rd;

    /* RNG for replica excange */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* LLR parameters and data */
    double                           _DeltaS;
    std::vector<double>              _AkVect;
    std::vector<std::vector<double>> _AkHist;
    std::vector<double>              _SkVect;

    /* Timing and logging */
    Timer      _T;       // Private Timer
    IO::Logger _Logger;  // Private Logger

    void ReplicaExcange();

  public:
    /* Constructor (only sets up the Action) */
    LLRController(Action::Params par, const fs::path& out_path, std::vector<int> lattice_size, uint nIntervals,
                  double scale);

    /* Run the actual LLR simulation */
    void run(const std::string& run_id, uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, bool save_data,
             bool save_config);

    /* setup */
    void setup() final = 0;

    /* execution */
    void execute() final = 0;
};

/*--------------------------------------------------------------------------------------------------
    Public methods implementation
--------------------------------------------------------------------------------------------------*/

template <class Action>
LLRController<Action>::LLRController(Action::Params par, const fs::path& out_path, std::vector<int> lattice_size,
                                     uint nIntervals, double scale) {
    // Begin initialization
    _T.reset();

    // Initialize the folder structure
    try {
        _WorkspacePath = fs::absolute(out_path);
        fs::create_directories(_WorkspacePath);
        _WorkspacePath = fs::canonical(_WorkspacePath);
        fs::create_directories(_WorkspacePath / "raw_data");
    } catch (const std::exception& Exept) {
        std::cerr << Exept.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the Logger
    try {
        _Logger.init(_WorkspacePath, "controller.log", "LLR_controller_LOGGER", true);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Log folder structure
    _Logger << IO::LI_time() + "llr_controller - Initialization started...\n";
    _Logger << IO::LI_void() + "    main oputput folder: " + _WorkspacePath.string() + '\n';
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "raw_data").string() + '\n';

    // Initialize the output file
    _HDF5OutputFile = _WorkspacePath / "llr.h5";
    IO::GlobalHdf5Handler.initFile(_HDF5OutputFile);

    // Log llr info
    _Logger << IO::LI_void() + "         ak output file: " + _HDF5OutputFile.string() + '\n';
    _Logger << IO::LI_void() + "             Intervals : " + std::to_string(nIntervals) + '\n';
    _Logger << IO::LI_void() + "             deltaS / V: " + std::to_string(scale) + '\n';

    // Initialize Sk values
    _DeltaS = scale * (double)std::accumulate(lattice_size.begin(), lattice_size.end(), 0, std::multiplies<>());
    _Logger << IO::LI_void() + "                 deltaS: " + std::to_string(_DeltaS) + '\n';
    _SkVect.reserve(nIntervals);
    for (uint Interval = 0; Interval < nIntervals; Interval++) {
        _SkVect.push_back(_DeltaS * ((double)Interval + 0.5));
    }

    // Initialize field, actions and Workers
    _Actions.resize(_SkVect.size());
    _Fields.resize(_SkVect.size());
    // #pragma omp parallel for schedule(static, 1)
    for (uint Interval = 0; Interval < nIntervals; Interval++) {
        // Initialize field
        _Fields[Interval] = std::make_unique<LatticeType>(lattice_size);
        // Initialize action
        _Actions[Interval] = std::make_unique<Action>(*_Fields[Interval], par);
    }

    // Initialize the internal RNG
    _Rng.seed(_Rd());

    // Log initialization finish
    _Logger << IO::LI_time() + std::format("llr_controller - Initialization completed in {:.3f} s\n", _T.elapsed_s());
}

template <class Action>
void LLRController<Action>::run(const std::string& run_id, uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo,
                                bool save_data, bool save_config) {
    // Initalize timer
    _T.reset();
    _Logger << IO::LI_time() + "llr_controller - Starting run [" + run_id + "]\n";

    // Initialize ak values
    _AkVect.resize(_SkVect.size());
    std::fill(_AkVect.begin(), _AkVect.end(), 0.0);
    // Clears the ak history vector
    _AkHist.clear();
    _AkHist.push_back(_AkVect);

    // Initialize field, actions and Workers
    _Workers.resize(_SkVect.size());
    for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
        // Initialize worker
        _Workers[WorkerId] =
            std::make_unique<LLRHMCWorker<Action>>(std::format("llr_worker[{:0>3}]", WorkerId),  // workerId
                                                   *_Actions[WorkerId],                          // Action reference
                                                   *_Fields[WorkerId],                           // Field reference
                                                   _Rd(),                                        // random seed
                                                   _WorkspacePath / "raw_data" / run_id,         // Output path
                                                   false,                                        // StdOut
                                                   save_data,     // Save raw Monte Carlo and Observables
                                                   save_config);  // Save configurations
        // Set parameters
        // _Workers[WorkerId]->setParams(5, 0.05, 0.2);  // HMC - Met
        _Workers[WorkerId]->setParams(0.2, 10);  // HMC
        // _Workers[WorkerId]->setParams(0.2);  // Met

        _Workers[WorkerId]->setLLRParams(_AkVect[WorkerId], _SkVect[WorkerId], _DeltaS);
    }

#pragma omp parallel
    {
        // initial burn in
#pragma omp for schedule(static, 1)
        for (auto& Worker : _Workers) {
            Worker->run("burn-in", 1000, 0, 1, true, false);
        }

#pragma omp single
        {
            _Logger << IO::LI_time() +
                           std::format("[{}] Burn-in completed           [{:.3f} s]\n", run_id, _T.elapsed_s());
            _T.reset();
        }

        // Newton-Raphson iterations
        for (uint Step = 0; Step < nNewton_Raphson; Step++) {
#pragma omp single
            _Logger << IO::LI_time() + std::format("[{}] Started NR iteration {}\n", run_id, Step);

            std::string RunName = std::format("NR_{}", Step);
#pragma omp for schedule(static, 1)
            for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
                _Workers[WorkerId]->run(RunName, nMonte_Carlo, 100, 1, false);
                // update the ak value
                double DSIm = _Workers[WorkerId]->getMCStats()._SIm - _SkVect[WorkerId];
                _AkVect[WorkerId] += DSIm / (_DeltaS * _DeltaS);
                _Workers[WorkerId]->set_ak(_AkVect[WorkerId]);
            }
#pragma omp single
            {
                _AkHist.push_back(_AkVect);
                ReplicaExcange();
            }
        }

#pragma omp single
        {
            _Logger << IO::LI_time() +
                           std::format("[{}] Newton-Raphson done         [{:.3f} s]\n", run_id, _T.elapsed_s());
            _T.reset();
        }

        // Robbins-Monro Iterations
        for (uint Step = 0; Step < nRobbins_Monro; Step++) {
#pragma omp single
            _Logger << IO::LI_time() + std::format("[{}] started RM iteration {}\n", run_id, Step);

            std::string RunName = std::format("RM_{}", Step);
#pragma omp for schedule(static, 1)
            for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
                _Workers[WorkerId]->run(RunName, nMonte_Carlo, 100, 1, false);
                // update the ak value
                double DSIm = _Workers[WorkerId]->getMCStats()._SIm - _SkVect[WorkerId];
                _AkVect[WorkerId] += DSIm / ((double)(Step + 1) * (_DeltaS * _DeltaS));
                _Workers[WorkerId]->set_ak(_AkVect[WorkerId]);
            }
#pragma omp single
            {
                _AkHist.push_back(_AkVect);
                ReplicaExcange();
            }
        }
    }
    _Logger << IO::LI_time() + std::format("[{}] Robbins-Monro done in       [{:.3f} s]\n", run_id, _T.elapsed_s());

    _T.reset();
    std::vector<double>     AkBuffer;
    std::vector<McDataType> McAvgBuffer;
    std::vector<McDataType> McVarBuffer;
    std::vector<ObsType>    ObsBuffer;
    AkBuffer.reserve(_AkHist.size());
    IO::GlobalHdf5Handler.createGroup(_HDF5OutputFile, run_id);
    for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
        AkBuffer.clear();
        for (const auto& Elem : _AkHist) {
            AkBuffer.push_back(Elem[WorkerId]);
        }
        IO::GlobalHdf5Handler.writeDataset(_HDF5OutputFile, std::format("{}/[{}]ak", run_id, WorkerId), AkBuffer);
    }
    _Logger << IO::LI_void() + std::format("[{}] Data saved                  [{:.3f} s]\n", run_id, _T.elapsed_s());
}  // namespace reticolo::LLR

template <class Action>
void LLRController<Action>::ReplicaExcange() {
    RealD SImA;
    RealD SImB;
    RealD SkA;
    RealD SkB;
    RealD AkA;
    RealD AkB;
    RealD Weight = (2.0 * _DeltaS * _DeltaS);

    // Generate random order for swaps
    std::vector<int> SwapIds(_Workers.size() - 1);
    std::iota(std::begin(SwapIds), std::end(SwapIds), 0);
    std::ranges::shuffle(SwapIds, _Rng);

    for (const auto& SwapId : SwapIds) {
        SImA = _Workers[SwapId]->getMCStats()._SIm;
        AkA = _AkVect[SwapId];
        SkA = _SkVect[SwapId];
        SImB = _Workers[SwapId + 1]->getMCStats()._SIm;
        AkB = _AkVect[SwapId + 1];
        SkB = _SkVect[SwapId + 1];

        RealD DAA = SImA - SkA;
        RealD DAB = SImA - SkB;
        RealD DBA = SImA - SkB;
        RealD DBB = SImB - SkB;

        RealD SwapWeight =                    //
            DAA * DAA / Weight + AkA * DAA -  // weight of A in A
            DAB * DAB / Weight - AkB * DAB +  // weight of A in B
            DBB * DBB / Weight + AkB * DBB -  // weight of B in B
            DBA * DBA / Weight - AkA * DBA;   // weight of B in A

        if (exp(SwapWeight) > _Unif(_Rng)) {
            std::swap(_Workers[SwapId], _Workers[SwapId + 1]);
            _Workers[SwapId]->setLLRParams(_AkVect[SwapId], _SkVect[SwapId], _DeltaS);
            _Workers[SwapId + 1]->setLLRParams(_AkVect[SwapId + 1], _SkVect[SwapId + 1], _DeltaS);
        }
    }
}

}  // namespace reticolo::MLLR
