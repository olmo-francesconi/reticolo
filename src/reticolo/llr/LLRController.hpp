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
#include "reticolo/llr/LLRHMCWorker.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace fs = std::filesystem;

namespace reticolo::LLR {

/*--------------------------------------------------------------------------------------------------
    LLRController Class declaration
--------------------------------------------------------------------------------------------------*/
template <class Action>
class LLRController {
  private:
    /* Private types definitions */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using ObsType = typename Action::Observables;

    /* Main Workspace folder */
    fs::path m_WorkspacePath;
    fs::path m_HDF5OutputFile;

    /* LLR workers vector */
    std::vector<std::unique_ptr<LLRHMCWorker<Action>>> m_Workers;

    /* Vector of Lattices */
    std::vector<std::unique_ptr<LatticeType>> m_Fields;

    /* Vector of actions */
    std::vector<std::unique_ptr<Action>> m_Actions;

    /* Main random device*/
    std::random_device m_Rd;

    /* RNG for replica excange */
    std::mt19937_64                        m_Rng;    // Random Number Generator
    std::uniform_real_distribution<double> m_Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> m_UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       m_Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* LLR parameters and data */
    double                           m_DeltaS;
    std::vector<double>              m_AkVect;
    std::vector<std::vector<double>> m_AkHist;
    std::vector<double>              m_SkVect;

    /* Timing and logging */
    Timer      m_T;       // Private Timer
    IO::Logger m_Logger;  // Private Logger

    void ReplicaExcange();

  public:
    /* Constructor (only sets up the Action) */
    LLRController(Action::Params par, const fs::path& out_path, std::vector<int> lattice_size, uint nIntervals,
                  double scale);

    /* Run the actual LLR simulation */
    void run(const std::string& run_id, uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, bool save_data,
             bool save_config);
};

/*--------------------------------------------------------------------------------------------------
    Public methods implementation
--------------------------------------------------------------------------------------------------*/

template <class Action>
LLRController<Action>::LLRController(Action::Params par, const fs::path& out_path, std::vector<int> lattice_size,
                                     uint nIntervals, double scale) {
    // Begin initialization
    m_T.reset();

    // Initialize the folder structure
    try {
        m_WorkspacePath = fs::absolute(out_path);
        fs::create_directories(m_WorkspacePath);
        m_WorkspacePath = fs::canonical(m_WorkspacePath);
        fs::create_directories(m_WorkspacePath / "raw_data");
    } catch (const std::exception& Exept) {
        std::cerr << Exept.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the Logger
    try {
        m_Logger.init(m_WorkspacePath, "controller.log", "LLR_controller_LOGGER", true);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Log folder structure
    m_Logger << IO::LI_time() + "llr_controller - Initialization started...\n";
    m_Logger << IO::LI_void() + "    main oputput folder: " + m_WorkspacePath.string() + '\n';
    m_Logger << IO::LI_void() + "    measurements folder: " + (m_WorkspacePath / "raw_data").string() + '\n';

    // Initialize the output file
    m_HDF5OutputFile = m_WorkspacePath / "llr.h5";
    IO::GlobalHdf5Handler.initFile(m_HDF5OutputFile);

    // Log llr info
    m_Logger << IO::LI_void() + "         ak output file: " + m_HDF5OutputFile.string() + '\n';
    m_Logger << IO::LI_void() + "             Intervals : " + std::to_string(nIntervals) + '\n';
    m_Logger << IO::LI_void() + "             deltaS / V: " + std::to_string(scale) + '\n';

    // Initialize Sk values
    m_DeltaS = scale * (double)std::accumulate(lattice_size.begin(), lattice_size.end(), 0, std::multiplies<>());
    m_Logger << IO::LI_void() + "                 deltaS: " + std::to_string(m_DeltaS) + '\n';
    m_SkVect.reserve(nIntervals);
    for (uint Interval = 0; Interval < nIntervals; Interval++) {
        m_SkVect.push_back(m_DeltaS * ((double)Interval + 0.5));
    }

    // Initialize field, actions and Workers
    m_Actions.resize(m_SkVect.size());
    m_Fields.resize(m_SkVect.size());
    // #pragma omp parallel for schedule(static, 1)
    for (uint Interval = 0; Interval < nIntervals; Interval++) {
        // Initialize field
        m_Fields[Interval] = std::make_unique<LatticeType>(lattice_size);
        // Initialize action
        m_Actions[Interval] = std::make_unique<Action>(*m_Fields[Interval], par);
    }

    // Initialize the internal RNG
    m_Rng.seed(m_Rd());

    // Log initialization finish
    m_Logger << IO::LI_time() + std::format("llr_controller - Initialization completed in {:.3f} s\n", m_T.elapsed_s());
}

template <class Action>
void LLRController<Action>::run(const std::string& run_id, uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo,
                                bool save_data, bool save_config) {
    // Initalize timer
    m_T.reset();
    m_Logger << IO::LI_time() + "llr_controller - Starting run [" + run_id + "]\n";

    // Initialize ak values
    m_AkVect.resize(m_SkVect.size());
    std::fill(m_AkVect.begin(), m_AkVect.end(), 0.0);
    // Clears the ak history vector
    m_AkHist.clear();
    m_AkHist.push_back(m_AkVect);

    // Initialize field, actions and Workers
    m_Workers.resize(m_SkVect.size());
    for (uint WorkerId = 0; WorkerId < m_Workers.size(); WorkerId++) {
        // Initialize worker
        m_Workers[WorkerId] =
            std::make_unique<LLRHMCWorker<Action>>(std::format("llr_worker[{:0>3}]", WorkerId),  // workerId
                                                   *m_Actions[WorkerId],                         // Action reference
                                                   *m_Fields[WorkerId],                          // Field reference
                                                   m_Rd(),                                       // random seed
                                                   m_WorkspacePath / "raw_data" / run_id,        // Output path
                                                   false,                                        // StdOut
                                                   save_data,     // Save raw Monte Carlo and Observables
                                                   save_config);  // Save configurations
        // Set parameters
        // _Workers[WorkerId]->setParams(5, 0.05, 0.2);  // HMC - Met
        m_Workers[WorkerId]->setParams(0.2, 10);  // HMC
        // _Workers[WorkerId]->setParams(0.2);  // Met

        m_Workers[WorkerId]->setLLRParams(m_AkVect[WorkerId], m_SkVect[WorkerId], m_DeltaS);
    }

#pragma omp parallel
    {
        // initial burn in
#pragma omp for schedule(static, 1)
        for (auto& Worker : m_Workers) {
            Worker->run("burn-in", 1000, 0, 1, true, false);
        }

#pragma omp single
        {
            m_Logger << IO::LI_time() +
                            std::format("[{}] Burn-in completed           [{:.3f} s]\n", run_id, m_T.elapsed_s());
            m_T.reset();
        }

        // Newton-Raphson iterations
        for (uint Step = 0; Step < nNewton_Raphson; Step++) {
#pragma omp single
            m_Logger << IO::LI_time() + std::format("[{}] Started NR iteration {}\n", run_id, Step);

            std::string RunName = std::format("NR_{}", Step);
#pragma omp for schedule(static, 1)
            for (uint WorkerId = 0; WorkerId < m_Workers.size(); WorkerId++) {
                m_Workers[WorkerId]->run(RunName, nMonte_Carlo, 100, 1, false);
                // update the ak value
                double DSIm = m_Workers[WorkerId]->getMCStats()._SIm - m_SkVect[WorkerId];
                m_AkVect[WorkerId] += DSIm / (m_DeltaS * m_DeltaS);
                m_Workers[WorkerId]->set_ak(m_AkVect[WorkerId]);
            }
#pragma omp single
            {
                m_AkHist.push_back(m_AkVect);
                ReplicaExcange();
            }
        }

#pragma omp single
        {
            m_Logger << IO::LI_time() +
                            std::format("[{}] Newton-Raphson done         [{:.3f} s]\n", run_id, m_T.elapsed_s());
            m_T.reset();
        }

        // Robbins-Monro Iterations
        for (uint Step = 0; Step < nRobbins_Monro; Step++) {
#pragma omp single
            m_Logger << IO::LI_time() + std::format("[{}] started RM iteration {}\n", run_id, Step);

            std::string RunName = std::format("RM_{}", Step);
#pragma omp for schedule(static, 1)
            for (uint WorkerId = 0; WorkerId < m_Workers.size(); WorkerId++) {
                m_Workers[WorkerId]->run(RunName, nMonte_Carlo, 100, 1, false);
                // update the ak value
                double DSIm = m_Workers[WorkerId]->getMCStats()._SIm - m_SkVect[WorkerId];
                m_AkVect[WorkerId] += DSIm / ((double)(Step + 1) * (m_DeltaS * m_DeltaS));
                m_Workers[WorkerId]->set_ak(m_AkVect[WorkerId]);
            }
#pragma omp single
            {
                m_AkHist.push_back(m_AkVect);
                ReplicaExcange();
            }
        }
    }
    m_Logger << IO::LI_time() + std::format("[{}] Robbins-Monro done in       [{:.3f} s]\n", run_id, m_T.elapsed_s());

    m_T.reset();
    std::vector<double>     AkBuffer;
    std::vector<McDataType> McAvgBuffer;
    std::vector<McDataType> McVarBuffer;
    std::vector<ObsType>    ObsBuffer;
    AkBuffer.reserve(m_AkHist.size());
    IO::GlobalHdf5Handler.createGroup(m_HDF5OutputFile, run_id);
    for (uint WorkerId = 0; WorkerId < m_Workers.size(); WorkerId++) {
        AkBuffer.clear();
        for (const auto& Elem : m_AkHist) {
            AkBuffer.push_back(Elem[WorkerId]);
        }
        IO::GlobalHdf5Handler.writeDataset(m_HDF5OutputFile, std::format("{}/[{}]ak", run_id, WorkerId), AkBuffer);
    }
    m_Logger << IO::LI_void() + std::format("[{}] Data saved                  [{:.3f} s]\n", run_id, m_T.elapsed_s());
}  // namespace reticolo::LLR

template <class Action>
void LLRController<Action>::ReplicaExcange() {
    RealD SImA;
    RealD SImB;
    RealD SkA;
    RealD SkB;
    RealD AkA;
    RealD AkB;
    RealD Weight = (2.0 * m_DeltaS * m_DeltaS);

    // Generate random order for swaps
    std::vector<int> SwapIds(m_Workers.size() - 1);
    std::iota(std::begin(SwapIds), std::end(SwapIds), 0);
    std::ranges::shuffle(SwapIds, m_Rng);

    for (const auto& SwapId : SwapIds) {
        SImA = m_Workers[SwapId]->getMCStats()._SIm;
        AkA = m_AkVect[SwapId];
        SkA = m_SkVect[SwapId];
        SImB = m_Workers[SwapId + 1]->getMCStats()._SIm;
        AkB = m_AkVect[SwapId + 1];
        SkB = m_SkVect[SwapId + 1];

        RealD DAA = SImA - SkA;
        RealD DAB = SImA - SkB;
        RealD DBA = SImA - SkB;
        RealD DBB = SImB - SkB;

        RealD SwapWeight =                    //
            DAA * DAA / Weight + AkA * DAA -  // weight of A in A
            DAB * DAB / Weight - AkB * DAB +  // weight of A in B
            DBB * DBB / Weight + AkB * DBB -  // weight of B in B
            DBA * DBA / Weight - AkA * DBA;   // weight of B in A

        if (exp(SwapWeight) > m_Unif(m_Rng)) {
            std::swap(m_Workers[SwapId], m_Workers[SwapId + 1]);
            m_Workers[SwapId]->setLLRParams(m_AkVect[SwapId], m_SkVect[SwapId], m_DeltaS);
            m_Workers[SwapId + 1]->setLLRParams(m_AkVect[SwapId + 1], m_SkVect[SwapId + 1], m_DeltaS);
        }
    }
}

}  // namespace reticolo::LLR
