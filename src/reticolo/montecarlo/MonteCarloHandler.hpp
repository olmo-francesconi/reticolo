/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/MonteCarloHandler.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <omp.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/signalHandler.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/random.hpp"

namespace fs = std::filesystem;

namespace reticolo::montecarlo {

/*--------------------------------------------------------------------------------------------------
  MonteCarloHandler Class Declaration
--------------------------------------------------------------------------------------------------*/

template <class Action>
class MonteCarloHandler {
  protected:
    /* Define types */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<FieldType>;
    using ObsType = typename Action::Observables;
    using ObsVectType = std::vector<typename Action::Observables>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using McDataVectType = std::vector<montecarlo::data<typename Action::ActionType>>;

    /* Metadata and output */
    const std::string m_HandlerName;    // Run identifier
    fs::path          m_WorkspacePath;  // Workspace folder path

    /* hdf5 stuff */
    fs::path m_Hdf5OutputFile;             // output file complete path
    size_t   m_MaxBufferSize = 10000;      // Maximun size for the measure buffer
    double   m_MaxBufferWriteDelay = 600;  // Maximum delay between data writes in seconds
    bool     m_SaveConfig;                 // whether or not save the configurations to disk
    bool     m_SaveData;                   // whether or not save Monte Carlo and Observables to disk

    /* Physics stuff */
    Action&      m_Action;  // reference to the Action
    LatticeType& m_Field;   // reference to the Lattice

    McDataType     m_McStats;        // current MonteCarlo status
    McDataVectType m_McStatsBuffer;  // MonteCarlo status buffer
    ObsType        m_Obs;            // latest measurements values
    ObsVectType    m_ObsBuffer;      // Observables measurements buffer
    unsigned long  m_NMeasurements;  // Total numebr of measurements

    /* RNG stuff */
    std::mt19937_64                        m_Rng;    // Random Number Generator
    std::uniform_real_distribution<double> m_Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> m_UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       m_Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* Timing and logging */
    Timer      m_Timer;   // Private Timer
    IO::Logger m_Logger;  // Private Logger

    /* File output utilities */
    void saveConfiguration(uint Iter);  // Save curretn configuration stored in _Lattice

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Clear the field (cold start)*/
    void resetField();

    /* Performs a single sweep of the lattice */
    virtual void updateField() = 0;

    /* Performs measure stuff, update AVGs and STDs and write if bufffer */
    void measure_utility(const std::string& RunName, unsigned long Iteration);

  public:
    /* Constructor */
    MonteCarloHandler(const std::string& output_path, std::string handler_name, Action& action, LatticeType& field,
                      uint seed, bool StdOut, bool save_data, bool save_config);

    /* Performs nMC Monte Carlo steps */
    void run(const std::string& RunName, uint nMC, uint nTherm, uint MeasureStep = 1, bool initialize_field = false,
             bool hot_start = false, double hs_scale = 1.0);

    /* Runs indefinitely */
    void run_cont(const std::string& RunName, uint nTherm, uint MeasureStep, bool initialize_field, bool hot_start,
                  double hs_scale);

    /* Getters for Monte Carlo stats and Observables */
    auto getMCStats() -> McDataType { return m_McStats; }
    auto getObs() -> ObsType { return m_Obs; }
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action>
MonteCarloHandler<Action>::MonteCarloHandler(const std::string& output_path, std::string handler_name, Action& action,
                                             LatticeType& field, uint seed, bool StdOut, bool save_data,
                                             bool save_config)
    : m_HandlerName(std::move(handler_name)),
      m_Action(action),
      m_Field(field),
      m_SaveConfig(save_config),
      m_SaveData(save_data) {
    // Begin initialization
    m_Timer.reset();

    // Initialize the folder structure
    try {
        m_WorkspacePath = fs::absolute(output_path);
        fs::create_directories(m_WorkspacePath);
        m_WorkspacePath = fs::canonical(m_WorkspacePath);
        m_Hdf5OutputFile = m_WorkspacePath / "measurements" / (m_HandlerName + ".h5");
        fs::create_directories(m_WorkspacePath / "logs");
        if (save_config) {
            fs::create_directories(m_WorkspacePath / "configurations");
        }
        if (save_data) {
            fs::create_directories(m_WorkspacePath / "measurements");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    try {
        m_Logger.init(m_WorkspacePath / "logs", m_HandlerName + ".log", m_HandlerName + "LOGGER", StdOut);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Log that the folder and loggind stuff has bee initialized properly
    m_Logger << IO::LI_time() + "[" + m_HandlerName + "] - Initialization started...\n";
    m_Logger << IO::LI_void() + "    measurements folder: " + (m_WorkspacePath / "measurements").string() + '\n';
    m_Logger << IO::LI_void() + "  configurations folder: " + (m_WorkspacePath / "configurations").string() + '\n';

    // Initialize the output file
    if (m_SaveData) {
        IO::GlobalHdf5Handler.initFile(m_Hdf5OutputFile);
        // Buffers
        m_McStatsBuffer.reserve(m_MaxBufferSize);
        m_ObsBuffer.reserve(m_MaxBufferSize);
    }

    // Log stuff
    m_Logger << IO::LI_void() + "            output file: " + m_Hdf5OutputFile.string() + "\n";
    m_Logger << IO::LI_void() + "[" + m_HandlerName + "] - Action-info\n";
    m_Logger << IO::LI_void() + "                  name : " + m_Action.name() + "\n";
    m_Logger << IO::LI_void() + "            parameters : " + m_Action.parameters() + "\n";
    m_Logger << IO::LI_void() + "               lattice : " + IO::print(m_Field.getSizes()) + "\n";

    // initialize the action
    m_Action.lattice_sync(m_Field);
    if (!RealValue<ActionType>) {
        m_Logger << IO::LI_warn() + "This is a complex-valued action -> Performing a phase-quenched simulation!!\n";
    }

    // RNGs stuff
    m_Rng.seed(seed);
    m_Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    m_UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    m_Norm = std::normal_distribution<double>(0.0, 1.0);
}

template <class Action>
inline void MonteCarloHandler<Action>::run(const std::string& RunName, uint nMC, uint nTherm, uint MeasureStep,
                                           bool initialize_field, bool hot_start, double hs_scale) {
    // set reasonable hdf5 dataset chunk size
    uint ChunkSize = 10000;
    uint TotMeasure = nMC / MeasureStep;
    if (TotMeasure < ChunkSize) {
        ChunkSize = TotMeasure;
    }
    if (ChunkSize == 0) {
        ChunkSize = 1;
    }
    if (m_SaveData && MeasureStep > 0) {
        IO::GlobalHdf5Handler.createGroup(m_Hdf5OutputFile, RunName);
        IO::GlobalHdf5Handler.setupExpandableDataset<ObsType>(  //
            m_Hdf5OutputFile, RunName + "/Observables", ChunkSize, true);
        IO::GlobalHdf5Handler.setupExpandableDataset<McDataType>(  //
            m_Hdf5OutputFile, RunName + "/MonteCarlo", ChunkSize, true);
    }

    // Log Run parameters
    m_Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    // Initialize the field
    if (initialize_field) {
        if (hot_start) {
            m_Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", hs_scale);
            randomizeField(hs_scale);
        } else {
            m_Logger << IO::LI_void() + "    initial conditions : cold start\n";
            resetField();
        }
    }

    // Initialize the starting value of the Action
    ActionType SInit = m_Action.compute_S(m_Field);
    m_McStats.setS(SInit);

    // Thermalize the field
    if (nTherm > 0) {
        // Log that the folder and logging stuff has bee initialized properly
        m_Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        m_Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", nTherm);
        // Reset the timer
        m_Timer.reset();
        // Perform the thermalization sweeps
        for (uint Iter = 0; Iter < nTherm; Iter++) {
            updateField();
        }
        // Log timing information
        m_Logger << IO::LI_time() +
                        std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, m_Timer.elapsed_ms());
    }
    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations
    m_Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    if (m_SaveData) {
        // Initialize std::vector of Monte Carlo stats
        m_McStatsBuffer.clear();
        m_McStatsBuffer.reserve(TotMeasure);
        // Initialize std::vector of observable measurements
        m_ObsBuffer.clear();
        m_ObsBuffer.reserve(TotMeasure);
    }

    m_Logger << IO::LI_void() + "         Total updates : " + std::to_string(nMC) + '\n';
    m_Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    m_Logger << IO::LI_void() + "    Total measurements : " + std::to_string(TotMeasure) + '\n';
    m_Timer.reset();

    for (uint Iteration = 1; Iteration <= nMC; Iteration++) {
        // perform a sweep
        updateField();
        // measure if this is a MeasureStep-th iteration
        if (Iteration % MeasureStep == 0) {
            measure_utility(RunName, Iteration);
        }
    }

    // save the last measurements in the buffer and flush
    if (m_SaveData && m_McStatsBuffer.size() != 0 && m_ObsBuffer.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/Observables", m_ObsBuffer);
        IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/MonteCarlo", m_McStatsBuffer);
        m_McStatsBuffer.clear();
        m_ObsBuffer.clear();
        m_Logger << IO::LI_void() + "Final data saved\n";
    }

    m_Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, m_Timer.elapsed_s());
}

template <class Action>
inline void MonteCarloHandler<Action>::run_cont(const std::string& RunName, uint nTherm, uint MeasureStep,
                                                bool initialize_field, bool hot_start, double hs_scale) {
    // set reasonable hdf5 dataset chunk size
    if (m_SaveData && MeasureStep > 0) {
        IO::GlobalHdf5Handler.createGroup(m_Hdf5OutputFile, RunName);
        IO::GlobalHdf5Handler.setupExpandableDataset<ObsType>(  //
            m_Hdf5OutputFile, RunName + "/Observables", 10000, true);
        IO::GlobalHdf5Handler.setupExpandableDataset<McDataType>(  //
            m_Hdf5OutputFile, RunName + "/MonteCarlo", 10000, true);
    }
    // Log Run parameters
    m_Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    // Initialize the field
    if (initialize_field) {
        if (hot_start) {
            m_Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", hs_scale);
            randomizeField(hs_scale);
        } else {
            m_Logger << IO::LI_void() + "    initial conditions : cold start\n";
            resetField();
        }
    }

    // Initialize the starting value of the Action
    ActionType SInit = m_Action.compute_S(m_Field);
    m_McStats.setS(SInit);

    // Thermalize the field
    if (nTherm > 0) {
        // Log that the folder and loggind stuff has bee initialized properly
        m_Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        m_Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", nTherm);
        // Reset the timer
        m_Timer.reset();
        // Perform the thermalization sweeps
        for (uint Iter = 0; Iter < nTherm; Iter++) {
            updateField();
        }
        // Log timing information
        m_Logger << IO::LI_time() +
                        std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, m_Timer.elapsed_ms());
    }
    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations
    m_Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    if (m_SaveData) {
        // Initialize std::vector of Monte Carlo stats
        m_McStatsBuffer.clear();
        m_McStatsBuffer.reserve(m_MaxBufferSize);
        // Initialize std::vector of observable measurements
        m_ObsBuffer.clear();
        m_ObsBuffer.reserve(m_MaxBufferSize);
    }

    m_Logger << IO::LI_void() + "         Total updates : inf (Soft exit via single Ctrl+C)\n";
    m_Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    m_Timer.reset();

    // Set the SIGINT handler to perform a SoftExit
    // [Ctrl+C]x1 -> stop simulation on the next iteration
    // [Ctrl+C]x2 -> immediate stop, possible datacorruption if that
    //               happpens during data output
    SignalHandler::set_SIGINT_handler(SignalHandler::SIGINT_SoftExit);
    // Keep track of the iteration number
    unsigned long Iteration = 0;
    while (!SignalHandler::SoftExitRequested) {
        // perform a sweep
        updateField();
        // measure if this is a MeasureStep-th iteration
        if (Iteration % MeasureStep == 0) {
            measure_utility(RunName, Iteration);
        }
        Iteration++;
    }
    SignalHandler::SIGINT_reset();

    // save the last measurements in the buffer and flush
    if (m_SaveData && m_McStatsBuffer.size() != 0 && m_ObsBuffer.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/Observables", m_ObsBuffer);
        IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/MonteCarlo", m_McStatsBuffer);
        m_McStatsBuffer.clear();
        m_ObsBuffer.clear();
        m_Logger << IO::LI_void() + "Final data saved\n";
    }
    m_Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, m_Timer.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
inline void MonteCarloHandler<Action>::saveConfiguration(uint Iter) {
    fs::path FileName = m_WorkspacePath / "cnfg" / (std::to_string(Iter) + ".h5");
    m_Field.save_Configuration(FileName);
}

template <class Action>
inline void MonteCarloHandler<Action>::randomizeField(double scale) {
    for (int Site = 0; Site < m_Field.getNsites(); Site++) {
        randomize(m_Field[Site], scale, m_Norm, m_Rng);
    }
    m_Action.lattice_sync(m_Field);
}

template <class Action>
inline void MonteCarloHandler<Action>::resetField() {
    for (int Site = 0; Site < m_Field.getNsites(); Site++) {
        m_Field[Site] = FieldType(0.0);
    }
    m_Action.lattice_sync(m_Field);
}

template <class Action>
inline void MonteCarloHandler<Action>::measure_utility(const std::string& RunName, unsigned long Iteration) {
    // Measure observables
    m_Obs = m_Action.Measure(m_Field);

    m_NMeasurements++;
    // Save the configuration
    if (m_SaveConfig) {
        saveConfiguration(Iteration);
    }
    // Save Monte Carlo data and Observables
    if (m_SaveData) {
        // Update Buffers
        m_McStatsBuffer.emplace_back(m_McStats);
        m_ObsBuffer.emplace_back(m_Obs);
        // Save to file and flush if:
        //   - The buffers are full (_MaxBufferSize)
        //   - Too much time has passed since last save (_MaxBufferWriteDelay)
        double Elapsed = m_Timer.elapsed_s();
        if (m_ObsBuffer.size() == m_MaxBufferSize || Elapsed > m_MaxBufferWriteDelay) {
            // save data
            IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/Observables", m_ObsBuffer);
            IO::GlobalHdf5Handler.appendDataset(m_Hdf5OutputFile, RunName + "/MonteCarlo", m_McStatsBuffer);
            // Log save state
            double IterPerSec = (m_ObsBuffer.size()) / (Elapsed);
            m_Logger << IO::LI_time() +
                            std::format("conf {:>12} : data saved [{:.2f} conf/s]\n", Iteration, IterPerSec);
            // clear the vectors
            m_McStatsBuffer.clear();
            m_ObsBuffer.clear();
            // reset the timer
            m_Timer.reset();
        }
    }
}

}  // namespace reticolo::montecarlo
