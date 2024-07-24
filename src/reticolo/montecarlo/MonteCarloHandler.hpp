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

#include "reticolo/lattice/lattice.hpp"
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
    using LatticeType = Lattice<FieldType, Action::Dims>;
    using ObsType = typename Action::Observables;
    using ObsVectType = std::vector<typename Action::Observables>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using McDataVectType = std::vector<montecarlo::data<typename Action::ActionType>>;

    /* Metadata and output */
    const std::string _HandlerName;    // Run identifier
    fs::path          _WorkspacePath;  // Workspace folder path

    /* hdf5 stuff */
    fs::path _Hdf5OutputFile;             // output file complete path
    size_t   _MaxBufferSize = 10000;      // Maximun size for the measure buffer
    double   _MaxBufferWriteDelay = 600;  // Maximum delay between data writes in seconds
    bool     _saveConfig;                 // whether or not save the configurations to disk
    bool     _saveData;                   // whether or not save Monte Carlo and Observables to disk

    /* Physics stuff */
    Action&      _Action;  // reference to the Action
    LatticeType& _Field;   // reference to the Lattice

    McDataType     _McStats;        // current MonteCarlo status
    McDataVectType _McStatsBuffer;  // MonteCarlo status buffer
    ObsType        _Obs;            // latest measurements values
    ObsVectType    _ObsBuffer;      // Observables measurements buffer
    unsigned long  _NMeasurements;  // Total numebr of measurements

    /* RNG stuff */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* Timing and logging */
    Timer      _T;       // Private Timer
    IO::Logger _Logger;  // Private Logger

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
    auto getMCStats() -> McDataType { return _McStats; }
    auto getObs() -> ObsType { return _Obs; }
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action>
MonteCarloHandler<Action>::MonteCarloHandler(const std::string& output_path, std::string handler_name, Action& action,
                                             LatticeType& field, uint seed, bool StdOut, bool save_data,
                                             bool save_config)
    : _HandlerName(std::move(handler_name)),
      _Action(action),
      _Field(field),
      _saveConfig(save_config),
      _saveData(save_data) {
    // Begin initialization
    _T.reset();

    // Initialize the folder structure
    try {
        _WorkspacePath = fs::absolute(output_path);
        fs::create_directories(_WorkspacePath);
        _WorkspacePath = fs::canonical(_WorkspacePath);
        _Hdf5OutputFile = _WorkspacePath / "measurements" / (_HandlerName + ".h5");
        fs::create_directories(_WorkspacePath / "logs");
        if (save_config) {
            fs::create_directories(_WorkspacePath / "configurations");
        }
        if (save_data) {
            fs::create_directories(_WorkspacePath / "measurements");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    try {
        _Logger.init(_WorkspacePath / "logs", _HandlerName + ".log", _HandlerName + "LOGGER", StdOut);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Log that the folder and loggind stuff has bee initialized properly
    _Logger << IO::LI_time() + "[" + _HandlerName + "] - Initialization started...\n";
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "measurements").string() + '\n';
    _Logger << IO::LI_void() + "  configurations folder: " + (_WorkspacePath / "configurations").string() + '\n';

    // Initialize the output file
    if (_saveData) {
        IO::GlobalHdf5Handler.initFile(_Hdf5OutputFile);
        // Buffers
        _McStatsBuffer.reserve(_MaxBufferSize);
        _ObsBuffer.reserve(_MaxBufferSize);
    }

    // Log stuff
    _Logger << IO::LI_void() + "            output file: " + _Hdf5OutputFile.string() + "\n";
    _Logger << IO::LI_void() + "[" + _HandlerName + "] - Action-info\n";
    _Logger << IO::LI_void() + "                  name : " + _Action.name() + "\n";
    _Logger << IO::LI_void() + "            parameters : " + _Action.parameters() + "\n";
    _Logger << IO::LI_void() + "               lattice : " + IO::print(_Field.getSizes()) + "\n";

    // initialize the action
    _Action.lattice_sync(_Field);
    if (!RealValue<ActionType>) {
        _Logger << IO::LI_warn() + "This is a complex-valued action -> Performing a phase-quenched simulation!!\n";
    }

    // RNGs stuff
    _Rng.seed(seed);
    _Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    _UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    _Norm = std::normal_distribution<double>(0.0, 1.0);
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
    if (_saveData && MeasureStep > 0) {
        IO::GlobalHdf5Handler.createGroup(_Hdf5OutputFile, RunName);
        IO::GlobalHdf5Handler.setupExpandableDataset<ObsType>(  //
            _Hdf5OutputFile, RunName + "/Observables", ChunkSize, true);
        IO::GlobalHdf5Handler.setupExpandableDataset<McDataType>(  //
            _Hdf5OutputFile, RunName + "/MonteCarlo", ChunkSize, true);
    }

    // Log Run parameters
    _Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    // Initialize the field
    if (initialize_field) {
        if (hot_start) {
            _Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", hs_scale);
            randomizeField(hs_scale);
        } else {
            _Logger << IO::LI_void() + "    initial conditions : cold start\n";
            resetField();
        }
    }

    // Initialize the starting value of the Action
    ActionType SInit = _Action.compute_S(_Field);
    _McStats.setS(SInit);

    // Thermalize the field
    if (nTherm > 0) {
        // Log that the folder and logging stuff has bee initialized properly
        _Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        _Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", nTherm);
        // Reset the timer
        _T.reset();
        // Perform the thermalization sweeps
        for (uint Iter = 0; Iter < nTherm; Iter++) {
            updateField();
        }
        // Log timing information
        _Logger << IO::LI_time() +
                       std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, _T.elapsed_ms());
    }
    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations
    _Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    if (_saveData) {
        // Initialize std::vector of Monte Carlo stats
        _McStatsBuffer.clear();
        _McStatsBuffer.reserve(TotMeasure);
        // Initialize std::vector of observable measurements
        _ObsBuffer.clear();
        _ObsBuffer.reserve(TotMeasure);
    }

    _Logger << IO::LI_void() + "         Total updates : " + std::to_string(nMC) + '\n';
    _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    _Logger << IO::LI_void() + "    Total measurements : " + std::to_string(TotMeasure) + '\n';
    _T.reset();

    for (uint Iteration = 1; Iteration <= nMC; Iteration++) {
        // perform a sweep
        updateField();
        // measure if this is a MeasureStep-th iteration
        if (Iteration % MeasureStep == 0) {
            measure_utility(RunName, Iteration);
        }
    }

    // save the last measurements in the buffer and flush
    if (_saveData && _McStatsBuffer.size() != 0 && _ObsBuffer.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/Observables", _ObsBuffer);
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/MonteCarlo", _McStatsBuffer);
        _McStatsBuffer.clear();
        _ObsBuffer.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }

    _Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, _T.elapsed_s());
}

template <class Action>
inline void MonteCarloHandler<Action>::run_cont(const std::string& RunName, uint nTherm, uint MeasureStep,
                                                bool initialize_field, bool hot_start, double hs_scale) {
    // set reasonable hdf5 dataset chunk size
    if (_saveData && MeasureStep > 0) {
        IO::GlobalHdf5Handler.createGroup(_Hdf5OutputFile, RunName);
        IO::GlobalHdf5Handler.setupExpandableDataset<ObsType>(  //
            _Hdf5OutputFile, RunName + "/Observables", 10000, true);
        IO::GlobalHdf5Handler.setupExpandableDataset<McDataType>(  //
            _Hdf5OutputFile, RunName + "/MonteCarlo", 10000, true);
    }
    // Log Run parameters
    _Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    // Initialize the field
    if (initialize_field) {
        if (hot_start) {
            _Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", hs_scale);
            randomizeField(hs_scale);
        } else {
            _Logger << IO::LI_void() + "    initial conditions : cold start\n";
            resetField();
        }
    }

    // Initialize the starting value of the Action
    ActionType SInit = _Action.compute_S(_Field);
    _McStats.setS(SInit);

    // Thermalize the field
    if (nTherm > 0) {
        // Log that the folder and loggind stuff has bee initialized properly
        _Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        _Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", nTherm);
        // Reset the timer
        _T.reset();
        // Perform the thermalization sweeps
        for (uint Iter = 0; Iter < nTherm; Iter++) {
            updateField();
        }
        // Log timing information
        _Logger << IO::LI_time() +
                       std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, _T.elapsed_ms());
    }
    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations
    _Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    if (_saveData) {
        // Initialize std::vector of Monte Carlo stats
        _McStatsBuffer.clear();
        _McStatsBuffer.reserve(_MaxBufferSize);
        // Initialize std::vector of observable measurements
        _ObsBuffer.clear();
        _ObsBuffer.reserve(_MaxBufferSize);
    }

    _Logger << IO::LI_void() + "         Total updates : inf (Soft exit via single Ctrl+C)\n";
    _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    _T.reset();

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
    if (_saveData && _McStatsBuffer.size() != 0 && _ObsBuffer.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/Observables", _ObsBuffer);
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/MonteCarlo", _McStatsBuffer);
        _McStatsBuffer.clear();
        _ObsBuffer.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }
    _Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, _T.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
inline void MonteCarloHandler<Action>::saveConfiguration(uint Iter) {
    fs::path FileName = _WorkspacePath / "cnfg" / (std::to_string(Iter) + ".h5");
    _Field.save_Configuration(FileName);
}

template <class Action>
inline void MonteCarloHandler<Action>::randomizeField(double scale) {
    for (int Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(_Field[Site], scale, _Norm, _Rng);
    }
    _Action.lattice_sync(_Field);
}

template <class Action>
inline void MonteCarloHandler<Action>::resetField() {
    for (int Site = 0; Site < _Field.getNsites(); Site++) {
        _Field[Site] = FieldType(0.0);
    }
    _Action.lattice_sync(_Field);
}

template <class Action>
inline void MonteCarloHandler<Action>::measure_utility(const std::string& RunName, unsigned long Iteration) {
    // Measure observables
    _Obs = _Action.Measure(_Field);

    _NMeasurements++;
    // Save the configuration
    if (_saveConfig) {
        saveConfiguration(Iteration);
    }
    // Save Monte Carlo data and Observables
    if (_saveData) {
        // Update Buffers
        _McStatsBuffer.emplace_back(_McStats);
        _ObsBuffer.emplace_back(_Obs);
        // Save to file and flush if:
        //   - The buffers are full (_MaxBufferSize)
        //   - Too much time has passed since last save (_MaxBufferWriteDelay)
        double Elapsed = _T.elapsed_s();
        if (_ObsBuffer.size() == _MaxBufferSize || Elapsed > _MaxBufferWriteDelay) {
            // save data
            IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/Observables", _ObsBuffer);
            IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/MonteCarlo", _McStatsBuffer);
            // Log save state
            double IterPerSec = (_ObsBuffer.size()) / (Elapsed);
            _Logger << IO::LI_time() + std::format("conf {:>12} : data saved [{:.2f} conf/s]\n", Iteration, IterPerSec);
            // clear the vectors
            _McStatsBuffer.clear();
            _ObsBuffer.clear();
            // reset the timer
            _T.reset();
        }
    }
}

}  // namespace reticolo::montecarlo
