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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/modules/montecarlo/algorithms/AlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/algorithms/AlgorithmFactory.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "yaml-cpp/node/node.h"

namespace fs = std::filesystem;

namespace reticolo::montecarlo {

/*--------------------------------------------------------------------------------------------------
  MonteCarloHandler Class Declaration
--------------------------------------------------------------------------------------------------*/

template <class Action>
class MonteCarloHandler : public ModuleBase {
  private:
    /* Define types */
    using action_type = Action::ActionType;
    using field_type = typename Action::FieldType;
    using observables_type = typename Action::Observables;
    using monte_carlo_data_type = montecarlo::data<typename Action::ActionType>;

    /* Metadata and output */
    std::string _HandlerName;                // Handler Id
    fs::path    _WorkspacePath;              // Workspace folder path
    fs::path    _Hdf5OutputFile;             // output file complete path
    size_t      _MaxBufferSize = 10000;      // Maximun size for the measure buffer
    double      _MaxBufferWriteDelay = 600;  // Maximum delay between data writes in seconds
    bool        _SaveConfig;                 // whether or not save the configurations to disk
    bool        _SaveData;                   // whether or not save Monte Carlo and Observables to disk
    bool        _StdOut;                     // whether or not to log to standard output

    /* Lattice and Action objects */
    std::unique_ptr<Action>              _Action;  // reference to the Action
    std::unique_ptr<Lattice<field_type>> _Field;   // reference to the Lattice

    /* Algorithm object */
    std::unique_ptr<AlgorithmBase<Action>> _Updater;

    /* montecarlo data storage */
    monte_carlo_data_type              _McStats;        // current MonteCarlo status
    std::vector<monte_carlo_data_type> _McStatsBuffer;  // MonteCarlo status buffer
    observables_type                   _Obs;            // latest measurements values
    std::vector<observables_type>      _ObsBuffer;      // Observables measurements buffer
    unsigned long                      _NMeasurements;  // Total numebr of measurements

    /* RNG */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* Timing and logging */
    Timer      _Timer;   // Private Timer
    IO::Logger _Logger;  // Private Logger

    /* File output utilities */
    void saveConfiguration(uint Iter);  // Save current configuration stored in _Lattice

    /* Performs measure stuff, update AVGs and STDs and write if bufffer */
    void measure_utility(const std::string& RunName, unsigned long Iteration) {};

  public:
    MonteCarloHandler() = default;

    /* setup */
    void setup(const YAML::Node& Config) final;

    /* execution */
    void execute(const YAML::Node& RunConfig) final;

    /* Getters for Monte Carlo stats and Observables */
    auto getMCStats() -> monte_carlo_data_type { return _McStats; }
    auto getObs() -> observables_type { return _Obs; }
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action>
inline void MonteCarloHandler<Action>::setup(const YAML::Node& Config) {
    /* Begin initialization */
    _Timer.reset();

    /* Parse settings from Configuration*/
    _HandlerName = Config["name"].as<std::string>();
    _SaveConfig = Config["save_config"].as<bool>();
    _SaveData = Config["save_data"].as<bool>();
    _StdOut = Config["console_output"].as<bool>();

    /* Initialize the folder structure */
    try {
        _WorkspacePath = fs::absolute(Config["output_dir"].as<std::string>());
        fs::create_directories(_WorkspacePath);
        _WorkspacePath = fs::canonical(_WorkspacePath);
        _Hdf5OutputFile = _WorkspacePath / "measurements" / (_HandlerName + ".h5");
        fs::create_directories(_WorkspacePath / "logs");
        if (_SaveConfig) {
            fs::create_directories(_WorkspacePath / "configurations");
        }
        if (_SaveData) {
            fs::create_directories(_WorkspacePath / "measurements");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    try {
        _Logger.init(_WorkspacePath / "logs", _HandlerName + ".log", _HandlerName + "LOGGER", _StdOut);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    /* Log that the folder and loggind stuff has bee initialized properly */
    _Logger << IO::LI_time() + "[" + _HandlerName + "] - Initialization started...\n";
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "measurements").string() + '\n';
    _Logger << IO::LI_void() + "  configurations folder: " + (_WorkspacePath / "configurations").string() + '\n';

    /* Initialize the output file */
    if (_SaveData) {
        IO::GlobalHdf5Handler.initFile(_Hdf5OutputFile);
        // Buffers
        _McStatsBuffer.reserve(_MaxBufferSize);
        _ObsBuffer.reserve(_MaxBufferSize);
        _Logger << IO::LI_void() + "            output file: " + _Hdf5OutputFile.string() + "\n";
    }

    /* Allocate Action and Lattice objects */
    _Action = std::make_unique<Action>();
    _Action->setup(Config["action"]["parameters"]);

    _Field = std::make_unique<Lattice<field_type>>(
        Config["lattice"]["size"].as<std::vector<typename Lattice<field_type>::SizeType>>());

    _Action->lattice_sync(*_Field);

    /* Log Action and Lattice parameters */
    _Logger << IO::LI_void() + "[" + _HandlerName + "] - Action-info\n";
    _Logger << IO::LI_void() + "                  name : " + _Action->GetName() + "\n";
    _Logger << IO::LI_void() + "            parameters : " + _Action->GetParameters() + "\n";
    _Logger << IO::LI_void() + "               lattice : " + IO::print(_Field->getSizes()) + "\n";

    /* Phase quenching warning */
    if (!RealValue<action_type>) {
        _Logger << IO::LI_warn() + "This is a complex-valued action -> Performing a phase-quenched simulation !!\n";
    }

    /* RNGs stuff */
    _Rng.seed(Config["main_seed"].as<unsigned long long>());
    _Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    _UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    _Norm = std::normal_distribution<double>(0.0, 1.0);

    // /* Allocate Algorithm object */
    // if (Config["algorithm"]["name"].as<std::string>() == "HMC") {
    //     _Updater = std::make_unique<HMC<Action>>();
    // } else if (Config["algorithm"].as<std::string>() == "Metropolis") {
    //     _Updater = nullptr;
    // }

    _Updater = AlgorithmFactory<Action>::MakeUpdater(Config["algorithm"]["name"].as<std::string>());

    /* Run the Updater setup */
    _Updater->setup(Config["algorithm"]["parameters"], *_Field);

    /* Log initialization end */
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _Timer.elapsed_ms());
}

template <class Action>
inline void MonteCarloHandler<Action>::execute(const YAML::Node& RunConfig) {
    auto   RunName = RunConfig["name"].as<std::string>();
    auto   NMeasures = RunConfig["measures"].as<uint>();
    auto   MeasureStep = RunConfig["measurte_step"].as<uint>();
    auto   NTherm = RunConfig["therm_steps"].as<uint>();
    auto   InitializeField = RunConfig["field_init"].as<bool>();
    bool   HotStart = false;
    double HotStartScale = 1.0;
    if (InitializeField) {
        HotStart = RunConfig["HotStart"].as<bool>();
        if (HotStart) {
            HotStartScale = RunConfig["HotStart"].as<double>();
        }
    }

    // set reasonable hdf5 dataset chunk size
    uint ChunkSize = 10000;
    uint TotMeasure = NMeasures / MeasureStep;
    if (TotMeasure < ChunkSize) {
        ChunkSize = TotMeasure;
    }
    if (ChunkSize == 0) {
        ChunkSize = 1;
    }
    if (_SaveData && MeasureStep > 0) {
        IO::GlobalHdf5Handler.createGroup(_Hdf5OutputFile, RunName);
        IO::GlobalHdf5Handler.setupExpandableDataset<observables_type>(  //
            _Hdf5OutputFile, RunName + "/Observables", ChunkSize, true);
        IO::GlobalHdf5Handler.setupExpandableDataset<monte_carlo_data_type>(  //
            _Hdf5OutputFile, RunName + "/MonteCarlo", ChunkSize, true);
    }

    // Log Run parameters
    _Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    // Initialize the field
    if (InitializeField) {
        if (HotStart) {
            _Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", HotStartScale);
            // _Field->randomizesFiled(HotStartScale, _Rng, _Norm);
        } else {
            _Logger << IO::LI_void() + "    initial conditions : cold start\n";
            // _Field->reset();
        }
    }

    // Initialize the starting value of the Action
    action_type SInit = _Action->compute_S(*_Field);
    _McStats.setS(SInit);

    // Thermalize the field
    if (NTherm > 0) {
        // Log that the folder and logging stuff has bee initialized properly
        _Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        _Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", NTherm);
        // Reset the timer
        _Timer.reset();
        // Perform the thermalization sweeps
        for (uint Iter = 0; Iter < NTherm; Iter++) {
            _Updater->updateField(*_Field, *_Action, _McStats);
        }
        // Log timing information
        _Logger << IO::LI_time() +
                       std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, _Timer.elapsed_ms());
    }
    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary
    // iterations
    _Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    if (_SaveData) {
        // Initialize std::vector of Monte Carlo stats
        _McStatsBuffer.clear();
        _McStatsBuffer.reserve(TotMeasure);
        // Initialize std::vector of observable measurements
        _ObsBuffer.clear();
        _ObsBuffer.reserve(TotMeasure);
    }

    _Logger << IO::LI_void() + "         Total updates : " + std::to_string(NMeasures) + '\n';
    _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    _Logger << IO::LI_void() + "    Total measurements : " + std::to_string(TotMeasure) + '\n';
    _Timer.reset();

    for (uint Iteration = 1; Iteration <= NMeasures; Iteration++) {
        // perform a sweep
        _Updater->updateField(*_Field, *_Action, _McStats);

        // measure if this is a MeasureStep-th iteration
        if (Iteration % MeasureStep == 0) {
            measure_utility(RunName, Iteration);
        }
    }

    // save the last measurements in the buffer and flush
    if (_SaveData && _McStatsBuffer.size() != 0 && _ObsBuffer.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/Observables", _ObsBuffer);
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName + "/MonteCarlo", _McStatsBuffer);
        _McStatsBuffer.clear();
        _ObsBuffer.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }

    _Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, _Timer.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

// template <class Action>
// inline void MonteCarloHandler<Action>::saveConfiguration(uint Iter) {
//     fs::path FileName = _WorkspacePath / "cnfg" / (std::to_string(Iter) + ".h5");
//     _Field.save_Configuration(FileName);
// }

// template <class Action>
// inline void MonteCarloHandler<Action>::measure_utility(const std::string&
// RunName, unsigned long Iteration) {
//     // Measure observables
//     _Obs = _Action.Measure(_Field);
//
//     _NMeasurements++;
//     // Save the configuration
//     if (_SaveConfig) {
//         saveConfiguration(Iteration);
//     }
//     // Save Monte Carlo data and Observables
//     if (_SaveData) {
//         // Update Buffers
//         _McStatsBuffer.emplace_back(_McStats);
//         _ObsBuffer.emplace_back(_Obs);
//         // Save to file and flush if:
//         //   - The buffers are full (_MaxBufferSize)
//         //   - Too much time has passed since last save
//         (_MaxBufferWriteDelay) double Elapsed = _Timer.elapsed_s(); if
//         (_ObsBuffer.size() == _MaxBufferSize || Elapsed >
//         _MaxBufferWriteDelay) {
//             // save data
//             IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName +
//             "/Observables", _ObsBuffer);
//             IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, RunName +
//             "/MonteCarlo", _McStatsBuffer);
//             // Log save state
//             double IterPerSec = (_ObsBuffer.size()) / (Elapsed);
//             _Logger << IO::LI_time() + std::format("conf {:>12} : data
//             saved [{:.2f} conf/s]\n", Iteration, IterPerSec);
//             // clear the vectors
//             _McStatsBuffer.clear();
//             _ObsBuffer.clear();
//             // reset the timer
//             _Timer.reset();
//         }
//     }
// }

}  // namespace reticolo::montecarlo
