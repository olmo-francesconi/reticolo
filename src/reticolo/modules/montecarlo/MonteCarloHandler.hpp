/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/MonteCarloHandler.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <H5Dpublic.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reticolo/core/storage/StorageFacade.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"
#include "reticolo/core/tools/io_utils.hpp"
#include "reticolo/core/tools/logger.hpp"
#include "reticolo/core/tools/omp_compat.hpp"
#include "reticolo/core/tools/timer.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/factory/MCAlgorithmFactory.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "yaml-cpp/node/node.h"

namespace fs = std::filesystem;

namespace reticolo::MMonteCarlo {

/*--------------------------------------------------------------------------------------------------
  MonteCarloHandler Class Declaration
--------------------------------------------------------------------------------------------------*/

template <class Action, class TGen = std::mt19937_64>
class MonteCarloHandler : public ModuleBase {
    /* Define types and interface */
  public:
    using action_type = Action::action_type;
    using field_type = Action::field_type;
    using impl_type = Action::impl_type;
    using size_type = Lattice<field_type>::size_type;
    using observables_type = Action::Observables;
    using monte_carlo_data_type = MMonteCarlo::data<action_type>;

    /* Default Constructor */
    MonteCarloHandler() = default;

    /* setup */
    void setup(const YAML::Node& Config) final;

    /* execution */
    void execute(const YAML::Node& RunConfig) final;

    /* Getters for Monte Carlo stats and Observables */
    auto getMCStats() -> monte_carlo_data_type { return _McStats; }
    auto getObs() -> observables_type { return _Obs; }

  private:
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
    std::unique_ptr<MCAlgorithmBase<Action>> _Updater;

    /* montecarlo data storage */
    monte_carlo_data_type              _McStats;        // current MonteCarlo status
    std::vector<monte_carlo_data_type> _McStatsBuffer;  // MonteCarlo status buffer
    observables_type                   _Obs;            // latest measurements values
    std::vector<observables_type>      _ObsBuffer;      // Observables measurements buffer
    size_type                          _NMeasurements;  // Total numebr of measurements

    /* RNG */
    TGen                                _Rng;
    std::normal_distribution<impl_type> _Norm;

    /* Timing and logging */
    Timer      _Timer;
    IO::Logger _Logger;

    /* File output utilities */
    void saveConfiguration(size_type Iter);  // Save current configuration stored in _Lattice

    /* Performs measure stuff, update AVGs and STDs and write if bufffer */
    void measure_utility(const std::string& RunName, unsigned long Iteration);
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action, class TGen>
inline void MonteCarloHandler<Action, TGen>::setup(const YAML::Node& Config) {
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
        throw std::runtime_error("Failed to initialize workspace for Monte Carlo handler '" + _HandlerName +
                                 "': " + e.what());
    }

    // Initialize the logger
    try {
        _Logger.init(_WorkspacePath / "logs", _HandlerName + ".log", _HandlerName + "LOGGER", _StdOut);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize logger for Monte Carlo handler '" + _HandlerName +
                                 "': " + e.what());
    }

    /* Log that the folder and loggind stuff has bee initialized properly */
    _Logger << IO::LI_time() + "[" + _HandlerName + "] - Initialization started...\n";
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "measurements").string() + '\n';
    _Logger << IO::LI_void() + "  configurations folder: " + (_WorkspacePath / "configurations").string() + '\n';

    /* Initialize the output file */
    if (_SaveData) {
        storage::GlobalStorage.initialize_file(_Hdf5OutputFile);
        // Buffers
        _McStatsBuffer.reserve(_MaxBufferSize);
        _ObsBuffer.reserve(_MaxBufferSize);
        _Logger << IO::LI_void() + "            output file: " + _Hdf5OutputFile.string() + "\n";
    }

    /* Allocate Action and Lattice objects */
    _Action = std::make_unique<Action>();
    _Action->setup(Config["action"]["parameters"]);

    /* check reasonable sizes for the lattice */
    auto                TmpSizes = Config["lattice"]["size"].as<std::vector<size_type>>();
    Indexing::size_type TmpVol = 1;
    Indexing::size_type NewVol = 1;
    for (const auto& Size : TmpSizes) {
        NewVol *= Size;
        if (NewVol < TmpVol) {
            throw std::runtime_error("Error allocationg the Lattice: maximun size exceeded");
        }
        TmpVol = NewVol;
    }

    if (TmpVol > Indexing::MaxSize / TmpSizes.size()) {
        throw std::runtime_error(
            std::format("Error allocationg the Lattice: maximun index size exceeded [Max Volume = {} (~{}^{})]",
                        (Indexing::MaxSize / TmpSizes.size()) + 1,                                        //
                        (int)std::pow((Indexing::MaxSize / TmpSizes.size()) + 1, 1.0 / TmpSizes.size()),  //
                        TmpSizes.size()));
    }

    _Field = std::make_unique<Lattice<field_type>>(TmpSizes);
    if (_Field->size() == 0) {
        throw std::runtime_error("Error allocationg the Lattice: possible cause is maximum memeory size exceeded");
    }

    _Action->lattice_sync(*_Field);
    /* Log Action and Lattice parameters */
    auto FieldRamSize = _Field->getNsites() * sizeof(field_type);
    auto IndexRamSize = _Field->getDim() * _Field->getNsites() * sizeof(size_type) * 2;
    _Logger << IO::LI_void() + "[" + _HandlerName + "] - Action-info\n";
    _Logger << IO::LI_void() + "                  name : " + _Action->GetName() + "\n";
    _Logger << IO::LI_void() + "            parameters : " + _Action->GetParameters() + "\n";
    _Logger << IO::LI_void() + "               lattice : " + IO::print(_Field->getSizes()) +  //
                   +" - field: " + IO::pretty_bytes(FieldRamSize) + ", index: " + IO::pretty_bytes(IndexRamSize) + "\n";

    /* Phase quenching warning */
    if (!RealValue<action_type>) {
        _Logger << IO::LI_warn() + "This is a complex-valued action -> Performing a phase-quenched simulation !!\n";
    }

    /* RNGs stuff */
    _Rng.seed(Config["main_seed"].as<unsigned long long>());
    _Norm = std::normal_distribution<impl_type>(0.0, 1.0);

    /* Create updater with Factory */
    _Updater = AlgorithmFactory::MakeUpdater<Action>(Config["algorithm"]["name"].as<std::string>());
    /* Run the Updater setup */
    _Updater->setup(Config["algorithm"]["parameters"], *_Field);

    /* Log initialization end */
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Initialization completed in {:.3f} ms\n", _Timer.elapsed_ms());
}

template <class Action, class TGen>
inline void MonteCarloHandler<Action, TGen>::execute(const YAML::Node& RunConfig) {
    /* Parse configuration from YAML::Node */
    auto      RunName = RunConfig["name"].as<std::string>();
    auto      NMeasures = RunConfig["measures"].as<size_type>();
    auto      MeasureStep = RunConfig["measure_step"].as<size_type>();
    auto      NTherm = RunConfig["therm_steps"].as<size_type>();
    auto      InitializeField = RunConfig["field_init"];
    bool      HotStart = false;
    impl_type HotStartScale = 1.0;

    /* <-- ################## HERE ################## --> */
    if (InitializeField.IsDefined()) {
        HotStart = RunConfig["hot_start"].as<bool>();
        if (HotStart) {
            HotStartScale = RunConfig["hot_start_scale"].as<impl_type>();
        }
    }

    /* Setup HDF5 output */
    size_type ChunkSize = 10000;
    size_type TotMeasure = NMeasures / MeasureStep;
    if (TotMeasure < ChunkSize) {
        ChunkSize = TotMeasure;
    }
    if (ChunkSize == 0) {
        ChunkSize = 1;
    }
    if (_SaveData && MeasureStep > 0) {
        storage::GlobalStorage.ensure_group(_Hdf5OutputFile, storage::schema::montecarlo::run_group(RunName));
        storage::GlobalStorage.setup_appendable_dataset<observables_type>(
            _Hdf5OutputFile, storage::schema::montecarlo::observables_stream(RunName, ChunkSize));
        storage::GlobalStorage.setup_appendable_dataset<monte_carlo_data_type>(
            _Hdf5OutputFile, storage::schema::montecarlo::monte_carlo_stream(RunName, ChunkSize));
    }

    /* Log Run parameters */
    _Logger << IO::LI_time() + "[" + RunName + "] - Starting Monte Carlo updates..\n";

    /* <-- ################## HERE ################## --> */
    /* Initialize the field */
    if (InitializeField) {
        if (HotStart) {
            _Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", HotStartScale);
            // _Field->randomizeField(HotStartScale, _Norm, _Rng);
        } else {
            _Logger << IO::LI_void() + "    initial conditions : cold start\n";
            // _Field->resetField();
        }
    }

    /* Initialize the starting value of the Action */
    action_type SInit = _Action->compute_S(*_Field);
    _McStats.setS(SInit);

    /* Thermalize the field */
    if (NTherm > 0) {
        _Logger << IO::LI_time() + "[" + RunName + "] - Thermalization started...\n";
        _Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", NTherm);
        _Timer.reset();
        for (size_type Iter = 0; Iter < NTherm; Iter++) {
            _Updater->updateField(*_Field, *_Action, _McStats, _Rng);
        }
        _Logger << IO::LI_time() +
                       std::format("[{}] - Thermalization finished in {:.2f} ms\n", RunName, _Timer.elapsed_ms());
    }

    _Logger << IO::LI_void() + "[" + RunName + "] - Starting Measurements..\n";

    /* Initialize measurements buffers */
    if (_SaveData) {
        _McStatsBuffer.clear();
        _McStatsBuffer.reserve(TotMeasure);
        _ObsBuffer.clear();
        _ObsBuffer.reserve(TotMeasure);
    }

    _Logger << IO::LI_void() + "         Total updates : " + std::to_string(NMeasures) + '\n';
    _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';
    _Logger << IO::LI_void() + "    Total measurements : " + std::to_string(TotMeasure) + '\n';

    for (size_type Iteration = 1; Iteration <= NMeasures; Iteration++) {
        // perform a sweep
        _Updater->updateField(*_Field, *_Action, _McStats, _Rng);

        // measure if this is a MeasureStep-th iteration
        if (Iteration % MeasureStep == 0) {
            measure_utility(RunName, Iteration);
        }
    }

    // save the last measurements in the buffer and flush
    if (_SaveData && _McStatsBuffer.size() != 0 && _ObsBuffer.size() != 0) {
        storage::GlobalStorage.append_dataset(_Hdf5OutputFile,
                                              storage::schema::montecarlo::observables_dataset(RunName), _ObsBuffer);
        storage::GlobalStorage.append_dataset(
            _Hdf5OutputFile, storage::schema::montecarlo::monte_carlo_dataset(RunName), _McStatsBuffer);
        _McStatsBuffer.clear();
        _ObsBuffer.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }

    _Logger << IO::LI_time() + std::format("[{}] - Measurements done in {:.3f}  s\n", RunName, _Timer.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action, class TGen>
inline void MonteCarloHandler<Action, TGen>::saveConfiguration(size_type Iter) {
    fs::path          FileName = _WorkspacePath / "configurations" / (std::to_string(Iter) + ".h5");
    std::stringstream RngState;
    RngState << _Rng;
    storage::GlobalStorage.save_lattice(FileName, storage::schema::lattice::field(), *_Field, RngState);
}

template <class Action, class TGen>
inline void MonteCarloHandler<Action, TGen>::measure_utility(const std::string& RunName, unsigned long Iteration) {
    // Measure observables
    _Obs = _Action->Measure(*_Field);

    _NMeasurements++;
    // Save the configuration
    if (_SaveConfig) {
        saveConfiguration(Iteration);
    }
    // Save Monte Carlo data and Observables
    if (_SaveData) {
        // Update Buffers
        _McStatsBuffer.emplace_back(_McStats);
        _ObsBuffer.emplace_back(_Obs);
        // Save to file and flush if:
        //   - The buffers are full (_MaxBufferSize)
        //   - Too much time has passed since last save(_MaxBufferWriteDelay)
        double Elapsed = _Timer.elapsed_s();
        if (_ObsBuffer.size() == _MaxBufferSize || Elapsed > _MaxBufferWriteDelay) {
            // save data
            storage::GlobalStorage.append_dataset(
                _Hdf5OutputFile, storage::schema::montecarlo::observables_dataset(RunName), _ObsBuffer);
            storage::GlobalStorage.append_dataset(
                _Hdf5OutputFile, storage::schema::montecarlo::monte_carlo_dataset(RunName), _McStatsBuffer);
            // Log save state
            double IterPerSec = (_ObsBuffer.size()) / (Elapsed);
            _Logger << IO::LI_time() +
                           std::format("     conf {:>12} : datasaved [{:.2f} conf/s]\n", Iteration, IterPerSec);
            // clear the vectors
            _McStatsBuffer.clear();
            _ObsBuffer.clear();
            // reset the timer
            _Timer.reset();
        }
    }
}

}  // namespace reticolo::MMonteCarlo
