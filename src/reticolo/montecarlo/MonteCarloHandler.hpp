/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/MonteCarloHandler.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Cpp.h>
#include <omp.h>
#include <unistd.h>

#include <array>
#include <cmath>
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
    const std::string _RunName;        // Run identifier
    fs::path          _WorkspacePath;  // Workspace folder path

    /* hdf5 stuff */
    fs::path     _Hdf5OutputFile;            // output file complete path
    const size_t _MaxBufferSize = 10000;     // Maximun size for the measure buffer
    const double _MaxBufferWriteDelay = 60;  // Maximum delay between data writes

    /* Physics stuff */
    Action&        _Action;   // reference to the Action
    LatticeType&   _Field;    // reference to the Lattice
    McDataType     _McStats;  // MonteCarlo stats
    McDataVectType _Stats;    // MonteCarlo stats buffer
    ObsVectType    _Obs;      // Observable measurements buffer

    /* RNG stuff */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* Timing and logging */
    Timer      _T;       // LLRWorker Private Timer
    IO::Logger _Logger;  // LLRWorker Private Logger

    /* File output utilities */
    void save_Configuration(uint Iter);  // Save curretn configuration stored in _Lattice

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Clear the field (cold start)*/
    void resetField();

    /* Performs a single sweep of the lattice */
    virtual void updateField() = 0;

    /* Performs nSteps thermalization steps and no measurements */
    void thermalize(uint nSteps, bool initialize_field = false, bool hot_start = false, double scale = 1.0);

  public:
    /* Constructor */
    MonteCarloHandler(std::string run_name, Action& action, LatticeType& field, uint seed,
                      const std::string& output_path);

    /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
    void run(uint nMC, uint nTherm, uint MeasureStep = 1, bool initialize_field = false, bool hot_start = false,
             double hs_scale = 1.0, bool save_config = false);
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action>
MonteCarloHandler<Action>::MonteCarloHandler(std::string run_name, Action& action, LatticeType& field, uint seed,
                                             const std::string& output_path)
    : _RunName(std::move(run_name)), _Action(action), _Field(field) {
    // Begin initialization
    _T.reset();

    // Initialize the folder structure
    try {
        _WorkspacePath = fs::absolute(output_path);
        fs::create_directories(_WorkspacePath);
        _WorkspacePath = fs::canonical(_WorkspacePath);
        _Hdf5OutputFile = _WorkspacePath / "meas" / (_RunName + ".h5");
        fs::create_directories(_WorkspacePath / "logs");
        fs::create_directories(_WorkspacePath / "meas");
        fs::create_directories(_WorkspacePath / "cnfg");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    try {
        _Logger.init(_WorkspacePath / "logs", _RunName + ".log", _RunName + "LOGGER", true);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Log that the folder and loggind stuff has bee initialized properly
    _Logger << IO::LI_time() + "Monte Carlo Handler - Initialization started...\n";
    _Logger << IO::LI_void() + "    main oputput folder: " + _WorkspacePath.string() + '\n';
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "meas").string() + '\n';
    _Logger << IO::LI_void() + "  configurations folder: " + (_WorkspacePath / "cnfg").string() + '\n';

    // Initialize the output file
    IO::GlobalHdf5Handler.initFile(_Hdf5OutputFile);
    IO::GlobalHdf5Handler.setupExpandableDataset<ObsType>(_Hdf5OutputFile, "Observables", _MaxBufferSize);
    IO::GlobalHdf5Handler.setupExpandableDataset<McDataType>(_Hdf5OutputFile, "MonteCarlo", _MaxBufferSize);

    // Log stuff
    _Logger << IO::LI_void() + "            output file: " + _Hdf5OutputFile.string() + "\n";
    _Logger << IO::LI_void() + "Monte Carlo Handler - Action-info\n";
    _Logger << IO::LI_void() + "                  name : " + _Action.action_name() + "\n";
    _Logger << IO::LI_void() + "            parameters : " + _Action.action_parameters() + "\n";
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

    // Buffers
    _Stats.reserve(_MaxBufferSize);
    _Obs.reserve(_MaxBufferSize);
}

template <class Action>
inline void MonteCarloHandler<Action>::run(uint nMC, uint nTherm, uint MeasureStep, bool initialize_field,
                                           bool hot_start, double hs_scale, bool save_config) {
    // Log Run parameters
    _Logger << IO::LI_time() + "Monte Carlo Handler - Starting Monte Carlo updates..\n";

    if (nTherm > 0) {
        thermalize(nTherm, initialize_field, hot_start, hs_scale);
    }

    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations

    _Logger << IO::LI_void() + "Monte Carlo Handler - Starting Measurements..\n";

    // Initialize std::vector of Monte Carlo stats
    McDataVectType Stats;
    Stats.clear();
    Stats.reserve(_MaxBufferSize);

    // Initialize std::vector of observable measurements
    std::vector<typename Action::Observables> Obs;
    Obs.clear();
    Obs.reserve(_MaxBufferSize);

    // Initialize the starting value of S
    ActionType SInit = _Action.compute_S(_Field);
    _McStats.setS(SInit);

    // Keep track of the iteration number
    unsigned long Iteration = 1;
    double        Time = GlobalTimer.elapsed_s();
    uint          SaveIter = 0;

    // Reset the timer
    _T.reset();
    // simulation workflow for indefinite end
    if (nMC == 0) {
        // LogMessage << IO::LI_void() << "        OpenMP threads : " << omp_get_max_threads() << "\n"
        _Logger << IO::LI_void() + "         Total Updates : inf (Soft exit via single Ctrl+C)\n";
        _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';

        // Set the SIGINT handler to perform a SoftExit
        // [Ctrl+C]x1 -> stop simulation on the next iteration
        // [Ctrl+C]x2 -> immediate stop, possible datacorruption if that
        //               happpens during data output
        SignalHandler::set_SIGINT_handler(SignalHandler::SIGINT_SoftExit);

        while (!SignalHandler::SoftExitRequested) {
            // perform a sweep
            updateField();
            // measure if this is a MeasureStep-th iteration
            if (Iteration % MeasureStep == 0) {
                // Save the configuration
                if (save_config) {
                    save_Configuration(Iteration);
                }
                Stats.emplace_back(_McStats);
                Obs.push_back(_Action.Measure(_Field));
                // Save to file and flush if:
                //   - The buffers are full (_MaxBudderSize)
                //   - Too much time has passed since last save (_MaxBufferWriteDelay)
                double NewTime = GlobalTimer.elapsed_s();
                if (Obs.size() == _MaxBufferSize || NewTime - Time > _MaxBufferWriteDelay) {
                    // save data
                    IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "Observables", Obs);
                    IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "MonteCarlo", Stats);

                    // clear the vectors
                    Stats.clear();
                    Obs.clear();
                    double IterPerSec = static_cast<double>(Iteration - SaveIter) / (NewTime - Time);
                    _Logger << IO::LI_time() +
                                   std::format("Data saved : conf {} [{:.2f} conf/s]\n", Iteration, IterPerSec);
                    SaveIter = Iteration;
                    Time = NewTime;
                }
            }
            Iteration++;
        }
        SignalHandler::SIGINT_reset();
    } else {
        // LogMessage << IO::LI_void() << "        OpenMP threads : " << omp_get_max_threads() << "\n"
        _Logger << IO::LI_void() + "         Total Updates : " + std::to_string(nMC) + '\n';
        _Logger << IO::LI_void() + "           Mesure step : " + std::to_string(MeasureStep) + '\n';

        for (uint Iteration = 1; Iteration <= nMC; Iteration++) {
            // perform a sweep
            updateField();
            // measure if this is a MeasureStep-th iteration
            if (Iteration % MeasureStep == 0) {
                // Save the configuration
                if (save_config) {
                    save_Configuration(Iteration);
                }
                Stats.emplace_back(_McStats);
                Obs.push_back(_Action.Measure(_Field));
                // Save to file and flush if:
                //   - The buffers are full (_MaxBudderSize)
                //   - Too much time has passed since last save (_MaxBufferWriteDelay)
                double NewTime = GlobalTimer.elapsed_s();
                if (Obs.size() == _MaxBufferSize || NewTime - Time > _MaxBufferWriteDelay) {
                    // save data
                    IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "Observables", Obs);
                    IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "MonteCarlo", Stats);
                    // clear the vectors
                    Stats.clear();
                    Obs.clear();
                    double IterPerSec = static_cast<double>(Iteration - SaveIter) / (NewTime - Time);
                    _Logger << IO::LI_time() +
                                   std::format("conf {:>12} : data saved [{:.2f} conf/s]\n", Iteration, IterPerSec);
                    SaveIter = Iteration;
                    Time = NewTime;
                }
            }
        }
    }

    // save the last measurements in the buffer and flush
    if (Stats.size() != 0 && Obs.size() != 0) {
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "Observables", Obs);
        IO::GlobalHdf5Handler.appendDataset(_Hdf5OutputFile, "MonteCarlo", Stats);
        Stats.clear();
        Obs.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }
    _Logger << IO::LI_time() + std::format("Monte Carlo Handler - Measurements done in {:.3f}  s\n", _T.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
inline void MonteCarloHandler<Action>::save_Configuration(uint Iter) {
    fs::path FileName = _WorkspacePath / "cnfg" / (std::to_string(Iter) + ".h5");
    try {
        _Field.save_Configuration(FileName);
    } catch (H5::Exception& _) {
        exit(EXIT_FAILURE);
    }
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
inline void MonteCarloHandler<Action>::thermalize(uint nSteps, bool initialize_field, bool hot_start, double scale) {
    // Log that the folder and loggind stuff has bee initialized properly
    _Logger << IO::LI_time() + "Monte Carlo Handler - Thermalization started...\n";
    _Logger << IO::LI_void() + std::format("  thermalizatoin steps : {}\n", nSteps);
    if (initialize_field) {
        if (hot_start) {
            randomizeField(scale);
            _Logger << IO::LI_void() + std::format("    initial conditions : hot start [scale : {}]\n", scale);
        } else {
            resetField();
            _Logger << IO::LI_void() + "    initial conditions : cold start\n";
        }
    }
    // Reset the timer
    _T.reset();

    // Initialize the value of the action
    ActionType SInit = _Action.compute_S(_Field);
    _McStats.setS(SInit);
    // Perform the thermalization sweeps
    for (uint Iter = 0; Iter < nSteps; Iter++) {
        updateField();
    }
    // Log timing information
    _Logger << IO::LI_time() +
                   std::format("Monte Carlo Handler - Thermalization finished in {:.2f} ms\n", _T.elapsed_ms());
}

}  // namespace reticolo::montecarlo
