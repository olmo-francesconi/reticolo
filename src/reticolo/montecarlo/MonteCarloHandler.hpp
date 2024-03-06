/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/metropolis.hpp

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
#include <sstream>
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

template <class Action, size_t dim>
class MonteCarloHandler {
  protected:
    /* Define types */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<FieldType, dim>;
    using ObsType = typename Action::Observables;
    using ObsVectType = std::vector<typename Action::Observables>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using McDataVectType = std::vector<montecarlo::data<typename Action::ActionType>>;

    /* Metadata and output */
    std::string _RunName;        // Run identifier
    fs::path    _WorkspacePath;  // Workspace folder path

    /* hdf5 stuff */
    fs::path           _Hdf5OutputFile;                                  // output file complete path
    const uint         _MaxBufferSize = 10000;                           // Maximun size for the measure buffer
    const double       _MaxBufferWriteDelay = 60;                        // Maximum delay between data writes
    const H5::CompType _McCompType = McDataType::make_hdf5_CompType();   // CompType for Monte Carlo stats
    const H5::CompType _ObsCompType = Action::make_obs_hdf5_CompType();  // CompType for Action observables

    /* Physics stuff */
    Action      _Action;   // Action
    LatticeType _Field;    // Lattice of the type specified by action
    McDataType  _McStats;  // MonteCarlo stats

    /* RNG stuff */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* Timing and logging */
    Timer      _T;       // LLRWorker Private Timer
    IO::Logger _Logger;  // LLRWorker Private Logger

    /* Various logging utilities */
    void log_MC(uint iter);                 // Log MonteCarlo data
    void log_obs(uint iter, ObsType& obs);  // Log Action observables

    /* File output utilities */
    void save_Measurements(McDataVectType& McVect, ObsVectType& ObsVect);  // save measurements
    void save_Configuration(uint Iter);  // Save curretn configuration stored in _Lattice

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Clear the field (cold start)*/
    void resetField();

    /* Performs a single sweep of the lattice */
    void updateField();

    /* Performs nSteps thermalization steps and no measurements */
    void thermalize(uint nSteps, bool hot_start = false, double scale = 1.0);

    /* Memory report */
    auto memoryReport() -> size_t;

  public:
    /* Constructor */
    MonteCarloHandler(std::string run_name, Action& action, Lattice<FieldType, dim>& field, uint seed,
                      const std::string& output_path);

    /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
    void run(uint nMC, uint nTherm, uint MeasureStep = 1, bool hot_start = false, double hs_scale = 1.0,
             bool save_config = false);
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/
template <class Action, size_t dim>
MonteCarloHandler<Action, dim>::MonteCarloHandler(  //
    std::string              run_name,              //
    Action&                  action,                //
    Lattice<FieldType, dim>& field,                 //
    uint                     seed,                  //
    const std::string&       output_path)
    : _RunName(std::move(run_name)),  //
      _Action(action),
      _Field(field) {
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
    _Logger << IO::LI_time() + "MetropolisWorker - Initialization started...\n";
    _Logger << IO::LI_void() + "    main oputput folder: " + _WorkspacePath.string() + '\n';
    _Logger << IO::LI_void() + "    measurements folder: " + (_WorkspacePath / "meas").string() + '\n';
    _Logger << IO::LI_void() + "  configurations folder: " + (_WorkspacePath / "cnfg").string() + '\n';

    // Initialize the output file
    try {
        // Create empty but unlimited dataspace
        std::array<hsize_t, 1> Hdf5Entries = {0};
        std::array<hsize_t, 1> Hdf5MaxDims = {H5S_UNLIMITED};
        H5::DataSpace          Hdf5DataSpace(1, Hdf5Entries.data(), Hdf5MaxDims.data());
        // Create file (truncation access)
        H5::H5File Hdf5File(_Hdf5OutputFile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        // Create Group
        H5::Group Hdf5Group = Hdf5File.createGroup(_RunName);
        // Create creation properties for dataset
        H5::DSetCreatPropList  Hdf5CParms;
        std::array<hsize_t, 1> Hdf5ChunkDims = {_MaxBufferSize};
        Hdf5CParms.setChunk(1, Hdf5ChunkDims.data());
        // Create DataSet
        H5::DataSet Hdf5ObsDataSet =
            Hdf5File.createDataSet(_RunName + "/Observables", _ObsCompType, Hdf5DataSpace, Hdf5CParms);
        // Create DataSet
        H5::DataSet Hdf5McDataSet =
            Hdf5File.createDataSet(_RunName + "/MonteCarlo", _McCompType, Hdf5DataSpace, Hdf5CParms);
    } catch (H5::Exception& _) {
        exit(EXIT_FAILURE);
    }

    // Log stuff
    _Logger << IO::LI_void() + "            output file: " + _Hdf5OutputFile.string() + "\n";
    _Logger << IO::LI_void() + "MetropolisWorker - Action-info\n";
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

    // Log stuff
    _Logger << IO::LI_time() +
                   std::format("MetropolisWorker - Initialization completed in {:.3f} ms\n", _T.elapsed_ms());
    _Logger << IO::LI_void() + "      memory allocated : " + IO::pretty_bytes(memoryReport()) + '\n';
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::run(  //
    uint   nMC,                                   //
    uint   nTherm,                                //
    uint   MeasureStep,                           //
    bool   hot_start,                             //
    double hs_scale,                              //
    bool   save_config) {
    // Log Run parameters
    _Logger << IO::LI_time() + "MetropolisWorker - Starting Monte Carlo updates..\n";

    if (nTherm > 0) {
        thermalize(nTherm, hot_start, hs_scale);
    }

    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations

    _Logger << IO::LI_void() + "MetropolisWorker - Starting Measurements..\n";

    // Reset the timer
    _T.reset();

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
                    save_Measurements(Stats, Obs);
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
                    save_Measurements(Stats, Obs);
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
        save_Measurements(Stats, Obs);
        Stats.clear();
        Obs.clear();
        _Logger << IO::LI_void() + "Final data saved\n";
    }
    _Logger << IO::LI_time() + std::format("MetropolisWorker - Measurements done in {:.3f}  s\n", _T.elapsed_s());
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::save_Measurements(McDataVectType& McVect, ObsVectType& ObsVect) {
    try {
        // Define dimension of stuff to write
        std::array<hsize_t, 1> Hdf5BufferSize = {ObsVect.size()};
        H5::DataSpace          Hdf5BufferMemorySpace(1, Hdf5BufferSize.data());

        // Open file
        H5::H5File Hdf5File(_Hdf5OutputFile, H5F_ACC_RDWR, H5P_DEFAULT, H5P_DEFAULT);

        // Observables //
        // Open DataSet
        H5::DataSet Hdf5ObsDataSet = Hdf5File.openDataSet(_RunName + "/Observables");
        // Get current size of the dataset
        H5::DataSpace          Hdf5ObsDataSpace = Hdf5ObsDataSet.getSpace();
        std::array<hsize_t, 1> Hdf5ObsDimsOld;
        Hdf5ObsDataSpace.getSimpleExtentDims(Hdf5ObsDimsOld.data());
        // Extend dataspace dimension
        std::array<hsize_t, 1> Hdf5ObsDimsNew = {Hdf5ObsDimsOld[0] + Hdf5BufferSize[0]};
        Hdf5ObsDataSet.extend(Hdf5ObsDimsNew.data());
        // Update dataspace
        Hdf5ObsDataSpace = Hdf5ObsDataSet.getSpace();
        // Select Hyperslab
        Hdf5ObsDataSpace.selectHyperslab(H5S_SELECT_SET, Hdf5BufferSize.data(), Hdf5ObsDimsOld.data());
        // Write to dataset
        Hdf5ObsDataSet.write(ObsVect.data(), _ObsCompType, Hdf5BufferMemorySpace, Hdf5ObsDataSpace);

        // MonteCarlo stuff //
        // Open DataSet
        H5::DataSet Hdf5McDataSet = Hdf5File.openDataSet(_RunName + "/MonteCarlo");
        // Get current size of the dataset
        H5::DataSpace          Hdf5McDataSpace = Hdf5McDataSet.getSpace();
        std::array<hsize_t, 1> Hdf5McDimsOld;
        Hdf5McDataSpace.getSimpleExtentDims(Hdf5McDimsOld.data());
        // Extend dataspace dimension
        std::array<hsize_t, 1> Hdf5McDimsNew = {Hdf5McDimsOld[0] + Hdf5BufferSize[0]};
        Hdf5McDataSet.extend(Hdf5McDimsNew.data());
        // Update dataspace
        Hdf5McDataSpace = Hdf5McDataSet.getSpace();
        // Select Hyperslab
        Hdf5McDataSpace.selectHyperslab(H5S_SELECT_SET, Hdf5BufferSize.data(), Hdf5McDimsOld.data());
        // Write to dataset
        Hdf5McDataSet.write(McVect.data(), _McCompType, Hdf5BufferMemorySpace, Hdf5McDataSpace);
    } catch (H5::Exception& _) {
        exit(EXIT_FAILURE);
    }
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::save_Configuration(uint Iter) {
    fs::path FileName = _WorkspacePath / "cnfg" / (std::to_string(Iter) + ".h5");
    try {
        _Field.save_Configuration(FileName);
    } catch (H5::Exception& _) {
        exit(EXIT_FAILURE);
    }
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::log_MC(uint iter) {
    _Logger.log_string(_RunName, std::format(" MC data_dump {} -> ", iter) + _McStats.dump_str());
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::log_obs(uint iter, ObsType& obs) {
    _Logger.log_string(_RunName, std::format("obs data_dump {} -> ", iter) + obs.dump_str());
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::randomizeField(double scale) {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(_Field[Site], scale, _Norm, _Rng);
    }
    _Action.lattice_sync(_Field);
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::resetField() {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        _Field[Site] = FieldType(0.0);
    }
    _Action.lattice_sync(_Field);
}

template <class Action, size_t dim>
inline void MonteCarloHandler<Action, dim>::thermalize(uint nSteps, bool hot_start, double scale) {
    std::stringstream LogMessage;
    // Log that the folder and loggind stuff has bee initialized properly
    LogMessage << IO::LI_time() << "MetropolisWorker - Thermalization started...\n"
               << IO::LI_void() << "  thermalizatoin steps : " << nSteps << "\n";
    if (hot_start) {
        randomizeField(scale);
        LogMessage << IO::LI_void() << "    initial conditions : hot start\n";
    } else {
        resetField();
        LogMessage << IO::LI_void() << "    initial conditions : cold start\n";
    }
    _Logger << LogMessage;

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
    LogMessage << IO::LI_time() << "MetropolisWorker - Thermalization finished in "
               << std::format("{:.2f}", _T.elapsed_ms()) << " ms\n";
    _Logger << LogMessage;
}

template <class Action, size_t dim>
inline auto MonteCarloHandler<Action, dim>::memoryReport() -> size_t {
    size_t Memory = 0;
    Memory += _Field.memoryReport();

    return Memory;
}

}  // namespace reticolo::montecarlo
