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
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
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
  LLRWorker Class Declaration
--------------------------------------------------------------------------------------------------*/

template <class Action>
class MetropolisWorker {
  private:
    /* Define types */
    using ActionType = typename Action::ActionType;
    using FieldType = typename Action::FieldType;
    using LatticeType = Lattice<typename Action::FieldType, Action::Dims>;
    using ObsType = typename Action::Observables;
    using ObsVectType = std::vector<typename Action::Observables>;
    using McDataType = montecarlo::data<typename Action::ActionType>;
    using McDataVectType = std::vector<montecarlo::data<typename Action::ActionType>>;

    /* Metadata and output */
    std::string _RunName;        // Run identifier
    fs::path    _WorkspacePath;  // Workspace folder path

    /* hdf5 stuff */
    const uint         _MaxBufferSize = 1000;                            // Maximun size for the measure buffer
    fs::path           _Hdf5OutputFile;                                  // output file complete path
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

  public:
    /* Constructor (only sets up the Action) */
    MetropolisWorker(Action& act) : _Action(act){};

    /* Initializer function -> actually does all the set-up */
    void init(uintvect<Action::Dims> sizes, const std::string& RunName, uint seed, const fs::path& output_path);

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Clear the field (cold start)*/
    void resetField();

    /* Performs a single sweep of the lattice (updates worker::MC_stats) */
    void sweep();

    /* Performs nSteps thermalization steps and no measurements */
    void thermalize(uint nSteps);

    /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
    void run(uint nMC, uint nTherm, uint MeasureStep = 1);

    /* Memory report */
    auto memoryReport() -> size_t;
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
void MetropolisWorker<Action>::init(uintvect<Action::Dims> sizes, const std::string& RunName, uint seed,
                                    const fs::path& output_path) {
    _T.reset();

    _RunName = RunName;

    // Initialize the folder structure
    try {
        _WorkspacePath = fs::absolute(output_path);

        fs::create_directories(_WorkspacePath);
        _WorkspacePath = fs::canonical(_WorkspacePath);
        _Hdf5OutputFile = _WorkspacePath / "meas" / (_RunName + ".h5");
        fs::create_directories(_WorkspacePath / "logs");
        fs::create_directories(_WorkspacePath / "meas");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    // Initialize the logger
    try {
        _Logger.init(output_path / "logs", _RunName + ".log", _RunName + "LOG", true);
    } catch (const std::exception& Exept) {
        std::cerr << Exept.what() << '\n';
        exit(EXIT_FAILURE);
    }

    std::stringstream LogMessage;

    // Log that the folder and loggind stuff has bee initialized properly
    LogMessage << IO::LI_time() << "MetropolisWorker - Initialization started...\n"
               << IO::LI_void() << "    main oputput folder: " << _WorkspacePath.string() << "\n"
               << IO::LI_void() << "    measurements folder: " << (_WorkspacePath / "meas").string() << "\n";
    _Logger << LogMessage;

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
    // clang-format off
    LogMessage << IO::LI_void() << "            output file: " << _Hdf5OutputFile.string() << "\n"
               << IO::LI_void() << "MetropolisWorker - Action-info\n"
               << IO::LI_void() << "                  name : " << _Action.action_name() << "\n"
               << IO::LI_void() << "            parameters : " << _Action.action_parameters() << "\n"
               << IO::LI_void() << "               lattice : " << IO::print(sizes) << "\n";
    // clang-format on
    _Logger << LogMessage;

    // initialize the action
    if (!RealValue<ActionType>) {
        LogMessage << IO::LI_warn() << "This is a complex-valued action -> Performing a phase-quenched simulation!!\n";
    }

    // Initialize the lattice
    _Field.init(sizes);

    // RNGs stuff
    _Rng.seed(seed);
    _Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    _UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    _Norm = std::normal_distribution<double>(0.0, 1.0);

    // Log stuff
    LogMessage << IO::LI_time() << "MetropolisWorker - Initialization completed in "
               << std::format("{:.3f}", _T.elapsed_ms()) << " ms\n";
    //    << IO::LI_void() << "     memory allocated : " << IO::pretty_bytes(memoryReport()) << std::endl;
    _Logger << LogMessage;
}

template <class Action>
void MetropolisWorker<Action>::randomizeField(double scale) {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(_Field[Site], scale, _Norm, _Rng);
    }
}

template <class Action>
void MetropolisWorker<Action>::resetField() {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        _Field[Site] = FieldType(0.0);
    }
}

template <class Action>
void MetropolisWorker<Action>::sweep() {
    uint Acc = 0;  // acceptance

    ActionType SVar;               // local action variation
    ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    FieldType FieldVar;  // local field variation

    // Loop over the entire lattice
    uintvect<Action::Dims> Sizes = _Field.getSizes();
    uintvect<Action::Dims> Coord;
    std::fill(Coord.begin(), Coord.end(), 0);
    for (uint Site = 0; Site < _Field.getNsites(); advance_coord(Sizes, Coord), Site++) {
        // Generate a randomized local field variation
        randomize(FieldVar, 0.2, _UnifC, _Rng);

        // Compute the associated action variation
        SVar = _Action.compute_dS_loc(_Field, FieldVar, Coord);

        // Metropolis acceptance check
        if (exp(-SVar.real()) > _Unif(_Rng)) {
            Acc++;
            _Field[Coord] += FieldVar;
            SVarTot += SVar;
            _McStats.S_re += SVar.real();
            _McStats.S_im += SVar.imag();
        }
    }

    // Save the action status after the sweep to _McStats
    _McStats.acceptance = static_cast<double>(Acc) / _Field.getNsites();
    _McStats.dS_re = SVarTot.real();
    _McStats.dS_im = SVarTot.imag();
}

template <class Action>
void MetropolisWorker<Action>::thermalize(uint nSteps) {
    // Reset the timer
    _T.reset();

    // Initialize the value of the action
    ActionType SInit = _Action.compute_S(_Field);
    _McStats.S_re = SInit.real();
    _McStats.S_im = SInit.imag();

    // Perform the thermalization sweeps
    for (uint Iter = 0; Iter < nSteps; Iter++) {
        sweep();
    }

    // Log timing information
    _Logger.log_timing(_RunName, std::format("Performed {:8>d} thermalization steps", nSteps), _T.elapsed_ms());
}

template <class Action>
void MetropolisWorker<Action>::run(uint nMC, uint nTherm, uint MeasureStep) {
    // Log Run parameters
    _Logger.log_string(_RunName, "Starting Monte Carlo updates..");
    _Logger.log_threadig(_RunName, omp_get_max_threads());

    // do checks on nMC and MeasureStep -> sanitize nMC from unnecessary iterations

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
    _McStats.S_re = SInit.real();
    _McStats.S_im = SInit.imag();

    // simulation workflow for indefinite end
    if (nMC == 0) {
        _Logger.log_string(_RunName, "    Total Updates: inf (Soft exit via single Ctrl+C)");
        _Logger.log_string(_RunName, std::format("      Mesure skip: {}", MeasureStep));

        // Set the SIGINT handler to perform a SoftExit
        // [Ctrl+C]x1 -> stop simulation on the next iteration
        // [Ctrl+C]x2 -> immediate stop, possible datacorruption if that
        //               happpens during data output
        SignalHandler::set_SIGINT_handler(SignalHandler::SIGINT_SoftExit);

        // keep track if the iteration number
        unsigned long Iteration = 0;

        while (!SignalHandler::SoftexitRequested) {
            // perform a sweep
            sweep();
            // measure if this is a MeasureStep-th iteration
            if (Iteration % MeasureStep == 0) {
                Stats.push_back(_McStats);
                Obs.push_back(_Action.Measure(_Field));
                // if the buffer are full save to file and flush
                if (Obs.size() == _MaxBufferSize) {
                    // save data
                    save_Measurements(Stats, Obs);
                    // clear the vectors
                    Stats.clear();
                    Obs.clear();
                }
            }
            Iteration++;
        }

        // save the last measurements in the buffer and flush
        if (Stats.size() != 0 && Obs.size() != 0) {
            save_Measurements(Stats, Obs);
            Stats.clear();
            Obs.clear();
        }
        _Logger.log_timing(_RunName, std::format("Performed {:8>d} MonteCarlo steps", Iteration), _T.elapsed_ms());

    } else {
        _Logger.log_string(_RunName, std::format("    Total Updates: {}", nMC));
        _Logger.log_string(_RunName, std::format("      Mesure skip: {}", MeasureStep));
        for (uint Iter = 0; Iter < nMC; Iter++) {
            std::cout << Iter << '\n';
            // perform a sweep
            sweep();
            usleep(100000);
            // measure if this is a MeasureStep-th iteration
            if (Iter % MeasureStep == 0) {
                Stats.push_back(_McStats);
                Obs.push_back(_Action.Measure(_Field));
                // if the buffer are full save to file and flush
                if (Obs.size() == _MaxBufferSize) {
                    // save data
                    save_Measurements(Stats, Obs);
                    // clear the vectors
                    Stats.clear();
                    Obs.clear();
                }
            }
        }

        // save the last measurements in the buffer and flush

        save_Measurements(Stats, Obs);
        Stats.clear();
        Obs.clear();

        _Logger.log_timing(_RunName, std::format("Performed {:8>d} MonteCarlo steps", nMC), _T.elapsed_ms());
    }
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
void MetropolisWorker<Action>::save_Measurements(McDataVectType& McVect, ObsVectType& ObsVect) {
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

template <class Action>
void MetropolisWorker<Action>::log_MC(uint iter) {
    _Logger.log_string(_RunName, std::format(" MC data_dump {} -> ", iter) + _McStats.dump_str());
}

template <class Action>
void MetropolisWorker<Action>::log_obs(uint iter, ObsType& obs) {
    _Logger.log_string(_RunName, std::format("obs data_dump {} -> ", iter) + obs.dump_str());
}
}  // namespace reticolo::montecarlo
