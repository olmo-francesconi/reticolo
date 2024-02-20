/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/worker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <random>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/tools/types/core.hpp"
#include "reticolo/tools/types/random.hpp"

namespace fs = std::filesystem;

namespace reticolo::LLR {

/*--------------------------------------------------------------------------------------------------
    LLRWorker Class Declaration
--------------------------------------------------------------------------------------------------*/

template <class Action>
class LLRWorker {
  private:
    /* Metadata and output */
    uint        _Id;         // Numerical identifier of the worker
    std::string _Name;       // Full LLRWorker name (llr_worker[###])
    std::string _RunId;      // Run identifier shared by all workers
    fs::path    _OutPath;    // Output path
    LOG_mode    _LogStatus;  // Private Logger status

    /* Physics stuff */
    Action                                            _Action;   // Action
    Lattice<typename Action::FieldType, Action::Dims> _Field;    // Lattice of the type specified by action
    montecarlo::data<typename Action::ActionType>     _McStats;  // MonteCarlo statisctics

    /* RNG stuff */
    std::mt19937_64                        _Rng;    // Random Number Generator
    std::uniform_real_distribution<double> _Unif;   // Uniform distribution [0.0, 1.0]
    std::uniform_real_distribution<double> _UnifC;  // Uniform distribution [-1.0, 1.0]
    std::normal_distribution<double>       _Norm;   // Normal distibution (mean: 0.0, stddev: 1.0 )

    /* LLR parameters */
    double _Ak;     // Current reweighting parameter
    double _Sk;     // Center of the windowing function
    double _Width;  // Width of the windowing function

    /* Timing and logging */
    Timer      _T;       // LLRWorker Private Timer
    IO::Logger _Logger;  // LLRWorker Private Logger

    /* Various logging utilities */
    void log_MC(std::string run_id, uint iter);
    void log_obs(std::string run_id, uint iter, Action::Observables& obs);

  public:
    /* Default constructor (does nothing) */
    LLRWorker() = default;

    /* Initializer function -> actually does all the set-up */
    void init(const fs::path& output_path, uint id, uintvect<Action::Dims> sizes, uint seed, Action action, double Sk,
              double width, LOG_mode log_mode = LOG_mode::silent);

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Performs a single sweep of the lattice (updates worker::MC_stats) */
    void MCMC_sweep();

    /* Performs nSteps thermalization steps and no measurements */
    void MonteCarlo_thermalize(uint nSteps, const std::string& run_id);

    /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
    auto MonteCarlo_run(uint nMC, const std::string& run_id) -> montecarlo::data<typename Action::ActionType>;

    /* Set the value of the LLR ak parameters*/
    void set_ak(double ak) { _Ak = ak; };

    /* Memory report */
    auto memoryReport() -> size_t;
};

/*--------------------------------------------------------------------------------------------------
    Public Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
void LLRWorker<Action>::init(const fs::path& output_path, uint id, uintvect<Action::Dims> sizes, uint rng_seed,
                             Action action, double Sk, double width, LOG_mode log_mode) {
    _T.reset();

    _Id = id;
    _Name = std::format("llr_worker[{:0>3d}]", _Id);
    _OutPath = output_path;

    _LogStatus = log_mode;

    // Initialize the logger
    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.init(output_path / "logs", _Name + ".log", _Name + "LOG", false);
    }

    // Initialize the output file
    if (_LogStatus == LOG_mode::file_only || _LogStatus == LOG_mode::all) {
#pragma omp critical
        {
            try {
                H5::Exception::dontPrint();
                H5::H5File File(_OutPath / "meas" / (_Name + ".h5"), H5F_ACC_TRUNC);
                // write attributes
            } catch (H5::Exception& Exep) {
                H5::Exception::printErrorStack();
                exit(EXIT_FAILURE);
            }
        }
    }
    // Initialize the lattice
    _Field.init(sizes);

    // initialize the action
    _Action = action;

    // RNGs stuff
    _Rng.seed(rng_seed);
    _Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    _UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    _Norm = std::normal_distribution<double>(0.0, 1.0);

    // initialize the ak parameter
    _Ak = 0.0;
    _Sk = Sk;
    _Width = width;

    double Elapsed = _T.elapsed_ms();

    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_memory(_Name, "Allocated", memoryReport());
        _Logger.log_string(_Name, std::format("Action: \"{}\" ({})", action.action_name(), action.action_parameters()));
        _Logger.log_string(_Name, std::format("LLR: Sk: {:>6f} width: {:>6f}", _Sk, _Width));
        _Logger.log_timing(_Name, "Initialization done", Elapsed);
    }
};

template <class Action>
void LLRWorker<Action>::randomizeField(double scale) {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(_Field[Site], scale, _Norm, _Rng);
    }
}

template <class Action>
void LLRWorker<Action>::MCMC_sweep() {
    uint Acc = 0;

    double WindowWeight;
    double LlrWeight;

    typename Action::ActionType SVar;               // local action variation
    typename Action::ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    typename Action::FieldType FieldVar;  // local field variation

    uintvect<Action::Dims> Sizes = _Field.getSizes();
    uintvect<Action::Dims> Coord;
    std::fill(Coord.begin(), Coord.end(), 0);
    for (uint Site = 0; Site < _Field.getNsites(); advance_coord(Sizes, Coord), Site++) {
        randomize(FieldVar, 0.2, _UnifC, _Rng);

        SVar = _Action.compute_dS_loc(_Field, FieldVar, Coord);

        WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * _McStats.S_im - 2.0 * _Sk)) / (2.0 * _Width * _Width);
        LlrWeight = _Ak * SVar.imag();

        if (exp(-(SVar.real() + WindowWeight + LlrWeight)) > _Unif(_Rng)) {
            Acc++;
            _Field[Coord] += FieldVar;
            SVarTot += SVar;
            _McStats.S_re += SVar.real();
            _McStats.S_im += SVar.imag();
        }
    }

    _McStats.acceptance = static_cast<double>(Acc) / _Field.getNsites();
    _McStats.dS_re = SVarTot.real();
    _McStats.dS_im = SVarTot.imag();
}

template <class Action>
void LLRWorker<Action>::MonteCarlo_thermalize(uint nSteps, const std::string& run_id) {
    _T.reset();

    typename Action::ActionType SInit = _Action.compute_S(_Field);

    _McStats.S_re = SInit.real();
    _McStats.S_im = SInit.imag();

    for (uint Iter = 0; Iter < nSteps; Iter++) {
        MCMC_sweep();
    }

    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_timing(_Name, std::format("Performed {} thermalization steps", nSteps), _T.elapsed_ms());
    }
}

template <class Action>
auto LLRWorker<Action>::MonteCarlo_run(uint nMC, const std::string& run_id)
    -> montecarlo::data<typename Action::ActionType> {
    _T.reset();

    montecarlo::data<typename Action::ActionType> Res;

    // Initialize std::vector of Monte Carlo stats
    std::vector<montecarlo::data<typename Action::ActionType>> Stats;
    Stats.clear();
    Stats.reserve(nMC);

    // Initialize std::vector of observable measurements
    std::vector<typename Action::Observables> Obs;
    Obs.clear();
    Obs.reserve(nMC);

    // Initialize the starting value of S
    typename Action::ActionType SInit = _Action.compute_S(_Field);
    _McStats.S_re = SInit.real();
    _McStats.S_im = SInit.imag();

    for (uint Iter = 0; Iter < nMC; Iter++) {
        MCMC_sweep();

        Stats.push_back(_McStats);

        Obs.push_back(_Action.Measure(_Field));

        Res += _McStats;
    }

    Res /= static_cast<double>(nMC);

#pragma omp critical
    try {
        H5::Exception::dontPrint();

        // Create the file
        H5::H5File File(_OutPath / "meas" / (_Name + ".h5"), H5F_ACC_RDWR);
        H5::Group  Group = File.createGroup("/" + run_id);

        // Create dataspace
        size_t                 DataRank = 1;
        std::array<hsize_t, 1> Entries = {nMC};
        H5::DataSpace          Space(DataRank, Entries.data());

        // Create datatype and write the observables dataset
        H5::CompType ObsType(sizeof(typename Action::Observables));
        _Action.make_hdf5_CompType(ObsType);
        H5::DataSet ObsDataset = File.createDataSet("/" + run_id + "/obs", ObsType, Space);
        ObsDataset.write(Obs.data(), ObsType);

        // Create datatype and write the observables dataset
        H5::CompType McType = montecarlo::make_mc_data_hdf5_CompType<typename Action::ActionType>();
        H5::DataSet  McDataset = File.createDataSet("/" + run_id + "/mc", McType, Space);
        McDataset.write(Stats.data(), McType);
    } catch (H5::Exception& Exep) {
        H5::Exception::printErrorStack();
        exit(EXIT_FAILURE);
    }

    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_timing(_Name, std::format("Performed {} MonteCarlo steps", nMC), _T.elapsed_ms());
    }

    return Res;
}

template <class Action>
auto LLRWorker<Action>::memoryReport() -> size_t {
    size_t Memory = 0;
    Memory += _Field.memoryReport();

    return Memory;
}

/*--------------------------------------------------------------------------------------------------
    Private Methods Implmentations
--------------------------------------------------------------------------------------------------*/

template <class Action>
void LLRWorker<Action>::log_MC(std::string run_id, uint iter) {
    _Logger.log_string(_Name,
                       std::format("MC data_dump ->\t{}\t{},{:+8e},{:+8e},{:+8e},{:+8e},{:+8e},", run_id, iter,
                                   _McStats.acceptance, _McStats.S_re, _McStats.S_im, _McStats.dS_re, _McStats.dS_im));
}

template <class Action>
void LLRWorker<Action>::log_obs(std::string run_id, uint iter, Action::Observables& obs) {
    _Logger.log_string(_Name, std::format("obs data_dump ->\t{}\t{},", run_id, iter) + obs.dump_str());
}

}  // namespace reticolo::LLR
