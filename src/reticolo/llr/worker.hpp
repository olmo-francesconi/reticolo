/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/worker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Cpp.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <random>
#include <string>
#include <vector>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/random.hpp"

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

    /* hdf5 stuff*/
    const H5::CompType _McCompType = montecarlo::data<typename Action::ActionType>::make_hdf5_CompType();
    const H5::CompType _ObsCompType = Action::make_obs_hdf5_CompType();

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

    void init(const fs::path& output_path, uint id, uintvect<Action::Dims> sizes, uint seed, Action& action, double ak,
              double Sk, double width, LOG_mode log_mode = LOG_mode::silent);

    /* Randomize the field (hot start)*/
    void randomizeField(double scale = 1.0);

    /* Clear the field (cold start)*/
    void resetField();

    /* Performs a single sweep of the lattice (updates worker::MC_stats) */
    void MCMC_sweep();

    /* Performs nSteps thermalization steps and no measurements */
    void MonteCarlo_thermalize(uint nSteps, std::string run_name, std::string run_id);

    /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
    auto MonteCarlo_run(uint nMC, std::string run_name, std::string run_id)
        -> montecarlo::data<typename Action::ActionType>;

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
                             Action& action, double ak, double Sk, double width, LOG_mode log_mode) {
    // Initialize the timer
    _T.reset();

    // Initialize metadata
    _OutPath = output_path;
    _Id = id;
    _Name = std::format("llr_worker[{:0>3d}]", _Id);

    // Initialize the logger
    _LogStatus = log_mode;
    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.init(output_path / "logs", _Name + ".log", _Name + "LOG", false);
    }

    // Initialize the output file
#pragma omp critical
    if (_LogStatus == LOG_mode::file_only || _LogStatus == LOG_mode::all) {
        try {
            H5::H5File File(_OutPath / "meas" / (_Name + ".h5"), H5F_ACC_TRUNC);
            // write attributes
        } catch (H5::Exception& Exep) {
            exit(EXIT_FAILURE);
        }
    }
    // Initialize the lattice
    _Field.init(sizes);

    // Initialize the action
    _Action = action;
    _Action.lattice_sync(_Field);

    // Initialize LLR variables
    _Ak = ak;
    _Sk = Sk;
    _Width = width;

    // RNGs stuff
    _Rng.seed(rng_seed);
    _Unif = std::uniform_real_distribution<double>(0.0, 1.0);
    _UnifC = std::uniform_real_distribution<double>(-1.0, 1.0);
    _Norm = std::normal_distribution<double>(0.0, 1.0);

    double Elapsed = _T.elapsed_ms();

    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_memory(_Name, "Allocated", memoryReport());
        _Logger.log_string(_Name, std::format("Action: \"{}\" ({})", action.action_name(), action.action_parameters()));
        _Logger.log_string(_Name, std::format("LLR: Sk: {:>6f} width: {:>6f}", _Sk, _Width));
        _Logger.log_timing(_Name, "Initialization done", Elapsed);
    }
}

template <class Action>
void LLRWorker<Action>::randomizeField(double scale) {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(_Field[Site], scale, _Norm, _Rng);
    }
}

template <class Action>
void LLRWorker<Action>::resetField() {
    for (size_t Site = 0; Site < _Field.getNsites(); Site++) {
        _Field[Site] = Action::FieldType(0.0);
    }
}

template <class Action>
void LLRWorker<Action>::MCMC_sweep() {
    uint Acc = 0;  // acceptance

    double WindowWeight;  // weight of the windowing function   [normal distribution]
    double LlrWeight;     // weight of the llr reweighting      [exp(-ak(S-Sk)]

    typename Action::ActionType SVar;               // local action variation
    typename Action::ActionType SVarTot(0.0, 0.0);  // cumulative action variation

    typename Action::FieldType FieldVar;  // local field variation

    for (uint Site = 0; Site < _Field.getNsites(); Site++) {
        randomize(FieldVar, 0.2, _UnifC, _Rng);

        SVar = _Action.compute_dS_loc(_Field, FieldVar, Site);

        WindowWeight = (SVar.imag() * (SVar.imag() + 2.0 * _McStats._SIm - 2.0 * _Sk)) / (2.0 * _Width * _Width);
        LlrWeight = _Ak * SVar.imag();

        if (exp(-(SVar.real() + WindowWeight + LlrWeight)) > _Unif(_Rng)) {
            Acc++;
            _Field[Site] += FieldVar;
            SVarTot += SVar;
            _McStats._SRe += SVar.real();
            _McStats._SIm += SVar.imag();
        }
    }

    _McStats._Acceptance = static_cast<double>(Acc) / _Field.getNsites();
    _McStats._DSRe = SVarTot.real();
    _McStats._DSIm = SVarTot.imag();
}

template <class Action>
void LLRWorker<Action>::MonteCarlo_thermalize(uint nSteps, std::string run_name, std::string run_id) {
    _T.reset();

    typename Action::ActionType SInit = _Action.compute_S(_Field);

    _McStats._SRe = SInit.real();
    _McStats._SIm = SInit.imag();

    for (uint Iter = 0; Iter < nSteps; Iter++) {
        MCMC_sweep();
    }
    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_timing(
            _Name, std::format("{:<16s} : {:>8s} : Performed {:8>d} thermalization steps", run_name, run_id, nSteps),
            _T.elapsed_ms());
    }
}

template <class Action>
auto LLRWorker<Action>::MonteCarlo_run(uint nMC, std::string run_name, std::string run_id)
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
    _McStats._SRe = SInit.real();
    _McStats._SIm = SInit.imag();

    for (uint Iter = 0; Iter < nMC; Iter++) {
        MCMC_sweep();

        Stats.push_back(_McStats);

        Obs.push_back(_Action.Measure(_Field));

        Res += _McStats;
    }

    Res /= static_cast<double>(nMC);

#pragma omp critical
    if (_LogStatus == LOG_mode::file_only || _LogStatus == LOG_mode::all) {
        try {
            // Create the file
            H5::H5File File(_OutPath / "meas" / (_Name + ".h5"), H5F_ACC_RDWR);
            // Check that the full path exists
            if (!File.nameExists(run_name)) {
                H5::Group RunGroup = File.createGroup("/" + run_name);
            }
            H5::Group Group = File.createGroup("/" + run_name + "/" + run_id);

            // Create dataspace
            std::array<hsize_t, 1> Entries = {nMC};
            H5::DataSpace          Space(1, Entries.data());

            // Create datatype and write the observables dataset
            H5::DataSet ObsDataset = File.createDataSet("/" + run_name + "/" + run_id + "/obs", _ObsCompType, Space);
            ObsDataset.write(Obs.data(), _ObsCompType);

            // Create datatype and write the observables dataset
            H5::DataSet McDataset = File.createDataSet("/" + run_name + "/" + run_id + "/mc", _McCompType, Space);
            McDataset.write(Stats.data(), _McCompType);
        } catch (H5::Exception& Exep) {
            exit(EXIT_FAILURE);
        }
    }

    if (_LogStatus == LOG_mode::log_only || _LogStatus == LOG_mode::all) {
        _Logger.log_timing(_Name,
                           std::format("{:<16s} : {:>8s} : Performed {:8>d} MonteCarlo steps", run_name, run_id, nMC),
                           _T.elapsed_ms());
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
    _Logger.log_string(_Name, std::format("MC data_dump ->\t{}\t{},", run_id, iter) + _McStats.dump_str());
}

template <class Action>
void LLRWorker<Action>::log_obs(std::string run_id, uint iter, Action::Observables& obs) {
    _Logger.log_string(_Name, std::format("obs data_dump ->\t{}\t{},", run_id, iter) + obs.dump_str());
}

}  // namespace reticolo::LLR
