/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/worker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <filesystem>
#include <random>

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/tools/types/basic.hpp"
#include "reticolo/tools/types/random.hpp"

namespace fs = std::filesystem;

namespace reticolo::LLR {
template <class Action>
class worker {
 private:
  /* Metadata and output */
  uint _id;
  std::string _name;
  std::string _run_id;
  fs::path _out_path;
  IO::LOG_mode _log_status;

  /* Physics stuff */
  Action _action;
  lattice<typename Action::FieldType, Action::Dims> field;
  montecarlo::data<typename Action::ActionType> MC_stats;

  /* RNG stuff */
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> unif;
  std::uniform_real_distribution<double> unif_c;
  std::normal_distribution<double> norm;

  /* Timing and logging */
  Timer T;
  IO::Logger logger;

  /* LLR parameters */
  double _ak, _Sk, _width;

 public:
  /* Default constructor (does nothing) */
  worker(){};

  /* Initializer function -> actually does all the set-up */
  void init(const fs::path& output_path, uint id, uintvect<Action::Dims> sizes, uint seed, Action action, double Sk,
            double width, IO::LOG_mode log_mode = IO::LOG_mode::silent);

  /* Randomize the field (hot start)*/
  void randomizeField(double scale = 1.0);

  /* Performs a single sweep of the lattice (updates worker::MC_stats) */
  void MCMC_sweep();

  /* Performs nSteps thermalization steps and no measurements */
  void MonteCarlo_thermalize(uint nSteps, const std::string& run_id);

  /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with
   * the averages */
  auto MonteCarlo_run(uint nMC, std::string run_id) -> montecarlo::data<typename Action::ActionType>;

  /* Set the value of the LLR ak parameters*/
  void set_ak(double a) { _ak = a; };

  /* Various logging utilities */
  auto memoryReport() -> size_t;
  void log_MC(std::string run_id, uint iter);
  void log_obs(std::string run_id, uint iter, Action::observables& obs);
};

template <class Action>
void worker<Action>::init(const fs::path& output_path, uint id, uintvect<Action::Dims> sizes, uint rng_seed,
                          Action action, double Sk, double width, IO::LOG_mode log_mode) {
  Timer T;

  _id = id;
  _name = std::format("llr_worker[{:0>3d}]", _id);
  _out_path = output_path;

  _log_status = log_mode;

  // Initialize the logger
  if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
    logger.init(output_path / "logs", _name + ".log", _name + "LOG");
  }

  // Initialize the output file
  if (_log_status == IO::LOG_mode::file_only || _log_status == IO::LOG_mode::all) {
#pragma omp critical
    {
      try {
        H5::Exception::dontPrint();
        H5::H5File File(_out_path / "meas" / (_name + ".h5"), H5F_ACC_TRUNC);
        // write attributes
      } catch (H5::Exception& e) {
        e.printErrorStack();
        exit(EXIT_FAILURE);
      }
    }
  }
  // Initialize the lattice
  field.init(sizes);

  // initialize the action
  _action = action;

  // RNGs stuff
  rng.seed(rng_seed);
  unif = std::uniform_real_distribution<double>(0.0, 1.0);
  unif_c = std::uniform_real_distribution<double>(-1.0, 1.0);
  norm = std::normal_distribution<double>(0.0, 1.0);

  // initialize the ak parameter
  _ak = 0.0;
  _Sk = Sk;
  _width = width;

  double Elapsed = T.elapsed_ms();

  if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
    logger.log_memory(_name, "Allocated", memoryReport());
    logger.log_string(_name, std::format("Action: \"{}\" ({})", action.action_name(), action.action_parameters()));
    logger.log_string(_name, std::format("LLR: Sk: {:>6f} width: {:>6f}", _Sk, _width));
    logger.log_timing(_name, "Initialization done", Elapsed);
  }
};

template <class Action>
void worker<Action>::randomizeField(double scale) {
  for (size_t Site = 0; Site < field.getNsites(); Site++) {
    randomize(field[Site], scale, norm, rng);
  }
}

template <class Action>
void worker<Action>::MCMC_sweep() {
  uint Acc = 0;

  double WindowWeight;
  double LlrWeight;

  typename Action::ActionType dS;                // local action variation
  typename Action::ActionType dS_tot(0.0, 0.0);  // cumulative action variation

  typename Action::FieldType dField;  // local field variation

  uintvect<Action::Dims> Sizes = field.getSizes();
  uintvect<Action::Dims> Coord;
  std::fill(Coord.begin(), Coord.end(), 0);
  for (uint Site = 0; Site < field.getNsites(); advance_coord(Sizes, Coord), Site++) {
    randomize(dField, 0.2, unif_c, rng);

    dS = _action.compute_dS_loc(field, dField, Coord);

    WindowWeight = (dS.imag() * (dS.imag() + 2.0 * MC_stats.S_im - 2.0 * _Sk)) / (2.0 * _width * _width);
    LlrWeight = _ak * dS.imag();

    if (exp(-(dS.real() + WindowWeight + LlrWeight)) > unif(rng)) {
      Acc++;
      field[Coord] += dField;
      dS_tot += dS;
      MC_stats.S_re += dS.real();
      MC_stats.S_im += dS.imag();
    }
  }

  MC_stats.acceptance = static_cast<double>(Acc) / field.getNsites();
  MC_stats.dS_re = dS_tot.real();
  MC_stats.dS_im = dS_tot.imag();
}

template <class Action>
void worker<Action>::MonteCarlo_thermalize(uint nSteps, const std::string& run_id) {
  T.reset();

  typename Action::ActionType S = _action.compute_S(field);

  MC_stats.S_re = S.real();
  MC_stats.S_im = S.imag();

  for (uint Iter = 0; Iter < nSteps; Iter++) {
    MCMC_sweep();

    if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
      log_MC(run_id + "_therm", Iter);
    }
  }
  if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
    logger.log_timing(_name, std::format("Performed {} thermalization steps", nSteps), T.elapsed_ms());
  }
}

template <class Action>
auto worker<Action>::MonteCarlo_run(uint nMC, std::string run_id) -> montecarlo::data<typename Action::ActionType> {
  T.reset();

  montecarlo::data<typename Action::ActionType> Res;

  // Initialize std::vector of Monte Carlo stats
  std::vector<montecarlo::data<typename Action::ActionType>> Stats;
  Stats.clear();
  Stats.reserve(nMC);

  // Initialize std::vector of observable measurements
  std::vector<typename Action::observables> Obs;
  Obs.clear();
  Obs.reserve(nMC);

  // Initialize the starting value of S
  typename Action::ActionType S = _action.compute_S(field);
  MC_stats.S_re = S.real();
  MC_stats.S_im = S.imag();

  for (uint Iter = 0; Iter < nMC; Iter++) {
    MCMC_sweep();

    Stats.push_back(MC_stats);

    Obs.push_back(_action.Measure(field));

    Res += MC_stats;

    if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
      log_MC(run_id, Iter);
      log_obs(run_id, Iter, Obs.back());
    }
  }

  Res /= static_cast<double>(nMC);

  // Write to HDF5 file
  if (_log_status == IO::LOG_mode::file_only || _log_status == IO::LOG_mode::all) {
#pragma omp critical
    {
      try {
        H5::Exception::dontPrint();

        // Create the file
        H5::H5File File(_out_path / "meas" / (_name + ".h5"), H5F_ACC_RDWR);
        H5::Group Group = File.createGroup("/" + run_id);

        // Create dataspace
        hsize_t Entries[] = {nMC};
        H5::DataSpace Space(1, Entries);

        // Create datatype and write the observables dataset
        H5::CompType ObsType(sizeof(typename Action::observables));
        _action.make_hdf5_CompType(ObsType);
        H5::DataSet ObsDataset = File.createDataSet("/" + run_id + "/obs", ObsType, Space);
        ObsDataset.write(Obs.data(), ObsType);

        // Create datatype and write the observables dataset
        H5::CompType McType = montecarlo::make_mc_data_hdf5_CompType<typename Action::ActionType>();
        H5::DataSet McDataset = File.createDataSet("/" + run_id + "/mc", McType, Space);
        McDataset.write(Stats.data(), McType);
      } catch (H5::Exception& e) {
        e.printErrorStack();
        exit(EXIT_FAILURE);
      }
    }
  }

  if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all) {
    logger.log_timing(_name, std::format("Performed {} MonteCarlo steps", nMC), T.elapsed_ms());
  }

  return Res;
}

template <class Action>
size_t worker<Action>::memoryReport() {
  size_t Memory = 0;
  Memory += field.memoryReport();

  return Memory;
}

template <class Action>
void worker<Action>::log_MC(std::string run_id, uint iter) {
  logger.log_string(_name,
                    std::format("MC data_dump ->\t{}\t{},{:+8e},{:+8e},{:+8e},{:+8e},{:+8e},", run_id, iter,
                                MC_stats.acceptance, MC_stats.S_re, MC_stats.S_im, MC_stats.dS_re, MC_stats.dS_im));
}

template <class Action>
void worker<Action>::log_obs(std::string run_id, uint iter, Action::observables& obs) {
  logger.log_string(_name, std::format("obs data_dump ->\t{}\t{},", run_id, iter) + obs.dump_str());
}

}  // namespace reticolo::LLR
