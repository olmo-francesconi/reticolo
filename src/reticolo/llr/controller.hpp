/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/controller.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <filesystem>
#include <vector>

#include "reticolo/llr/worker.hpp"
#include "reticolo/tools/types/basic.hpp"
namespace fs = std::filesystem;

namespace reticolo::LLR {
template <class Action>
class controller {
 private:
  std::vector<worker<Action>> workers;
  std::random_device rd;
  fs::path output_path;
  uint nMC, nNR, nRM;

  Action action;

  std::vector<double> ak_vect;
  std::vector<std::vector<double>> ak_hist;
  std::vector<double> Sk_vect;
  double deltaS;

  void NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo);
  void RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo);
  void ReplicaExcange();

 public:
  controller(Action& act) : action(act){};

  void init(const fs::path& out_path, uintvect<Action::Dims> lattice_size, uint nWorkers, double scale,
            IO::LOG_mode MC_log_mode = IO::LOG_mode::silent);

  void run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, const std::string& run_id, uint replicas);

  auto memoryReport() -> size_t;
};

template <class Action>
void controller<Action>::init(const fs::path& out_path, uintvect<Action::Dims> lattice_size, uint nWorkers,
                              double scale, IO::LOG_mode MC_log_mode) {
  Timer T;

  // Initialize the folder structure
  try {
    output_path = fs::absolute(out_path);

    fs::create_directory(output_path);
    output_path = fs::canonical(output_path);
    fs::create_directories(output_path / "llr" / "logs");
    fs::create_directories(output_path / "llr" / "meas");
    // fs::create_directory(output_path / "llr" / "conf");
    // IO::InfoLogger.init(output_path / "logs", "info.log");
    // IO::ErrorLogger.init(output_path / "logs", "err.log");
    IO::GlobalLogger.init(output_path, "reticolo.log");
    // log.init(output_path / "logs", "llr_controller.log",
    // "llr_controllerLOG");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    exit(EXIT_FAILURE);
  }

  std::stringstream LogMessage;

  // Log that the folder and loggind stuff has bee initialized properly
  LogMessage << IO::LogLineInit_time() << "llr_controller - Initialization started...\n"
             << IO::LogLineInit_void() << "                     main oputput folder: " << output_path.string() << "\n"
             << IO::LogLineInit_void()
             << "                     measurements folder: " << (output_path / "llr" / "meas").string() << "\n"
             << IO::LogLineInit_void()
             << "                             logs folder: " << (output_path / "llr" / "logs").string() << '\n';
  IO::GlobalLogger << LogMessage;

  // Initialize the output file
  try {
    H5::Exception::dontPrint();
    auto* File = new H5::H5File(output_path / "llr" / "meas" / "llr.h5", H5F_ACC_TRUNC);
    // write attributes
    delete File;
  } catch (H5::Exception& e) {
    e.printErrorStack();
    exit(EXIT_FAILURE);
  }

  // Log stuff
  LogMessage << IO::LogLineInit_void()
             << "                          ak output file: " << (output_path / "llr" / "meas" / "llr.h5").string()
             << "\n"
             << IO::LogLineInit_void() << "llr_controller - Action-info\n"
             << IO::LogLineInit_void() << "                           name : " << action.action_name() << "\n"
             << IO::LogLineInit_void() << "                     parameters : " << action.action_parameters() << "\n"
             << IO::LogLineInit_time() << "llr_controller - Workers initialization started...\n"
             << IO::LogLineInit_void() << "                     workers #: " << nWorkers << "\n"
             << IO::LogLineInit_void() << "                    deltaS / V: " << scale << "\n"
             << IO::LogLineInit_void() << "                        deltaS: " << scale * (double)get_volume(lattice_size)
             << std::endl;
  IO::GlobalLogger << LogMessage;

  // Initialize Sk values
  deltaS = scale * (double)get_volume(lattice_size);
  Sk_vect.resize(nWorkers);
  for (int i = 0; i < nWorkers; i++) {
    Sk_vect[i] = deltaS * ((double)i + 0.5);
  }

  // Initialize ak values
  ak_vect.resize(nWorkers);
  std::fill(ak_vect.begin(), ak_vect.end(), 0.0);

  // Initialize workers
  workers.resize(nWorkers);
#pragma omp parallel for schedule(static, 1)
  for (uint WorkerId = 0; WorkerId < nWorkers; WorkerId++) {
    workers[WorkerId].init(out_path / "llr", WorkerId, lattice_size, rd(), action.p, Sk_vect[WorkerId], deltaS,
                           MC_log_mode);
    workers[WorkerId].set_ak(ak_vect[WorkerId]);
  }

  // Log stuff
  LogMessage << IO::LogLineInit_time() << "llr_controller - Initialization completed in " << T.elapsed_ms() << " ms\n"
             << IO::LogLineInit_void()
             << "                 Total memory allocated: " << IO::pretty_bytes(memoryReport()) << std::endl;
  IO::GlobalLogger << LogMessage;
}

template <class Action>
void controller<Action>::NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo) {
  // Newton-Raphson Iterations
  for (int Step = 0; Step < nNewton_Raphson; Step++) {
#pragma omp parallel for schedule(static, 1)
    for (int WorkerId = 0; WorkerId < workers.size(); WorkerId++) {
      montecarlo::data<typename Action::ActionType> Res;

      // MonteCarlo_thermalize to the newly updated value of ak
      workers[WorkerId].MonteCarlo_thermalize(100, std::format("NR_{}", Step));

      // run the montecarlo steps
      Res = workers[WorkerId].MonteCarlo_run(nMonte_Carlo, std::format("NR_{}", Step));

      // update the ak value
      ak_vect[WorkerId] += -(Sk_vect[WorkerId] - Res.S_im) / (deltaS * deltaS);
      workers[WorkerId].set_ak(ak_vect[WorkerId]);
      //       if (log_mode == IO::LOG_mode::log_only || log_mode == IO::LOG_mode::all)
      // #pragma omp critical
      //         log.log_string("llr-controller", std::format("NR_{}_{}", step, i));
    }

    ak_hist.push_back(ak_vect);
    // replica excange
  }
}

template <class Action>
void controller<Action>::RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo) {
  // Robbins-Monro Iterations
  for (int Step = 0; Step < nRobbins_Monro; Step++) {
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < workers.size(); i++) {
      montecarlo::data<typename Action::ActionType> Res;

      // MonteCarlo_thermalize to the newly updated value of ak
      workers[i].MonteCarlo_thermalize(100, std::format("RM_{}", Step));

      // run the montecarlo steps
      Res = workers[i].MonteCarlo_run(nMonte_Carlo, std::format("RM_{}", Step));

      // update the ak value
      ak_vect[i] += -(Sk_vect[i] - Res.S_im) / (static_cast<double>(Step + 1) * (deltaS * deltaS));
      workers[i].set_ak(ak_vect[i]);
      //                     if (log_mode == IO::LOG_mode::log_only || log_mode
      //                     == IO::LOG_mode::all)
      // #pragma omp critical
      //                         log.log_string("llr-controller",
      //                         std::format("RM_{}_{}", step, i));
    }

    ak_hist.push_back(ak_vect);
    // replica excange
  }
}

template <class Action>
void controller<Action>::ReplicaExcange() {}

template <class Action>
void controller<Action>::run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, const std::string& run_id,
                             uint replicas) {
  for (uint i = 0; i < replicas; i++) {
    std::string RunName = run_id + std::format("{}", i);

    // Initialize ak values
    std::fill(ak_vect.begin(), ak_vect.end(), 0.0);
    ak_hist.clear();
    ak_hist.push_back(ak_vect);
    for (uint i = 0; i < ak_vect.size(); i++) {
      workers[i].set_ak(ak_vect[i]);
    }

// initial thermalization
#pragma omp parallel for schedule(static, 1)
    for (auto& Worker : workers) {
      // w.randomizeField(0.1);

      // call imaginary action thermalization
      Worker.MonteCarlo_thermalize(1000, "burn-in");
    }

    NewtonRaphson(nNewton_Raphson, nMonte_Carlo);

    RobbinsMonro(nRobbins_Monro, nMonte_Carlo);

    // File output
    try {
      H5::Exception::dontPrint();

      // Create the file and group
      H5::H5File File(output_path / "llr" / "meas" / "llr.h5", H5F_ACC_RDWR);
      H5::Group Group = File.createGroup("/" + RunName);

      // Create dataspace
      hsize_t Entries[] = {ak_hist.size()};
      H5::DataSpace Dataspace(1, Entries);

      // Create datatype
      H5::FloatType Datatype = H5::PredType::NATIVE_DOUBLE;

      for (uint i = 0; i < workers.size(); i++) {
        std::vector<double> Data;
        Data.reserve(ak_hist.size());
        for (const auto& el : ak_hist) {
          Data.push_back(el[i]);
        }

        // Create dataset and write the observables dataset
        H5::DataSet Dataset =
            File.createDataSet("/" + RunName + "/ak" + std::format("_{:0>3d}", i), Datatype, Dataspace);
        Dataset.write(Data.data(), Datatype);
      }
    } catch (H5::Exception& e) {
      e.printErrorStack();
      exit(EXIT_FAILURE);
    }
  }
}

template <class Action>
auto controller<Action>::memoryReport() -> size_t {
  size_t Memory = 0;
  for (auto& w : workers) {
    Memory += w.memoryReport();
  }

  return Memory;
}

}  // namespace reticolo::LLR
