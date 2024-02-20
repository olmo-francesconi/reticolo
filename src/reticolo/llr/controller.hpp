/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/controller.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>

#include <algorithm>
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

#include "reticolo/llr/worker.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/tools/types/core.hpp"

namespace fs = std::filesystem;

namespace reticolo::LLR {
////////////////////////////////////////////////////////////////////////////////
// LLRController Class declaration                                            //
////////////////////////////////////////////////////////////////////////////////
template <class Action>
class LLRController {
  private:
    /* Physics */
    Action _Action;

    /* LLR workers vector */
    std::vector<LLRWorker<Action>> _Workers;

    /* Output */
    fs::path _OutputPath;

    /* Main random device*/
    std::random_device _Rd;

    /* LLR parameters and data */
    std::vector<double>              _AkVect;
    std::vector<std::vector<double>> _AkHist;
    std::vector<double>              _SkVect;
    double                           _DeltaS;

    /* Private methods */
    void NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo);
    void RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo);
    void ReplicaExcange();

  public:
    /* Constructor (only sets up the Action) */
    LLRController(Action& act) : _Action(act){};

    /* Initializer function -> actually does all the set-up */
    void init(const fs::path& out_path, uintvect<Action::Dims> lattice_size, uint nWorkers, double scale,
              LOG_mode MC_log_mode = LOG_mode::silent);

    /* Run the actual LLR simulation */
    void run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, const std::string& run_id, uint replicas);

    /* Various logging utilities */
    auto memoryReport() -> size_t;
};

////////////////////////////////////////////////////////////////////////////////
// Public methods implementation                                              //
////////////////////////////////////////////////////////////////////////////////
template <class Action>
void LLRController<Action>::init(const fs::path& out_path, uintvect<Action::Dims> lattice_size, uint nWorkers,
                                 double scale, LOG_mode MC_log_mode) {
    Timer Time;

    // Initialize the folder structure
    try {
        _OutputPath = fs::absolute(out_path);

        fs::create_directories(_OutputPath);
        _OutputPath = fs::canonical(_OutputPath);
        fs::create_directories(_OutputPath / "llr" / "logs");
        fs::create_directories(_OutputPath / "llr" / "meas");

        IO::GlobalLogger.init(_OutputPath, "reticolo.log");
        IO::GlobalLogger.init(_OutputPath, "reticolo.log");

    } catch (const std::exception& Exept) {
        std::cerr << Exept.what() << '\n';
        exit(EXIT_FAILURE);
    }

    std::stringstream LogMessage;

    // Log that the folder and loggind stuff has bee initialized properly
    // clang-format off
    LogMessage << IO::LI_time() << "llr_controller - Initialization started..." << "\n"
             << IO::LI_void() << "    main oputput folder: " << _OutputPath.string() << "\n"
             << IO::LI_void() << "    measurements folder: " << (_OutputPath / "llr" / "meas").string() << "\n"
             << IO::LI_void() << "            logs folder: " << (_OutputPath / "llr" / "logs").string() << '\n';
    IO::GlobalLogger << LogMessage;
    // clang-format on

    // Initialize the output file
    try {
        H5::Exception::dontPrint();
        auto* File = new H5::H5File(_OutputPath / "llr" / "meas" / "llr.h5", H5F_ACC_TRUNC);
        // write attributes
        delete File;
    } catch (H5::Exception& Exep) {
        H5::Exception::printErrorStack();
        exit(EXIT_FAILURE);
    }

    // Log stuff
    // clang-format off
    LogMessage << IO::LI_void() << "        ak output file: " << (_OutputPath / "llr" / "meas" / "llr.h5").string() << "\n"
              << IO::LI_void() << "llr_controller - Action-info\n"
              << IO::LI_void() << "                 name : " << _Action.action_name() << "\n"
              << IO::LI_void() << "           parameters : " << _Action.action_parameters() << "\n"
              << IO::LI_void() << "              lattice : " << IO::print(lattice_size) << "\n"
              << IO::LI_time() << "llr_controller - Workers initialization started...\n"
              << IO::LI_void() << "             workers #: " << nWorkers << "\n"
              << IO::LI_void() << "            deltaS / V: " << scale << "\n"
              << IO::LI_void() << "                deltaS: " << scale * (double)get_volume(lattice_size) << std::endl;
    IO::GlobalLogger << LogMessage;
    // clang-format on

    // Initialize Sk values
    _DeltaS = scale * (double)get_volume(lattice_size);
    _SkVect.resize(nWorkers);
    for (int WorkerId = 0; WorkerId < nWorkers; WorkerId++) {
        _SkVect[WorkerId] = _DeltaS * ((double)WorkerId + 0.5);
    }

    // Initialize ak values
    _AkVect.resize(nWorkers);
    std::fill(_AkVect.begin(), _AkVect.end(), 0.0);

    // Initialize workers
    _Workers.resize(nWorkers);
#pragma omp parallel for schedule(static, 1)
    for (uint WorkerId = 0; WorkerId < nWorkers; WorkerId++) {
        _Workers[WorkerId].init(out_path / "llr", WorkerId, lattice_size, _Rd(), _Action.p, _SkVect[WorkerId], _DeltaS,
                                MC_log_mode);
        _Workers[WorkerId].set_ak(_AkVect[WorkerId]);
    }

    // Log stuff
    // clang-format off
    LogMessage << IO::LI_time() << "llr_controller - Initialization completed in " << IO::print(Time.elapsed_ms()) << " ms\n"
              << IO::LI_void() << "     memory allocated : " << IO::pretty_bytes(memoryReport()) << std::endl;
    IO::GlobalLogger << LogMessage;
    // clang-format on
}

template <class Action>
void LLRController<Action>::run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, const std::string& run_id,
                                uint replicas) {
    Timer Time;

    for (uint Rep = 0; Rep < replicas; Rep++) {
        std::string RunName = run_id + std::format("{}", Rep);

        // Initialize ak values
        std::fill(_AkVect.begin(), _AkVect.end(), 0.0);
        _AkHist.clear();
        _AkHist.push_back(_AkVect);
        for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
            _Workers[WorkerId].set_ak(_AkVect[WorkerId]);
        }

// initial thermalization
#pragma omp parallel for schedule(static, 1)
        for (auto& Worker : _Workers) {
            // w.randomizeField(0.1);

            // call imaginary action thermalization
            Worker.MonteCarlo_thermalize(1000, "burn-in");
        }
        std::cout << "Thermalization done in " << Time.elapsed_ms() << " ms\n";
        // IO::GlobalLogger.log_string("llr-controller", std::format("Thermalizarion done in {} ms",
        // Time.elapsed_ms()));

        Time.reset();
        NewtonRaphson(nNewton_Raphson, nMonte_Carlo);
        std::cout << "Newton-Raphson done in " << Time.elapsed_ms() << " ms\n";
        // IO::GlobalLogger.log_string("llr-controller", std::format("Newton-Raphson done in {} ms",
        // Time.elapsed_ms()));

        Time.reset();
        RobbinsMonro(nRobbins_Monro, nMonte_Carlo);
        std::cout << "Robbins-Monro done in " << Time.elapsed_ms() << " ms\n";
        // IO::GlobalLogger.log_string("llr-controller", std::format("Robbins-Monro done in {} ms", Time.elapsed_ms()));

        // File output
        Time.reset();
        try {
            H5::Exception::dontPrint();

            // Create the file and group
            H5::H5File File(_OutputPath / "llr" / "meas" / "llr.h5", H5F_ACC_RDWR);
            H5::Group  Group = File.createGroup("/" + RunName);

            // Create dataspace
            const hsize_t                 DataRank = 1;
            std::array<hsize_t, DataRank> Entries = {_AkHist.size()};
            H5::DataSpace                 DataSpace(DataRank, Entries.data());

            // Create datatype
            H5::FloatType DataType = H5::PredType::NATIVE_DOUBLE;

            for (uint Worker = 0; Worker < _Workers.size(); Worker++) {
                std::vector<double> Data;
                Data.reserve(_AkHist.size());
                for (const auto& Val : _AkHist) {
                    Data.push_back(Val[Worker]);
                }

                // Create dataset and write the observables dataset
                H5::DataSet Dataset =
                    File.createDataSet("/" + RunName + "/ak" + std::format("_{:0>3d}", Worker), DataType, DataSpace);
                Dataset.write(Data.data(), DataType);
            }
        } catch (H5::Exception& Exep) {
            H5::Exception::printErrorStack();
            exit(EXIT_FAILURE);
        }

        IO::GlobalLogger.log_string("llr-controller", std::format("Data saved in {} ms", IO::print(Time.elapsed_ms())));
        std::cout << "Data saved in " << Time.elapsed_ms() << " ms"
                  << "\n";
    }
}

////////////////////////////////////////////////////////////////////////////////
// Private methods implementation                                             //
////////////////////////////////////////////////////////////////////////////////
template <class Action>
void LLRController<Action>::NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo) {
    // Newton-Raphson Iterations
    for (int Step = 0; Step < nNewton_Raphson; Step++) {
#pragma omp parallel for schedule(static, 1)
        for (int WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
            montecarlo::data<typename Action::ActionType> Res;

            // MonteCarlo_thermalize to the newly updated value of ak
            _Workers[WorkerId].MonteCarlo_thermalize(100, std::format("NR_{}", Step));

            // run the montecarlo steps
            Res = _Workers[WorkerId].MonteCarlo_run(nMonte_Carlo, std::format("NR_{}", Step));

            // update the ak value
            _AkVect[WorkerId] += -(_SkVect[WorkerId] - Res.S_im) / (_DeltaS * _DeltaS);
            _Workers[WorkerId].set_ak(_AkVect[WorkerId]);
        }

        _AkHist.push_back(_AkVect);
        // replica excange
    }
}

template <class Action>
void LLRController<Action>::RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo) {
    // Robbins-Monro Iterations
    for (int Step = 0; Step < nRobbins_Monro; Step++) {
#pragma omp parallel for schedule(static, 1)
        for (uint WorkerId = 0; WorkerId < _Workers.size(); WorkerId++) {
            montecarlo::data<typename Action::ActionType> Res;

            // MonteCarlo_thermalize to the newly updated value of ak
            _Workers[WorkerId].MonteCarlo_thermalize(100, std::format("RM_{}", Step));

            // run the montecarlo steps
            Res = _Workers[WorkerId].MonteCarlo_run(nMonte_Carlo, std::format("RM_{}", Step));

            // update the ak value
            _AkVect[WorkerId] +=
                -(_SkVect[WorkerId] - Res.S_im) / (static_cast<double>(Step + 1) * (_DeltaS * _DeltaS));
            _Workers[WorkerId].set_ak(_AkVect[WorkerId]);
        }

        _AkHist.push_back(_AkVect);
        // replica excange
    }
}

template <class Action>
void LLRController<Action>::ReplicaExcange() {}

template <class Action>
auto LLRController<Action>::memoryReport() -> size_t {
    size_t Memory = 0;
    for (auto& Worker : _Workers) {
        Memory += Worker.memoryReport();
    }

    return Memory;
}

}  // namespace reticolo::LLR
