/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/impl/controller.cpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "llr/controller.hpp"

#include "llr/worker.hpp"

#include "tools/types.hpp"
#include "tools/io.hpp"

#include "action/phi4.hpp"

#include <vector>
#include <omp.h>

#include <filesystem>
namespace fs = std::filesystem;

namespace reticolo
{
    namespace LLR
    {
        template <class Action>
        void controller<Action>::NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo)
        {
            // Newton-Raphson Iterations
            for (int step = 0; step < nNewton_Raphson; step++)
            {
#pragma omp parallel for schedule(static, 1)
                for (int i = 0; i < workers.size(); i++)
                {
                    montecarlo::data<typename Action::ActionType> res;

                    // MonteCarlo_thermalize to the newly updated value of ak
                    workers[i].MonteCarlo_thermalize(100, std::format("NR_{}", step));

                    // run the montecarlo steps
                    res = workers[i].MonteCarlo_run(nMonte_Carlo, std::format("NR_{}", step));

                    // update the ak value
                    ak_vect[i] += -(Sk_vect[i] - res.S_im);
                    workers[i].set_ak(ak_vect[i]);
                    if (log_mode == IO::LOG_mode::log_only || log_mode == IO::LOG_mode::all)
#pragma omp critical
                        log.log_string("llr-controller", std::format("NR_{}_{}", step, i));
                }

                ak_hist.push_back(ak_vect);
                // replica excange
            }
        }

        template <class Action>
        void controller<Action>::RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo)
        {
            // Robbins-Monro Iterations
            for (int step = 0; step < nRobbins_Monro; step++)
            {
#pragma omp parallel for schedule(static, 1)
                for (int i = 0; i < workers.size(); i++)
                {
                    montecarlo::data<typename Action::ActionType> res;

                    // MonteCarlo_thermalize to the newly updated value of ak
                    workers[i].MonteCarlo_thermalize(100, std::format("RM_{}", step));

                    // run the montecarlo steps
                    res = workers[i].MonteCarlo_run(nMonte_Carlo, std::format("RM_{}", step));

                    // update the ak value
                    ak_vect[i] += -(Sk_vect[i] - res.S_im) / static_cast<double>(step + 1);
                    workers[i].set_ak(ak_vect[i]);

                    if (log_mode == IO::LOG_mode::log_only || log_mode == IO::LOG_mode::all)
                }

                ak_hist.push_back(ak_vect);
                // replica excange
            }
        }

        template <class Action>
        void controller<Action>::ReplicaExcange()
        {
        }

        template <class Action>
        void controller<Action>::init(fs::path out_path,                   //
                                      uintvect<Action::dims> lattice_size, //
                                      uint nWorkers,                       //
                                      Action::params par,                  //
                                      IO::LOG_mode main_log_mode,          //
                                      IO::LOG_mode MC_log_mode)            //
        {
            // reset global timer
            GlobalTimer.reset();

            log_mode = main_log_mode;

            try
            {
                output_path = fs::absolute(out_path);

                fs::create_directory(output_path);
                output_path = fs::canonical(output_path);
                fs::create_directory(output_path / "logs");
                fs::create_directory(output_path / "meas");
                fs::create_directory(output_path / "conf");
                // IO::InfoLogger.init(output_path / "logs", "info.log");
                // IO::ErrorLogger.init(output_path / "logs", "err.log");
                // IO::GlobalLogger.init(output_path / "logs", "global.log");
                log.init(output_path / "logs", "llr_controller.log", "llr_controllerLOG");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }

            if (log_mode == IO::LOG_mode::log_only || log_mode == IO::LOG_mode::all)
            {
                // Log that the folder and loggind stuff has bee initialized properly to info
                log.log_string("llr_controller", "Initialization started");
                log.log_string("llr_controller", "main oputput folder: " + output_path.string());
                log.log_string("llr_controller", "measurements folder: " + (output_path / "meas").string());
                log.log_string("llr_controller", "        logs folder: " + (output_path / "logs").string());

                // info-Log that we are starting to initialize the workers
                // info-Log the threading configuration
                log.log_threadig("llr_controller", omp_get_max_threads());
            }

            // Initialize the output file
            if (log_mode == IO::LOG_mode::file_only || log_mode == IO::LOG_mode::all)
            {
                try
                {
                    H5::Exception::dontPrint();
                    H5::H5File *file = new H5::H5File(output_path / "meas" / "llr.h5", H5F_ACC_TRUNC);
                    // write attributes
                    delete file;
                }
                catch (H5::Exception &e)
                {
                    e.printErrorStack();
                    exit(EXIT_FAILURE);
                }
            }

            // Initialize ak values
            ak_vect.resize(nWorkers);
            std::fill(ak_vect.begin(), ak_vect.end(), 0.0);
            ak_hist.push_back(ak_vect);

            // Initialize Sk values
            Sk_vect.resize(nWorkers);
            for (int i = 0; i < nWorkers; i++)
                Sk_vect[i] = 0.1 * double(i);

            // Initialize workers
            workers.resize(nWorkers);

#pragma omp parallel for schedule(static, 1)
            for (uint i = 0; i < nWorkers; i++)
            {
                workers[i].init(out_path, i, lattice_size, rd(), par, Sk_vect[i], 1.0, MC_log_mode);
                workers[i].set_ak(ak_vect[i]);
            }

            if (log_mode == IO::LOG_mode::log_only || log_mode == IO::LOG_mode::all)
                log.log_memory("llr_controller", "Total memory allocated: ", memoryReport());
        }

        template <class Action>
        void controller<Action>::run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, std::string run_id, uint replicas)
        {
            for (uint i = 0; i < replicas; i++)
            {
                std::string run_name = run_id + std::format("{}", i);

                // reset ak_hist
                ak_hist.clear();

                // initial thermalization
                for (auto &w : workers)
                {
                    w.randomizeField(0.1);

                    // call imaginary action thermalization
                    w.MonteCarlo_thermalize(1000, "burn-in");
                }

                NewtonRaphson(nNewton_Raphson, nMonte_Carlo);

                RobbinsMonro(nRobbins_Monro, nMonte_Carlo);

                // File output
                if (log_mode == IO::LOG_mode::file_only || log_mode == IO::LOG_mode::all)
                {
                    try
                    {
                        H5::Exception::dontPrint();

                        // Create the file and group
                        H5::H5File file(output_path / "meas" / "llr.h5", H5F_ACC_RDWR);
                        H5::Group group = file.createGroup("/" + run_name);

                        // Create dataspace
                        hsize_t entries[] = {ak_hist.size()};
                        H5::DataSpace dataspace(1, entries);

                        // Create datatype
                        H5::FloatType datatype = H5::PredType::NATIVE_DOUBLE;

                        for (uint i = 0; i < workers.size(); i++)
                        {
                            std::vector<double> data;
                            for (const auto &el : ak_hist)
                                data.push_back(el[i]);

                            // Create dataset and write the observables dataset
                            H5::DataSet dataset = file.createDataSet("/" + run_name + "/ak" + std::format("_{:0>3d}", i), datatype, dataspace);
                            dataset.write(data.data(), datatype);
                        }
                    }
                    catch (H5::Exception &e)
                    {
                        e.printErrorStack();
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

        template <class Action>
        size_t controller<Action>::memoryReport()
        {
            size_t memory = 0;
            for (auto &w : workers)
                memory += w.memoryReport();

            return memory;
        }

        // Explicit instantiation
        template class controller<action::phi4>;

    } // namespace LLR

} // namespace reticolo