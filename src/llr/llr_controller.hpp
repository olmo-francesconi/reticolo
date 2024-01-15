#pragma once

#include "llr/llr_worker.hpp"

#include "tools/types.hpp"
#include "tools/io.hpp"

#include <vector>
#include <omp.h>

#include <filesystem>
namespace fs = std::filesystem;

namespace reticolo
{
    namespace LLR
    {
        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        class controller
        {
        private:
            std::vector<worker<Action, FieldType, ActionType, dim>> workers;
            std::random_device rd;
            fs::path output_path;
            uint nMC, nNR, nRM;

            std::vector<double> ak_vect;
            std::vector<double> Sk_vect;

            void NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo);
            void RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo);
            void ReplicaExcange();

            IO::Logger log;

        public:
            controller(){};

            void init(fs::path output_path, vect<dim> lattice_size, uint nWorkers, Action<FieldType, ActionType>::params par);

            void run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo);

            size_t memoryReport();
        };

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void controller<Action, FieldType, ActionType, dim>::NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo)
        {
            // Newton-Raphson Iterations
            for (int step = 0; step < nNewton_Raphson; step++)
            {
#pragma omp parallel for schedule(static, 1)
                for (int i = 0; i < workers.size(); i++)
                {
                    // thermalize to the newly updated value of ak
                    workers[i].thermalize(100);

                    // run the montecarlo steps
                    workers[i].MCMC_update(nMonte_Carlo, std::format("NR_{}", step));

                    // update the ak value
                    // ak_vect[i] = res.dS.imag();
                }

                // replica excange
            }
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void controller<Action, FieldType, ActionType, dim>::RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo)
        {
            // Robbins-Monro Iterations
            for (int step = 0; step < nRobbins_Monro; step++)
            {
#pragma omp parallel for schedule(static, 1)
                for (int i = 0; i < workers.size(); i++)
                {
                    // thermalize to the newly updated value of ak
                    workers[i].thermalize(100);

                    // run the montecarlo steps
                    workers[i].MCMC_update(nMonte_Carlo, std::format("NR_{}", step));

                    // update the ak value
                    // ak_vect[i] = res.dS.imag();
                }
            }
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void controller<Action, FieldType, ActionType, dim>::ReplicaExcange()
        {
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void controller<Action, FieldType, ActionType, dim>::init(fs::path out_path, vect<dim> lattice_size, uint nWorkers, Action<FieldType, ActionType>::params par)
        {
            // reset global timer
            GlobalTimer.reset();

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
                IO::GlobalLogger.init(output_path / "logs", "global.log");
                log.init(output_path / "logs", "llr_controller.log");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }

            // Log that the folder and loggind stuff has bee initialized properly to info
            log.log_string("llr_controller", "Initialization started");
            log.log_string("llr_controller", "main oputput folder: " + output_path.string());
            log.log_string("llr_controller", "measurements folder: " + (output_path / "meas").string());
            log.log_string("llr_controller", "        logs folder: " + (output_path / "logs").string());

            // info-Log that we are starting to initialize the workers
            // info-Log the threading configuration
            log.log_threadig("llr_controller", omp_get_max_threads());

            // Initialize ak values
            ak_vect.resize(nWorkers);
            std::fill(ak_vect.begin(), ak_vect.end(), 0.0);

            // Initialize Sk values
            Sk_vect.resize(nWorkers);
            for (int i = 0; i < nWorkers; i++)
                Sk_vect[i] = 100.0 * (double(i) + 0.5);

            // Initialize workers
            workers.resize(nWorkers);

#pragma omp parallel for schedule(static, 1)
            for (uint i = 0; i < nWorkers; i++)
            {
                workers[i].init(out_path, i, lattice_size, rd(), par, Sk_vect[i], 1.0, IO::LOG_mode::all);
                workers[i].set_ak(ak_vect[i]);
            }

            log.log_memory("llr_controller", "Total memory allocated: ", memoryReport());
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void controller<Action, FieldType, ActionType, dim>::run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo)
        {
            // initial thermalization
            for (auto &w : workers)
            {
                // call imaginary action thermalization

                // do a simple initial thermalization in the imaginary action interval
                w.thermalize(100);
            }

            NewtonRaphson(nNewton_Raphson, nMonte_Carlo);
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline size_t controller<Action, FieldType, ActionType, dim>::memoryReport()
        {
            size_t memory = 0;
            for (auto &w : workers)
                memory += w.memoryReport();

            return memory;
        }
    } // namespace LLR
} // namespace reticolo