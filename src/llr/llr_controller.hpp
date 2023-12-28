#pragma once

#include "llr/llr_worker.hpp"

#include "tools/types.hpp"
#include "tools/io.hpp"

#include <vector>
#include <omp.h>

#include <filesystem>
namespace fs = std::filesystem;

namespace LLR
{
    class llr_controller
    {
    private:
        std::vector<llr_worker> workers;
        std::random_device rd;
        fs::path output_path;
        uint nMC, nNR, nRM;

    public:
        llr_controller(fs::path output_path, uint nWorkers, uint nMC) { init(output_path, nWorkers, nMC); };

        void init(fs::path output_path, uint nWorkers, uint nMC);

        void thermalize();
        void NewtonRaphson(uint nNR, uint nMC);
        void RobbinsMonroe(uint nRM, uint nMC);

        size_t memoryReport();
    };

    inline void llr_controller::init(fs::path out_path, uint nWorkers, uint nMC)
    {
        // reset global timer
        GlobalTimer.reset();

        output_path = fs::canonical(fs::absolute(out_path));

        try
        {
            fs::create_directory(output_path);
            fs::create_directory(output_path / "logs");
            fs::create_directory(output_path / "meas");
            fs::create_directory(output_path / "conf");
            IO::InfoLogger.init(output_path / "logs", "info.log");
            IO::ErrorLogger.init(output_path / "logs", "err.log");
            IO::PerfLogger.init(output_path / "logs", "perf.log");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }

        // info-Log that the folder and loggind stuff has bee initialized properly to info
        IO::PerfLogger.log_string("llr_controller", "Initialization started");
        IO::PerfLogger.log_string("llr_controller", "main oputput folder: " + output_path.string());
        IO::PerfLogger.log_string("llr_controller", "measurements folder: " + (output_path / "meas").string());
        IO::PerfLogger.log_string("llr_controller", "        logs folder: " + (output_path / "logs").string());

        // info-Log that we are starting to initialize the workers
        // info-Log the threading configuration

        IO::PerfLogger.log_threadig("llr_copntroller", omp_get_max_threads());

        // Initialize action parameters
        action::phi4_params par;
        par.lambda = 1.0;
        par.eta = 9.0;
        par.mu = 0.0;

        // Initialize workers
        workers.resize(nWorkers);
#pragma omp parallel for
        for (uint i = 0; i < nWorkers; i++)
        {
            Timer T;
            T.reset();

            workers[i].init(i, par, {8, 8, 8, 8}, rd());

            double elapsed = T.elapsed_ms();
#pragma omp critical
            {
                IO::PerfLogger.log_timing(std::format("llr_worker[{:0>3d}] on thread_{:0>3d}", i, omp_get_thread_num()), "Initialization done", elapsed);
                IO::PerfLogger.log_memory(std::format("llr_worker[{:0>3d}] on thread_{:0>3d}", i, omp_get_thread_num()), "Allocated ", workers[i].memoryReport());
            }
        }

        IO::PerfLogger.log_memory("llr_controller", "Total memory allocated ", this->memoryReport());
    }

    // #pragma omp parallel for
    //         for (uint i = 0; i < nWorkers; i++)
    //         {
    //             Timer T;
    //             T.reset();

    //             for (uint t = 0; t < nMC; t++)
    //             {
    //                 workers[i].MCMC_update();
    //             }
    //             double elapsed = T.elapsed_ms();

    // #pragma omp critical
    //             IO::PerfLogger.log_timing(std::format("llr_worker[{:0>3d}] on thread_{:0>3d}", i, omp_get_thread_num()), std::format("Performed {} MonteCarlo steps", nMC), elapsed);
    //         }
    //     }

    inline size_t llr_controller::memoryReport()
    {
        size_t memory = 0;
        for (auto &w : workers)
            memory += w.memoryReport();

        return memory;
    }

} // namespace LLR