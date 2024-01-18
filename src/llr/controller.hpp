/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/controller.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "llr/worker.hpp"

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
        template <class Action>
        class controller
        {
        private:
            IO::LOG_mode log_mode;

            std::vector<worker<Action>> workers;
            std::random_device rd;
            fs::path output_path;
            uint nMC, nNR, nRM;

            std::vector<double> ak_vect;
            std::vector<std::vector<double>> ak_hist;
            std::vector<double> Sk_vect;

            void NewtonRaphson(uint nNewton_Raphson, uint nMonte_Carlo);
            void RobbinsMonro(uint nRobbins_Monro, uint nMonte_Carlo);
            void ReplicaExcange();

            IO::Logger log;

        public:
            controller(){};

            void init(fs::path output_path, uintvect<Action::dims> lattice_size, uint nWorkers, Action::params par, IO::LOG_mode main_log_mode, IO::LOG_mode MC_log_mode);

            void run(uint nNewton_Raphson, uint nRobbins_Monro, uint nMonte_Carlo, std::string run_id, uint replicas);

            size_t memoryReport();
        };

    } // namespace LLR

} // namespace reticolo