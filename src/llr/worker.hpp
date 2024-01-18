/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/worker.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "lattice/lattice.hpp"
#include "tools/io.hpp"
#include "tools/timer.hpp"
#include "tools/types.hpp"
#include "montecarlo/montecarlo.hpp"

#include "H5Cpp.h"

#include <random>
#include <complex>
#include <filesystem>

namespace fs = std::filesystem;

namespace reticolo
{
    namespace LLR
    {
        template <class Action>
        class worker
        {
        private:
            /* Metadata and output */
            uint _id;
            std::string _name;
            std::string _run_id;
            fs::path _out_path;
            IO::LOG_mode _log_status;

            /* Physics stuff */
            Action action;
            lattice<typename Action::FieldType, Action::dims> field;
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
            double
                _ak,
                _Sk,
                _width;

        public:
            /* Default constructor (does nothing) */
            worker(){};

            /* Initializer function -> actually does all the set-up */
            void init(fs::path output_path, uint id, uintvect<Action::dims> sizes, uint seed, Action::params par, double Sk, double width, IO::LOG_mode log_mode = IO::LOG_mode::silent);

            /* Randomize the field (hot start)*/
            void randomizeField(double scale = 1.0);

            /* Performs a single sweep of the lattice (updates worker::MC_stats) */
            void MCMC_sweep();

            /* Performs nSteps thermalization steps and no measurements */
            void MonteCarlo_thermalize(uint nSteps, std::string run_id);

            /* Performs nMC Monte Carlo steps and returns a montecarlo::data object with the averages */
            montecarlo::data<typename Action::ActionType> MonteCarlo_run(uint nMC, std::string run_id);

            /* Set the value of the LLR ak parameters*/
            void set_ak(double a) { _ak = a; };

            /* Various logging utilities */
            size_t memoryReport();
            void log_MC(std::string run_id, uint iter);
            void log_obs(std::string run_id, uint iter, Action::observables &obs);
        };

    } // namespace LLR

} // namespace reticolo
