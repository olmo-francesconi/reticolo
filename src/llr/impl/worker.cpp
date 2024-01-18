/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: llr/impl/worker.cpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "llr/worker.hpp"

#include "lattice/lattice.hpp"
#include "tools/io.hpp"
#include "tools/timer.hpp"
#include "tools/types.hpp"

#include "action/phi4.hpp"

#include "H5Cpp.h"

#include <vector>
#include <numeric>
#include <random>
#include <complex>
#include <format>
#include <filesystem>
#include <omp.h>
#include <cmath>

namespace reticolo
{
    namespace LLR
    {
        template <class Action>
        void worker<Action>::init(fs::path output_path,
                                  uint id,
                                  uintvect<Action::dims> sizes,
                                  uint rng_seed,
                                  typename Action::params par,
                                  double Sk,
                                  double width,
                                  IO::LOG_mode log_mode)
        {
            Timer T;

            _id = id;
            _name = std::format("llr_worker[{:0>3d}]", _id);
            _out_path = output_path;

            _log_status = log_mode;

            // Initialize the logger
            if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
                logger.init(output_path / "logs", _name + ".log", _name + "LOG");

            // Initialize the output file
            if (_log_status == IO::LOG_mode::file_only || _log_status == IO::LOG_mode::all)
            {
#pragma omp critical
                {
                    try
                    {
                        H5::Exception::dontPrint();
                        H5::H5File file(_out_path / "meas" / (_name + ".h5"), H5F_ACC_TRUNC);
                        // write attributes
                    }
                    catch (H5::Exception &e)
                    {
                        e.printErrorStack();
                        exit(EXIT_FAILURE);
                    }
                }
            }
            // Initialize the lattice
            field.init(sizes);

            // initialize the action
            action = Action(par);

            // RNGs stuff
            rng.seed(rng_seed);
            unif = std::uniform_real_distribution<double>(0.0, 1.0);
            unif_c = std::uniform_real_distribution<double>(-1.0, 1.0);
            norm = std::normal_distribution<double>(0.0, 1.0);

            // initialize the ak parameter
            _ak = 0.0;
            _Sk = Sk;
            _width = width;

            double elapsed = T.elapsed_ms();

            if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
            {
                logger.log_memory(_name, "Allocated", memoryReport());
                logger.log_string(_name, std::format("Action: \"{}\" ({})", action.action_name(), action.action_parameters()));
                logger.log_string(_name, std::format("LLR: Sk: {:>6f} width: {:>6f}", _Sk, _width));
                logger.log_timing(_name, "Initialization done", elapsed);
            }
        };

        template <class Action>
        void worker<Action>::randomizeField(double scale)
        {
            for (size_t site = 0; site < field.getNsites(); site++)
                field[site] = scale * random<typename Action::FieldType>(norm, rng);
        }

        template <class Action>
        void worker<Action>::MCMC_sweep()
        {
            uint acc = 0;

            double window_weight, llr_weight;

            typename Action::ActionType
                dS,               // local action variation
                dS_tot(0.0, 0.0); // cumulative action variation

            typename Action::FieldType dfield; // local field variation

            uintvect<Action::dims> sizes = field.getSizes();
            uintvect<Action::dims> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_coord(sizes, coord), i++)
            {
                // dfield = Action::FieldType(0.2 * (unif(rng) - 0.5), 0.2 * (unif(rng) - 0.5));
                dfield = 0.2 * random<typename Action::FieldType>(unif_c, rng);

                dS = action.compute_dS_loc(field, dfield, coord);

                // std::cout << dS.real() << ", " << dS.imag() << std::endl;

                window_weight = (dS.imag() * (dS.imag() + 2.0 * MC_stats.S_im - 2.0 * _Sk)) / (2.0 * _width * _width);
                llr_weight = _ak * dS.imag();

                if (exp(-(dS.real() + window_weight + llr_weight)) > unif(rng))
                {
                    acc++;
                    field[coord] += dfield;
                    dS_tot += dS;
                    MC_stats.S_re += dS.real();
                    MC_stats.S_im += dS.imag();
                }
            }

            MC_stats.acceptance = static_cast<double>(acc) / field.getNsites();
            MC_stats.dS_re = dS_tot.real();
            MC_stats.dS_im = dS_tot.imag();
        }

        template <class Action>
        void worker<Action>::MonteCarlo_thermalize(uint nSteps, std::string run_id)
        {
            T.reset();

            typename Action::ActionType S = action.compute_S(field);

            MC_stats.S_re = S.real();
            MC_stats.S_im = S.imag();

            for (uint iter = 0; iter < nSteps; iter++)
            {
                MCMC_sweep();

                if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
                    log_MC(run_id + "_therm", iter);
            }
            if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
                logger.log_timing(_name, std::format("Performed {} thermalization steps", nSteps), T.elapsed_ms());
        }

        // template <class Action>
        // montecarlo::data<typename Action::Actiontype> worker<Action>::MonteCarlo_run(uint nMC, std::string run_id)
        // {
        //     return montecarlo::data<typename Action::Actiontype>();
        // }

        template <class Action>
        montecarlo::data<typename Action::ActionType> worker<Action>::MonteCarlo_run(uint nMC, std::string run_id)
        {
            T.reset();

            montecarlo::data<typename Action::ActionType> res;

            // Initialize std::vector of Monte Carlo stats
            std::vector<montecarlo::data<typename Action::ActionType>> stats;
            stats.clear();
            stats.reserve(nMC);

            // Initialize std::vector of observable measurements
            std::vector<typename Action::observables> obs;
            obs.clear();
            obs.reserve(nMC);

            // Initialize the starting value of S
            typename Action::ActionType S = action.compute_S(field);
            MC_stats.S_re = S.real();
            MC_stats.S_im = S.imag();

            for (uint iter = 0; iter < nMC; iter++)
            {
                MCMC_sweep();

                stats.push_back(MC_stats);

                obs.push_back(action.Measure(field));

                res += MC_stats;

                if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
                {
                    log_MC(run_id, iter);
                    log_obs(run_id, iter, obs.back());
                }
            }

            res /= static_cast<double>(nMC);

            // Write to HDF5 file
            if (_log_status == IO::LOG_mode::file_only || _log_status == IO::LOG_mode::all)
            {
#pragma omp critical
                {
                    try
                    {
                        H5::Exception::dontPrint();

                        // Create the file
                        H5::H5File file(_out_path / "meas" / (_name + ".h5"), H5F_ACC_RDWR);
                        H5::Group group = file.createGroup("/" + run_id);

                        // Create dataspace
                        hsize_t entries[] = {nMC};
                        H5::DataSpace space(1, entries);

                        // Create datatype and write the observables dataset
                        H5::CompType obs_type(sizeof(typename Action::observables));
                        action.make_hdf5_CompType(obs_type);
                        H5::DataSet obs_dataset = file.createDataSet("/" + run_id + "/obs", obs_type, space);
                        obs_dataset.write(obs.data(), obs_type);

                        // Create datatype and write the observables dataset
                        H5::CompType MC_type = montecarlo::make_mc_data_hdf5_CompType<typename Action::ActionType>();
                        H5::DataSet MC_dataset = file.createDataSet("/" + run_id + "/mc", MC_type, space);
                        MC_dataset.write(stats.data(), MC_type);
                    }
                    catch (H5::Exception &e)
                    {
                        e.printErrorStack();
                        exit(EXIT_FAILURE);
                    }
                }
            }

            if (_log_status == IO::LOG_mode::log_only || _log_status == IO::LOG_mode::all)
                logger.log_timing(_name, std::format("Performed {} MonteCarlo steps", nMC), T.elapsed_ms());

            return res;
        }

        template <class Action>
        size_t worker<Action>::memoryReport()
        {
            size_t memory = 0;
            memory += field.memoryReport();

            return memory;
        }

        template <class Action>
        void worker<Action>::log_MC(std::string run_id, uint iter)
        {
            logger.log_string(
                _name,
                std::format("MC data_dump ->\t{}\t{},{:+8e},{:+8e},{:+8e},{:+8e},{:+8e},",
                            run_id,
                            iter,
                            MC_stats.acceptance,
                            MC_stats.S_re,
                            MC_stats.S_im,
                            MC_stats.dS_re,
                            MC_stats.dS_im));
        }

        template <class Action>
        void worker<Action>::log_obs(std::string run_id, uint iter, Action::observables &obs)
        {
            logger.log_string(_name, std::format("obs data_dump ->\t{}\t{},", run_id, iter) + obs.dump_str());
        }

        // Explicit instantiations
        template class worker<action::phi4>;

    } // namespace LLR

} // namespace reticolo