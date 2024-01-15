#pragma once

#include "lattice/lattice.hpp"
#include "tools/io.hpp"
#include "tools/timer.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

#include <vector>
#include <numeric>
#include <random>
#include <complex>
#include <format>
#include <filesystem>
#include <omp.h>
#include <cmath>

namespace fs = std::filesystem;

namespace reticolo
{
    namespace LLR
    {
        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        class worker
        {
        public:
            struct MC_data
            {
                double
                    acceptance,
                    S_re, S_im,
                    dS_re, dS_im;

                std::string dump_str() { return std::format("{:+8e},{:+8e},{:+8e},{:+8e},{:+8e}", acceptance, S_re, S_im, dS_re, dS_im); }
                std::vector<double> dump_data() const { return std::vector<double>({acceptance, S_re, S_im, dS_re, dS_im}); }
            };
            void make_hdf5_CompType(H5::CompType &type)
            {
                type.insertMember("acceptance", HOFFSET(MC_data, acceptance), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("S_re", HOFFSET(MC_data, S_re), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("S_im", HOFFSET(MC_data, S_im), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("dS_re", HOFFSET(MC_data, dS_re), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("dS_im", HOFFSET(MC_data, dS_im), H5::PredType::NATIVE_DOUBLE);
            }

        private:
            uint _id;
            std::string name;
            fs::path out_path;
            uint log_status;

            lattice<FieldType, dim> field;
            Action<FieldType, ActionType> action;

            std::mt19937_64 rng;
            std::uniform_real_distribution<double> unif;
            std::normal_distribution<double> norm;

            Timer T;
            IO::Logger logger;

            MC_data MC_stats;

            double
                _ak,
                _Sk,
                _width;

            void MCMC_sweep();

        public:
            worker(){};

            void init(fs::path output_path, uint id, vect<dim> sizes, uint seed, Action<FieldType, ActionType>::params par, double Sk, double width, uint log_mode = IO::LOG_mode::silet);

            void randomizeField(double scale = 1.0);

            void imag_thermalize();
            void thermalize(uint nSteps);

            void MCMC_update(uint nMC, std::string run_id);

            void set_ak(double a) { _ak = a; };

            size_t memoryReport();
            void log_MC(std::string run_id, uint iter);
            void log_obs(std::string run_id, uint iter, Action<FieldType, ActionType>::observables &obs);
        };

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::init(fs::path output_path,
                                                                     uint id,
                                                                     vect<dim> sizes,
                                                                     uint seed,
                                                                     typename Action<FieldType, ActionType>::params par,
                                                                     double Sk,
                                                                     double width,
                                                                     uint log_mode)
        {
            Timer T;

            _id = id;
            name = std::format("llr_worker[{:0>3d}]", _id);
            out_path = output_path;

            log_status = log_mode;

            // Initialize the logger
            if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
                logger.init(output_path / "logs", name + ".log");

            // Initialize the output file
            if (log_status == IO::LOG_mode::file_only || log_status == IO::LOG_mode::all)
            {
#pragma omp critical
                {
                    try
                    {
                        H5::Exception::dontPrint();
                        H5::H5File *file = new H5::H5File(out_path / "meas" / (name + ".h5"), H5F_ACC_TRUNC);
                        // write attributes
                        delete file;
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
            action = Action<FieldType, ActionType>(par);

            // RNGs stuff
            rng.seed(seed);
            unif = std::uniform_real_distribution<double>(0.0, 1.0);
            norm = std::normal_distribution<double>(0.0, 1.0);

            // initialize the ak parameter
            _ak = 0.0;
            _Sk = Sk;
            _width = width;

            double elapsed = T.elapsed_ms();

            if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
            {
                logger.log_memory(name, "Allocated", memoryReport());
                logger.log_string(name, std::format("Action: \"{}\" ({})", action.action_name(), action.action_parameters()));
                logger.log_string(name, std::format("LLR: Sk: {:>6f} width: {:>6f}", _Sk, _width));
                logger.log_timing(name, "Initialization done", elapsed);
            }
        };

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::randomizeField(double scale)
        {
            for (size_t site = 0; site < field.getNsites(); site++)
                field[site] = scale * FieldType(norm(rng), norm(rng));
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::MCMC_sweep()
        {
            int acc = 0;

            ActionType
                dS,               // local action variation
                dS_tot(0.0, 0.0); // cumulative action variation

            FieldType dfield; // local field variation

            vect<dim> sizes = field.getSizes();
            vect<dim> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_vect(sizes, coord), i++)
            {
                // dfield = FieldType(0.2 * norm(rng), 0.2 * norm(rng));
                dfield = FieldType(0.2 * (unif(rng) - 0.5), 0.2 * (unif(rng) - 0.5));

                dS = action.compute_dS_loc(field, dfield, coord);

                if (exp(-dS.real()) > unif(rng))
                {
                    acc++;
                    field[coord] += dfield;
                    dS_tot += dS;
                }
            }

            MC_stats.acceptance = static_cast<double>(acc) / field.getNsites();
            MC_stats.S_re += dS_tot.real();
            MC_stats.S_im += dS_tot.imag();
            MC_stats.dS_re = dS_tot.real();
            MC_stats.dS_im = dS_tot.imag();
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::thermalize(uint nSteps)
        {
            T.reset();

            ActionType S = action.compute_S(field);

            MC_stats.S_re = S.real();
            MC_stats.S_im = S.imag();

            for (uint iter = 0; iter < nSteps; iter++)
            {
                MCMC_sweep();

                if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
                    log_MC("therm", iter);
            }
            if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
                logger.log_timing(name, std::format("Performed {} thermalization steps", nSteps), T.elapsed_ms());
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::MCMC_update(uint nMC, std::string run_id)
        {
            T.reset();

            // Initialize std::vector of Monte Carlo stats
            std::vector<MC_data> stats;
            stats.clear();
            stats.reserve(nMC);

            // Initialize std::vector of observable measurements
            std::vector<typename Action<FieldType, ActionType>::observables> obs;
            obs.clear();
            obs.reserve(nMC);

            // Initialize the starting value of S
            ActionType S = action.compute_S(field);
            MC_stats.S_re = S.real();
            MC_stats.S_im = S.imag();

            for (uint iter = 0; iter < nMC; iter++)
            {
                MCMC_sweep();

                stats.push_back(MC_stats);

                obs.push_back(action.Measure(field));

                if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
                {
                    log_MC(run_id, iter);
                    log_obs(run_id, iter, obs.back());
                }
            }

            // Write to HDF5 file
            if (log_status == IO::LOG_mode::file_only || log_status == IO::LOG_mode::all)
            {
#pragma omp critical
                {
                    std::cout << name << "_" << run_id << std::endl;

                    try
                    {
                        H5::Exception::dontPrint();

                        // Create the file
                        H5::H5File *file = new H5::H5File(out_path / "meas" / (name + ".h5"), H5F_ACC_RDWR);
                        H5::Group *group = new H5::Group(file->createGroup("/" + run_id));

                        // Create dataspace
                        hsize_t entries[] = {nMC};
                        H5::DataSpace space(1, entries);

                        // Create datatype and write the observables dataset
                        H5::CompType obs_type(sizeof(typename Action<FieldType, ActionType>::observables));
                        action.make_hdf5_CompType(obs_type);
                        H5::DataSet *obs_dataset = new H5::DataSet(file->createDataSet("/" + run_id + "/obs", obs_type, space));
                        obs_dataset->write(obs.data(), obs_type);
                        delete obs_dataset;

                        // Create datatype and write the observables dataset
                        H5::CompType MC_type(sizeof(MC_data));
                        this->make_hdf5_CompType(MC_type);
                        H5::DataSet *MC_dataset = new H5::DataSet(file->createDataSet("/" + run_id + "/mc", MC_type, space));
                        MC_dataset->write(stats.data(), MC_type);
                        delete MC_dataset;

                        delete file;
                    }
                    catch (H5::Exception &e)
                    {
                        e.printErrorStack();
                        exit(EXIT_FAILURE);
                    }
                }
            }

            if (log_status == IO::LOG_mode::log_only || log_status == IO::LOG_mode::all)
                logger.log_timing(name, std::format("Performed {} MonteCarlo steps", nMC), T.elapsed_ms());
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline size_t worker<Action, FieldType, ActionType, dim>::memoryReport()
        {
            size_t memory = 0;
            memory += field.memoryReport();

            return memory;
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::log_MC(std::string run_id, uint iter)
        {
            logger.log_string(
                name,
                std::format("MC data_dump ->\t{}\t{},{:+8e},{:+8e},{:+8e},{:+8e},{:+8e},",
                            run_id,
                            iter,
                            MC_stats.acceptance,
                            MC_stats.S_re,
                            MC_stats.S_im,
                            MC_stats.dS_re,
                            MC_stats.dS_im));
        }

        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        inline void worker<Action, FieldType, ActionType, dim>::log_obs(std::string run_id, uint iter, Action<FieldType, ActionType>::observables &obs)
        {
            logger.log_string(name, std::format("obs data_dump ->\t{}\t{},", run_id, iter) + obs.dump_str());
        }
    } // namespace LLR
} // namespace reticolo
