#pragma once

#include "tools/types.hpp"
#include "tools/io/stuff.hpp"
#include "tools/timer.hpp"

#include <filesystem>
#include <array>
#include <string>
#include <format>
#include <iostream>
#include <fstream>
#include <system_error>

namespace reticolo
{
    namespace IO
    {
        class Logger
        {
        private:
            std::string file_name, logger_name;
            std::filesystem::path path;
            int state;

            std::ofstream file;

        public:
            Logger() : state(0), logger_name("unnamed"){};
            Logger(std::string logger_name) : state(0), logger_name(logger_name){};
            Logger(std::filesystem::path p, std::string fn)
            {
                init(p, fn);
            };

            Logger(Logger &&other) = default; // move constructor

            ~Logger()
            {
                if (file.is_open())
                    file.close();
            }

            void init(std::filesystem::path p, std::string fn)
            {
                path = std::filesystem::absolute(p);
                file_name = fn;

                file.open(path / file_name, std::ios::out | std::ios::trunc);

                if (!file.is_open())
                {
                    state = -1;
                    throw std::runtime_error(std::string("reticolo: LOGGER ERROR : Logger (") +
                                             logger_name +
                                             std::string(") could not create log file (") +
                                             std::string(path / file_name) + std::string(")"));
                }
                else
                    state = 1;

                file << pretty_welcome() << "\n"
                     << logger_name << "\n"
                     << "[Log-type].....[time] ms /   [who]   -   [what]\n"
                     << std::endl;
            }

            void log_string(std::string who, std::string what)
            {
                if (state == 1)
                    file << "reticolo-info..." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << std::endl;
                else
                    throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_timing(std::string who, std::string what, double time)
            {
                if (state == 1)
                    file << "reticolo-time..." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << " in " << std::format("{:>8.2f}", time) << " ms" << std::endl;
                else
                    throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_memory(std::string who, std::string what, size_t memory)
            {
                if (state == 1)
                    file << "reticolo-memory." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << " " << pretty_bytes(memory) << std::endl;
                else
                    throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_threadig(std::string who, size_t nThreads)
            {
                if (state == 1)
                    file << "reticolo-OpenMP." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - Running on " << nThreads << " threads" << std::endl;
                else
                    throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
            }
        };
    } // namespace IO
} // namespace reticolo