#pragma once

#include <filesystem>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <system_error>

namespace LLR
{
    namespace IO
    {
        std::string pretty_bytes(size_t bytes)
        {
            std::array<std::string, 7> suffixes({"B", "KB", "MB", "GB", "TB", "PB", "EB"});
            uint s = 0; // which suffix to use
            double count = bytes;
            while (count >= 1024 && s < 7)
            {
                s++;
                count /= 1024;
            }
            return std::format("{:>6.2f}", count) + " " + suffixes[s];
        };

        std::string pretty_welcome()
        {
            return "LLR";
        }

        class Logger
        {
        private:
            std::string file_name, logger_name;
            std::filesystem::path path;
            int state;

            std::ofstream file;

        public:
            Logger(std::string logger_name = "unnamed") : state(0), logger_name(logger_name){};
            Logger(std::filesystem::path p, std::string fn)
            {
                init(p, fn);
            };

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
                    throw std::runtime_error(std::string("LLR: LOGGER ERROR : Logger (") +
                                             logger_name +
                                             std::string(") could not create log file (") +
                                             std::string(path / file_name) + std::string(")"));
                }
                else
                    state = 1;

                file << pretty_welcome() << "\n\n"
                     << logger_name << "\n\n"
                     << "[Log-type].....[time] ms /   [who]   -   [what]\n"
                     << std::endl;
            }
            void log_string(std::string who, std::string what)
            {
                if (state == 1)
                    file << "LLR-info..." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << std::endl;
                else
                    throw std::runtime_error("LLR: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_timing(std::string who, std::string what, double time)
            {
                if (state == 1)
                    file << "LLR-time..." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << " in " << std::format("{:>8.2f}", time) << " ms" << std::endl;
                else
                    throw std::runtime_error("LLR: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_memory(std::string who, std::string what, size_t memory)
            {
                if (state == 1)
                    file << "LLR-memory." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - " << what << " " << pretty_bytes(memory) << std::endl;
                else
                    throw std::runtime_error("LLR: LOGGER ERROR : Trying to write to an uninitialized logger");
            }

            void log_threadig(std::string who, size_t nThreads)
            {
                if (state == 1)
                    file << "LLR-OpenMP." << std::format("{:.>10.1f}", GlobalTimer.elapsed_ms()) << " ms / " << who << " - Running on " << nThreads << " threads" << std::endl;
                else
                    throw std::runtime_error("LLR: LOGGER ERROR : Trying to write to an uninitialized logger");
            }
        };

        // Set of default Loggers
        // These are global but need to be initialized somewhere in the code
        inline Logger ErrorLogger("ErrorLogger");
        inline Logger PerfLogger("PerfLogger");
        inline Logger InfoLogger("InfoLogger");

    } // namespace IO

} // namespace LLR