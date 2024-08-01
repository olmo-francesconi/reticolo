/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io/logger.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "reticolo/tools/io_utils.hpp"

namespace reticolo::IO {

/*--------------------------------------------------------------------------------------------------
    Logger Class Definition
--------------------------------------------------------------------------------------------------*/

class Logger {
  private:
    /* Naming and paths */
    std::string           m_FileName;    // Output Log filename
    std::string           m_LoggerName;  // Name of the logger
    std::filesystem::path m_Path;        // Output file path

    /* State and variables */
    int         m_State;   // State of the logger( 0: uninitialized, 1: initialized, -1: Generic error)
    bool        m_SdtOut;  // Print to stdout as well
    std::string m_Msg;     // String buffer

    /* Streams */
    std::ofstream m_File;  // File opuput stream

  public:
    /* Constructors */
    Logger() : m_LoggerName("unnamed"), m_State(0), m_SdtOut(true) {};
    Logger(std::string name) : m_LoggerName(std::move(name)), m_State(0), m_SdtOut(true) {};
    Logger(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "",
           bool StdOut = true) {
        init(OutPath, FileName, LogName, StdOut);
    };
    Logger(Logger&& other) noexcept = default;

    /* Destructor */
    ~Logger() {
        if (m_File.is_open()) {
            m_File.close();
        }
    }

    /* Initializer (This actually initialize the fstream and checks paths) */
    inline void init(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "",
                     bool StdOut = true);

    /* Log single liners for common stuff */
    inline void log_string(const std::string& who, const std::string& what);
    inline void log_timing(const std::string& who, const std::string& what, double time);
    inline void log_memory(const std::string& who, const std::string& what, size_t memory);
    inline void log_threadig(const std::string& who, size_t nThreads);

    /* Log from a stringstream, */
    inline void log(std::stringstream& message);
    inline void operator<<(std::stringstream& message);
    inline void operator<<(const std::string& message);
};

/*--------------------------------------------------------------------------------------------------
    Public methods implementation
--------------------------------------------------------------------------------------------------*/

inline void Logger::init(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName,
                         bool StdOut) {
    if (m_File.is_open()) {
        m_File.close();
    }

    m_Path = std::filesystem::absolute(OutPath);
    m_FileName = FileName;
    if (!LogName.empty()) {
        m_LoggerName = LogName;
    }
    m_SdtOut = StdOut;

    m_File.open(m_Path / m_FileName, std::ios::out | std::ios::trunc);

    if (!m_File.is_open()) {
        m_State = -1;
        throw std::runtime_error(std::string("reticolo: LOGGER ERROR : Logger (") + m_LoggerName +
                                 std::string(") could not create log file (") + std::string(m_Path / m_FileName) +
                                 std::string(")"));
    }
    m_State = 1;
    if (m_SdtOut) {
        std::cout << pretty_welcome() << '\n';
    }
    m_File << pretty_welcome() << '\n' << std::flush;
}

inline void Logger::log_string(const std::string& who, const std::string& what) {
    if (m_State == 1) {
        m_Msg = LI_time() + who + " - " + what + '\n';
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_timing(const std::string& who, const std::string& what, double time) {
    if (m_State == 1) {
        m_Msg = LI_time() + who + " - " + what + "\t[" + std::format("{:.2f}", time) + " ms]\n";
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_memory(const std::string& who, const std::string& what, size_t memory) {
    if (m_State == 1) {
        m_Msg = LI_time() + who + " - " + what + " " + pretty_bytes(memory) + '\n';
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_threadig(const std::string& who, size_t nThreads) {
    if (m_State == 1) {
        m_Msg = LI_time() + who + " - Running on " + std::to_string(nThreads) + " threads" + '\n';
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log(std::stringstream& message) {
    if (m_State == 1) {
        m_Msg = message.str();
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
        message.str(std::string());
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::operator<<(std::stringstream& message) {
    if (m_State == 1) {
        m_Msg = message.str();
        m_File << m_Msg << std::flush;
        if (m_SdtOut) {
            std::cout << m_Msg;
        }
        message.str(std::string());
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::operator<<(const std::string& message) {
    if (m_State == 1) {
        m_File << message << std::flush;
        if (m_SdtOut) {
            std::cout << message;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

/*--------------------------------------------------------------------------------------------------
    Global Logger explicit instantiation
--------------------------------------------------------------------------------------------------*/

// Gloabl Logger, available to the entirety of the code
inline Logger GlobalLogger("GlobalLogger");

}  // namespace reticolo::IO
