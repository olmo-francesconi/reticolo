/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io/logger.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "reticolo/core/tools/io_utils.hpp"

namespace reticolo::IO {

/*--------------------------------------------------------------------------------------------------
    Logger Class Definition
--------------------------------------------------------------------------------------------------*/

class Logger {
  private:
    /* Naming and paths */
    std::string           _FileName;    // Output Log filename
    std::string           _LoggerName;  // Name of the logger
    std::filesystem::path _Path;        // Output file path

    /* State and variables */
    int         _State;   // State of the logger( 0: uninitialized, 1: initialized, -1: Generic error)
    bool        _SdtOut;  // Print to stdout as well
    std::string _Msg;     // String buffer

    /* Streams */
    std::ofstream _File;  // File opuput stream

  public:
    /* Constructors */
    Logger() : _LoggerName("unnamed"), _State(0), _SdtOut(true){};
    Logger(std::string name) : _LoggerName(std::move(name)), _State(0), _SdtOut(true){};
    Logger(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "",
           bool StdOut = true) {
        init(OutPath, FileName, LogName, StdOut);
    };
    Logger(Logger&& other) noexcept = default;

    /* Destructor */
    ~Logger() {
        if (_File.is_open()) {
            _File.close();
        }
    }

    /* Initializer (This actually initialize the fstream and checks paths) */
    void init(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "",
              bool StdOut = true);

    /* Log single liners for common stuff */
    void log_string(const std::string& who, const std::string& what);
    void log_timing(const std::string& who, const std::string& what, double time);
    void log_memory(const std::string& who, const std::string& what, size_t memory);
    void log_threadig(const std::string& who, size_t nThreads);

    /* Log from a stringstream, */
    void log(std::stringstream& message);
    void operator<<(std::stringstream& message);
    void operator<<(const std::string& message);
};

/*--------------------------------------------------------------------------------------------------
    Public methods implementation
--------------------------------------------------------------------------------------------------*/

inline void Logger::init(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName,
                         bool StdOut) {
    if (_File.is_open()) {
        _File.close();
    }

    _Path = std::filesystem::absolute(OutPath);
    _FileName = FileName;
    if (!LogName.empty()) {
        _LoggerName = LogName;
    }
    _SdtOut = StdOut;

    _File.open(_Path / _FileName, std::ios::out | std::ios::trunc);

    if (!_File.is_open()) {
        _State = -1;
        throw std::runtime_error(std::string("reticolo: LOGGER ERROR : Logger (") + _LoggerName +
                                 std::string(") could not create log file (") + std::string(_Path / _FileName) +
                                 std::string(")"));
    }
    _State = 1;
    _File << pretty_welcome() << '\n' << std::flush;
}

inline void Logger::log_string(const std::string& who, const std::string& what) {
    if (_State == 1) {
        _Msg = LI_time() + who + " - " + what + '\n';
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_timing(const std::string& who, const std::string& what, double time) {
    if (_State == 1) {
        _Msg = LI_time() + who + " - " + what + "\t[" + std::format("{:.2f}", time) + " ms]\n";
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_memory(const std::string& who, const std::string& what, size_t memory) {
    if (_State == 1) {
        _Msg = LI_time() + who + " - " + what + " " + pretty_bytes(memory) + '\n';
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log_threadig(const std::string& who, size_t nThreads) {
    if (_State == 1) {
        _Msg = LI_time() + who + " - Running on " + std::to_string(nThreads) + " threads" + '\n';
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::log(std::stringstream& message) {
    if (_State == 1) {
        _Msg = message.str();
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
        message.str(std::string());
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::operator<<(std::stringstream& message) {
    if (_State == 1) {
        _Msg = message.str();
        _File << _Msg << std::flush;
        if (_SdtOut) {
            std::cout << _Msg;
        }
        message.str(std::string());
    } else {
        throw std::runtime_error("reticolo: LOGGER ERROR : Trying to write to an uninitialized logger");
    }
}

inline void Logger::operator<<(const std::string& message) {
    if (_State == 1) {
        _File << message << std::flush;
        if (_SdtOut) {
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
