/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io/logger.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/timer.hpp"

namespace reticolo::IO {
//----------------------------------------------------------
// Logging helper functions
//----------------------------------------------------------

/* Default reticolo log line init with timing */
inline auto LI_time() -> std::string {
  std::string Message = "reticolo..." + std::format("{:.>10.3f}", GlobalTimer.elapsed_s()) + " s | ";
  return Message;
};

/* Default reticolo log line init with dots */
inline auto LI_dots() -> std::string {
  std::string Message = "reticolo............... | ";
  return Message;
};

/* Default reticolo log line empty init */
inline auto LI_void() -> std::string {
  std::string Message = "                        | ";
  return Message;
};

/* Default reticolo log error line init */
inline auto LI_erro() -> std::string {
  std::string Message = "reticolo..........ERROR | ";
  return Message;
};

/* Default reticolo log warning line init */
inline auto LI_warn() -> std::string {
  std::string Message = "reticolo..........ERROR | ";
  return Message;
};

//----------------------------------------------------------
// Class: Logger
//----------------------------------------------------------
//
// Simple interface to log messages and data
//

enum LOG_mode { silent = 0, log_only = 1, file_only = 2, all = 3 };

class Logger {
 private:
  std::string file_name;
  std::string logger_name;
  std::filesystem::path path;

  int state;

  std::ofstream file;

 public:
  Logger() : logger_name("unnamed"), state(0){};
  Logger(std::string name) : logger_name(std::move(name)), state(0){};
  Logger(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "") {
    init(OutPath, FileName, LogName);
  };

  Logger(Logger&& other) noexcept = default;  // move constructor

  ~Logger() {
    if (file.is_open()) {
      file.close();
    }
  }

  inline void init(const std::filesystem::path& OutPath, const std::string& FileName, const std::string& LogName = "") {
    if (state != 0) {
      throw std::runtime_error(std::string("reticolo...LOGGER ERROR | Logger (") + logger_name +
                               std::string(") init() called on an already initialized logger"));
    }

    path = std::filesystem::absolute(OutPath);
    this->file_name = FileName;
    if (!LogName.empty()) {
      this->logger_name = LogName;
    }

    file.open(path / file_name, std::ios::out | std::ios::trunc);

    if (!file.is_open()) {
      state = -1;
      throw std::runtime_error(std::string("reticolo: LOGGER ERROR : Logger (") + logger_name +
                               std::string(") could not create log file (") + std::string(path / file_name) +
                               std::string(")"));
    }
    state = 1;

    file << pretty_welcome() << '\n';
  }

  inline void log_string(const std::string& who, const std::string& what) {
    if (state == 1) {
      file << LI_time() << who << " - " << what << '\n';
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }

  inline void log_timing(const std::string& who, const std::string& what, double time) {
    if (state == 1) {
      file << LI_time() << who << " - " << what << " in " << std::format("{:>8.2f}", time) << " ms" << '\n';
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }

  inline void log_memory(const std::string& who, const std::string& what, size_t memory) {
    if (state == 1) {
      file << LI_time() << who << " - " << what << " " << pretty_bytes(memory) << '\n';
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }

  inline void log_threadig(const std::string& who, size_t nThreads) {
    if (state == 1) {
      file << LI_time() << who << " - Running on " << nThreads << " threads" << '\n';
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }

  inline void log(std::stringstream& message) {
    if (state == 1) {
      file << message.str() << '\n';
      message.str(std::string());
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }

  /* Writes message to file and clears message */
  inline void operator<<(std::stringstream& message) {
    if (state == 1) {
      file << message.str();
      std::cout << message.str();
      message.str(std::string());
    } else {
      throw std::runtime_error(
          "reticolo: LOGGER ERROR : Trying to write to an "
          "uninitialized logger");
    }
  }
};

// Set of default Loggers
// These are global but need to be initialized somewhere in the code
// inline Logger ErrorLogger("ErrorLogger");
inline Logger GlobalLogger("GlobalLogger");
// inline Logger InfoLogger("InfoLogger");

}  // namespace reticolo::IO
