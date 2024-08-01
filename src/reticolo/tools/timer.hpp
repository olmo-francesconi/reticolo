/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/timer.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <chrono>
#include <ratio>

namespace reticolo {

// Timer Class
class Timer {
  private:
    using clock_ = std::chrono::high_resolution_clock;
    using s_ = std::chrono::duration<double>;
    using ms_ = std::chrono::duration<double, std::milli>;
    using us_ = std::chrono::duration<double, std::micro>;

    std::chrono::time_point<clock_> m_Beg;

  public:
    Timer() : m_Beg(clock_::now()) {}
    void reset() { m_Beg = clock_::now(); }

    [[nodiscard]] auto elapsed_s() const -> double {
        return std::chrono::duration_cast<s_>(clock_::now() - m_Beg).count();
    }
    [[nodiscard]] auto elapsed_ms() const -> double {
        return std::chrono::duration_cast<ms_>(clock_::now() - m_Beg).count();
    }
    [[nodiscard]] auto elapsed_us() const -> double {
        return std::chrono::duration_cast<us_>(clock_::now() - m_Beg).count();
    }
};

// Global timer
inline Timer GlobalTimer;

}  // namespace reticolo
