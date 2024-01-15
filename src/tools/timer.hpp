#pragma once

#include <chrono>

namespace reticolo
{
    // Timer Class
    class Timer
    {
    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double> s_;
        typedef std::chrono::duration<double, std::milli> ms_;
        typedef std::chrono::duration<double, std::micro> us_;

        std::chrono::time_point<clock_> _beg;

    public:
        Timer() : _beg(clock_::now()) {}
        void reset() { _beg = clock_::now(); }

        double elapsed_s() const { return std::chrono::duration_cast<s_>(clock_::now() - _beg).count(); }
        double elapsed_ms() const { return std::chrono::duration_cast<ms_>(clock_::now() - _beg).count(); }
        double elapsed_us() const { return std::chrono::duration_cast<us_>(clock_::now() - _beg).count(); }
    };

    // Global timer
    inline Timer GlobalTimer;

} // namespace reticolo