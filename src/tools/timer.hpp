#pragma once

#include <chrono>

namespace LLR
{
    // Timer Class
    class Timer
    {
    public:
        Timer() : beg_(clock_::now()) {}
        void reset() { beg_ = clock_::now(); }
        double elapsed_s() const
        {
            return std::chrono::duration_cast<s_>(clock_::now() - beg_).count();
        }

        double elapsed_ms() const
        {
            return std::chrono::duration_cast<ms_>(clock_::now() - beg_).count();
        }

    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double> s_;
        typedef std::chrono::duration<double, std::milli> ms_;

        std::chrono::time_point<clock_> beg_;
    };

    // Global timer
    Timer GlobalTimer;
}