#pragma once

#include "tools/io/stuff.hpp"
#include "tools/io/logger.hpp"

namespace reticolo
{
    namespace IO
    {
        enum LOG_mode
        {
            silet = 0,
            log_only = 1,
            file_only = 2,
            all = 3
        };

        // Set of default Loggers
        // These are global but need to be initialized somewhere in the code
        // inline Logger ErrorLogger("ErrorLogger");
        inline Logger GlobalLogger("GlobalLogger");
        // inline Logger InfoLogger("InfoLogger");

    } // namespace IO
} // namespace reticolo