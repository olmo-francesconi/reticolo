// /******************************************************************************

//  - reticolo (www.github.com/olmo-francesconi/reticolo.git)

//  - SourceFile: tools/sig_handler.hpp

//  - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

// ******************************************************************************/
// #pragma once

// #include <csignal>
// #include <cstdlib>
// #include <iostream>

// #include "reticolo/tools/io_utils.hpp"

// using Handler = void (*)(int);

// namespace reticolo::SignalHandler {

// /* SIGINT soft-exit handler */
// inline bool SoftExitRequested = false;
// inline void SIGINT_SoftExit(int signum) {
//     if (!SoftExitRequested) {
//         std::cout << '\n' << IO::LI_warn() << "SoftExit requested (Ctrl+C again to exit immediately)\n";
//     } else {
//         std::cout << '\n' << IO::LI_erro() << "Immediate exit requested, possible data corruption.\n";
//         std::cout << IO::LI_erro() << "Bye :(\n";

//         exit(SIGINT);
//     }
//     SoftExitRequested = true;
// }

// /* Install new handler */
// inline void set_SIGINT_handler(void (*new_hadler)(int)) { signal(SIGINT, new_hadler); }

// /* Restore original handler */
// inline void SIGINT_reset() { signal(SIGINT, SIG_DFL); }

// }  // namespace reticolo::SignalHandler