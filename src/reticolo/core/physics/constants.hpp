/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: physics/constants.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <numbers>

namespace reticolo::PhysicalConstants {

/* Pysical constants values
   Input parameters from PDG:
       The Review of Particle Physics (2020), P.A. Zyla et al.
       (Particle Data Group), Prog. Theor. Exp. Phys. 2020, 083C01 (2020)
       retrieved 04 Sep 2020 from pdg.lbl.gov */
extern const double c = 299792458;                        // Speed of light (m / s)
extern const double h = 6.62607015e-34;                   // Plank constant (J s)
extern const double hbar = h / (2.0 * std::numbers::pi);  // Reduced Plank constant (J s)
extern const double e = 1.602176634e-19;                  // Electron charge (J / eV)
extern const double kB = 1.3806491e-23;                   // Boltzmann constant (J / K)

}  // namespace reticolo::PhysicalConstants