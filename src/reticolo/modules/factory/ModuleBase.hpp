/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/ModuleBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include "yaml-cpp/node/node.h"

namespace reticolo {

class ModuleBase {
  public:
    /* virtual destructor */
    virtual ~ModuleBase() = default;

    /* setup */
    virtual void setup(const YAML::Node&) = 0;

    /* execution */
    virtual void execute(const YAML::Node&) = 0;
};

}  // namespace reticolo
