/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io_utils.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>
*******************************************************************************/

#pragma once

#include <random>

namespace reticolo {

template <typename TEngine, typename TImpl>
class RNGEngine : TEngine {
  public:
    RNGEngine(TEngine::result_type seed = 0) : TEngine(seed) {
        /* Initialize the distributions */
        _Unif = std::uniform_int_distribution<TImpl>(0.0, 1.0);
        _Unifc = std::uniform_int_distribution<TImpl>(-1.0, 1.0);
    };

    /* Get uniformly distributed variable */
    template <typename return_type>
    inline auto get_unif(return_type& min, return_type& max) -> return_type;

    /* Get normally distributed variable */
    template <typename return_type>
    inline auto get_norm(return_type& mean, return_type& std) -> return_type;

  private:
    std::uniform_real_distribution<TImpl> _Unif;
    std::uniform_real_distribution<TImpl> _Unifc;
};

}  // namespace reticolo