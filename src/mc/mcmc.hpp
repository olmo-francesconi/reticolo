#pragma once

#include "tools/types.hpp"

namespace reticolo
{
    namespace montecarlo
    {
        template <template <typename, typename> class Action, typename FieldType, ComplexValue ActionType, size_t dim>
        class Markov_worker
        {
        private:
        public:
            Markov_worker(){};

            init();

            run(uint steps);
        };

    } // namespace montecarlo

} // namespace reticolo