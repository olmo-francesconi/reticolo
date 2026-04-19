#pragma once

#if defined(RETICOLO_USE_OPENMP)
#include <omp.h>
#else
using omp_lock_t = int;

inline void omp_init_lock(omp_lock_t* lock) {
    if (lock != nullptr) {
        *lock = 0;
    }
}

inline void omp_destroy_lock(omp_lock_t* /*lock*/) {}

inline void omp_set_lock(omp_lock_t* /*lock*/) {}

inline void omp_unset_lock(omp_lock_t* /*lock*/) {}
#endif
