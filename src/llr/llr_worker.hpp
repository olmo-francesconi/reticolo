#pragma once

#include "lattice/lattice.hpp"
#include "action/action.hpp"
#include "tools/timer.hpp"
#include "tools/types.hpp"

#include <vector>
#include <numeric>
#include <random>
#include <complex>
#include <format>
#include <omp.h>
#include <cmath>

namespace LLR
{
    class llr_worker
    {
    private:
        uint _id;
        lattice<std::complex<double>> l;
        action::phi4 act;
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif;
        std::normal_distribution<double> norm;

    public:
        llr_worker(){};
        llr_worker(uint id, action::phi4_params par, vect4 sizes, uint seed) { init(id, par, sizes, seed); };

        void init(uint id, action::phi4_params par, vect4 sizes, uint seed);

        void randomize();

        void imag_thermalize();
        void thermalize();
        void MCMC_update();

        size_t memoryReport() const;
    };

    inline void llr_worker::init(uint id, action::phi4_params par, vect4 sizes, uint seed)
    {
        _id = id;
        l.init(sizes);

        // RNGs stuff
        rng.seed(seed);
        unif = std::uniform_real_distribution<double>(0.0, 1.0);
        norm = std::normal_distribution<double>(0.0, 1.0);

        // initialize the action
        act.init(&l, par);
    }

    inline void llr_worker::randomize()
    {
        for (size_t site = 0; site < l.getNsites(); site++)
            l[site] = std::complex<double>(norm(rng), norm(rng));
    }

    inline void llr_worker::MCMC_update()
    {
        int acc = 0;
        std::complex<double> old_S, new_S;
        std::complex<double> old_S_full, new_S_full;
        std::complex<double> old_phi;

        // LATICE UPDATE
        vect4 sizes = l.getSizes();
        vect4 coord;

        for (coord[_t] = 0; coord[_t] < sizes[_t]; coord[_t]++)
            for (coord[_x] = 0; coord[_x] < sizes[_x]; coord[_x]++)
                for (coord[_y] = 0; coord[_y] < sizes[_y]; coord[_y]++)
                    for (coord[_z] = 0; coord[_z] < sizes[_z]; coord[_z]++)
                    {
                        old_S = act.compute_S_loc(coord);
                        old_phi = l[coord];

                        // std::complex<double> dfield = std::polar(0.1 * sqrt(unif(rng)), 2.0 * M_PI * unif(rng));
                        std::complex<double> dfield(0.1 * norm(rng), 0.1 * norm(rng));

                        l[coord] += dfield;

                        new_S = act.compute_S_loc(coord);

                        if (exp(old_S.real() - new_S.real()) > unif(rng))
                            acc++;
                        else
                            l[coord] = old_phi;
                    }

        // MEASUREMENTS
        double phi_abs = 0.0;
        for (size_t i = 0; i < l.getNsites(); i++)
        {
            phi_abs += 0.5 * (l[i].real() * l[i].real() + l[i].imag() * l[i].imag());
        }
        phi_abs /= l.getNsites();

        std::complex<double> S = act.compute_S();
        // #pragma omp critical
        //         std::cout << std::format("{:6.4f}", (double)acc / l.getNsites()) << "\t"
        //                   << std::format("{:10.8e}", S.real()) << "\t"
        //                   << std::format("{:10.8e}", S.imag()) << "\t"
        //                   << std::format("{:10.8e}", phi_abs) << std::endl;
    }

    inline size_t llr_worker::memoryReport() const
    {
        size_t memory = 0;
        memory += l.memoryReport();

        return memory;
    }

} // namespace LLR
