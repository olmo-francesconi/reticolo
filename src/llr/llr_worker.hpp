#pragma once

#include "lattice/lattice.hpp"
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
        lattice<ComplexD> l;
        action::phi4 act;
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif;
        std::normal_distribution<double> norm;

    public:
        llr_worker(){};
        llr_worker(uint id, vect4 sizes, uint seed) { init(id, sizes, seed); };

        void init(uint id, vect4 sizes, uint seed);

        void randomize(double scale = 1.0);

        void imag_thermalize();
        void thermalize();
        void MCMC_update();

        size_t memoryReport() const;
    };

    inline void llr_worker::init(uint id, vect4 sizes, uint seed)
    {
        _id = id;
        l.init(sizes);

        // RNGs stuff
        rng.seed(seed);
        unif = std::uniform_real_distribution<double>(0.0, 1.0);
        norm = std::normal_distribution<double>(0.0, 1.0);

        // initialize the action
        act = action::phi4(1.0, 9.0, 1.0);
    }

    inline void llr_worker::randomize(double scale)
    {
        for (size_t site = 0; site < l.getNsites(); site++)
            l[site] = scale * ComplexD(norm(rng), norm(rng));
    }

    inline void llr_worker::MCMC_update()
    {
        int acc = 0;
        ComplexD old_S, new_S;
        ComplexD old_field;

        // LATICE UPDATE
        vect4 sizes = l.getSizes();
        vect4 coord;

        int steps = 1000;

        double t1, t2, ratio = 0.0;

        Timer t;

        rng.seed(1);
        randomize();
        t.reset();
        for (size_t i = 0; i < steps; i++)
            for (coord[_t] = 0; coord[_t] < sizes[_t]; coord[_t]++)
                for (coord[_x] = 0; coord[_x] < sizes[_x]; coord[_x]++)
                    for (coord[_y] = 0; coord[_y] < sizes[_y]; coord[_y]++)
                        for (coord[_z] = 0; coord[_z] < sizes[_z]; coord[_z]++)
                        {
                            old_S = act.compute_S_loc(l, coord);
                            old_field = l[coord];

                            // // ComplexD dfield = std::polar(0.1 * sqrt(unif(rng)), 2.0 * M_PI * unif(rng));
                            // ComplexD dfield(0.1 * norm(rng), 0.1 * norm(rng));
                            ComplexD dfield(0.1, 0.0);
                            l[coord] += dfield;

                            new_S = act.compute_S_loc(l, coord);

                            if (exp(old_S.real() - new_S.real()) > unif(rng))
                                acc++;
                            else
                                l[coord] = old_field;
                        }
        t1 = t.elapsed_ms();
        std::cout << std::format("{:+e}", t1) << " ms\t" << acc << std::endl;

        acc = 0;
        rng.seed(1);
        randomize();
        t.reset();
        for (size_t i = 0; i < steps; i++)
            for (coord[_t] = 0; coord[_t] < sizes[_t]; coord[_t]++)
                for (coord[_x] = 0; coord[_x] < sizes[_x]; coord[_x]++)
                    for (coord[_y] = 0; coord[_y] < sizes[_y]; coord[_y]++)
                        for (coord[_z] = 0; coord[_z] < sizes[_z]; coord[_z]++)
                        {
                            // ComplexD dfield = std::polar(0.1 * sqrt(unif(rng)), 2.0 * M_PI * unif(rng));
                            // ComplexD dfield(0.1 * norm(rng), 0.1 * norm(rng));
                            ComplexD dfield(0.1, 0.0);

                            ComplexD dS = act.compute_dS_loc(l, dfield, coord);

                            if (exp(-dS.real()) > unif(rng))
                            {
                                acc++;
                                l[coord] += dfield;
                            }
                        }
        t2 = t.elapsed_ms();
        std::cout << std::format("{:+e}", t2) << " ms\t" << acc << std::endl;
        std::cout << std::format("{:>13.2f}", 100 * t2 / t1) << " %" << std::endl;

        // MEASUREMENTS
        // double phi_abs = 0.0;
        // for (size_t i = 0; i < l.getNsites(); i++)
        // {
        //     phi_abs += 0.5 * (l[i].real() * l[i].real() + l[i].imag() * l[i].imag());
        // }
        // phi_abs /= l.getNsites();

        // ComplexD S = act.compute_S(l);
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
