/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/impl/lattice.cpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "lattice/lattice.hpp"

namespace reticolo
{
    template <typename T_field, size_t dim>
    inline void lattice<T_field, dim>::init(uintvect<dim> sizes)
    {
        N = sizes;

        n_sites = std::accumulate(N.begin(), N.end(), 1, std::multiplies<uint>());

        for (int i = 0; i < dim - 1; i++)
            sub_vols[i] = std::accumulate(N.begin() + i + 1, N.end(), 1, std::multiplies<uint>());
        sub_vols.back() = 1;

        field.clear();
        field.resize(n_sites);
    };

    template class lattice<RealD, 1>;
    template class lattice<RealD, 2>;
    template class lattice<RealD, 3>;
    template class lattice<RealD, 4>;

    template class lattice<RealF, 1>;
    template class lattice<RealF, 2>;
    template class lattice<RealF, 3>;
    template class lattice<RealF, 4>;

    template class lattice<ComplexD, 1>;
    template class lattice<ComplexD, 2>;
    template class lattice<ComplexD, 3>;
    template class lattice<ComplexD, 4>;

    template class lattice<ComplexF, 1>;
    template class lattice<ComplexF, 2>;
    template class lattice<ComplexF, 3>;
    template class lattice<ComplexF, 4>;

    template class lattice<HField, 4>;

} // namespace reticolo