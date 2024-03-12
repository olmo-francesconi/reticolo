/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/MetropoliMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

    Montecarlo workspace folder structure:
        WeakFieldEuclideanGR
        ├── meas
        │   └── WeakFieldEuclideanGR.h5
        └── reticolo.log
******************************************************************************/

#include <omp.h>

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
// #include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/HMC.hpp"
// #include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder */
    std::string OutPath = "test";

    uint NUpdates = std::stoi(argv[1]);

    {
        std::cout << "hmc\n";
        std::string                         RunName = std::format("HMC");
        Lattice<ComplexD, 4>                Lattice({4, 4, 4, 4});
        action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
        action::RelativisticBoseGas         Action(Lattice, Par);
        montecarlo::HMC                     HMCWorker(RunName, Action, Lattice, 0, OutPath);
        HMCWorker.setParams(std::stoi(argv[2]), std::stod(argv[3]));
        HMCWorker.run(NUpdates, 0, 1, true, false, 1.0, false);
    }
    {
        std::cout << "met\n";
        std::string                         RunName = std::format("Met");
        Lattice<ComplexD, 4>                Lattice({4, 4, 4, 4});
        action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
        action::RelativisticBoseGas         Action(Lattice, Par);
        montecarlo::Metropolis              MetropolisWorker(RunName, Action, Lattice, 0, OutPath);
        MetropolisWorker.setParams(0.5);
        MetropolisWorker.run(10 * NUpdates, 0, 10, true, false, 1.0, false);
    }

    // {
    //     std::cout << "hmc\n";
    //     std::string                          RunName = std::format("HMC");
    //     Lattice<HField<RealD>, 4>            Lattice({4, 4, 4, 4});
    //     action::WeakFieldEuclideanGR::Params Par(5.7);
    //     action::WeakFieldEuclideanGR         Action(Lattice, Par);
    //     montecarlo::HMC                      HMCWorker(RunName, Action, Lattice, 0, OutPath);
    //     HMCWorker.setParams(std::stoi(argv[2]), std::stod(argv[3]));
    //     HMCWorker.run(NUpdates, 0, 1, true, false, 1.0, false);
    // }
    // {
    //     std::cout << "met\n";
    //     std::string                          RunName = std::format("Met");
    //     Lattice<HField<RealD>, 4>            Lattice({4, 4, 4, 4});
    //     action::WeakFieldEuclideanGR::Params Par(5.7);
    //     action::WeakFieldEuclideanGR         Action(Lattice, Par);
    //     montecarlo::Metropolis               MetropolisWorker(RunName, Action, Lattice, 0, OutPath);
    //     MetropolisWorker.setParams(0.1);
    //     MetropolisWorker.run(NUpdates, 0, 1, true, false, 1.0, false);
    // }
    return EXIT_SUCCESS;
}

// /*
//  *  This example writes 64 datasets to a HDF5 file, using multiple threads
//  *  (OpenMP).  Each thread grab the lock while it tries to call HDF5 functions
//  *  to write out dataset.  In this way, the HDF5 calls are serialized, while
//  *  the calculation part is in parallel.   This is one of the ways to do
//  *  OpenMP computation with HDF.  As long as not to do HDF I/O in parallel,
//  *  it is safe to use HDF.
//  */

// #include <hdf5.h>
// #include <math.h>
// #include <omp.h>

// #define NUM_THREADS 4
// #define NUM_MDSET   16
// #define FILE        "SDS.h5"
// #define NX          5 /* dataset dimensions */
// #define NY          18
// #define RANK        2

// void CalcWriteData(hid_t, hid_t, hid_t);

// /*Global variable, OpenMP lock*/
// omp_lock_t lock;

// int main(void) {
//     hid_t   fid;                 /* file and dataset handles */
//     hid_t   datatype, dataspace; /* handles */
//     hsize_t dimsf[2];            /* dataset dimensions */
//     herr_t  status;
//     int     i;

//     /*
//      * Create a new file using H5F_ACC_TRUNC access,
//      * default file creation properties, and default file
//      * access properties.
//      */
//     fid = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

//     /*
//      * Describe the size of the array and create the data space for fixed
//      * size dataset.
//      */
//     dimsf[0] = NX;
//     dimsf[1] = NY;
//     dataspace = H5Screate_simple(RANK, dimsf, NULL);

//     /*
//      * Define datatype for the data in the file.
//      * We will store little endian INT numbers.
//      */
//     datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
//     status = H5Tset_order(datatype, H5T_ORDER_LE);

//     /*Disable dynamic allocation of threads*/
//     omp_set_dynamic(0);

//     /*Allocate threads*/
//     omp_set_num_threads(NUM_THREADS);

//     /*Initialize lock*/
//     omp_init_lock(&lock);

// /*Each thread grab one iteration in the for loop and call function
//  * CaclWriteData*/
// #pragma omp parallel default(shared)
//     {
// #pragma omp for
//         for (i = 0; i < NUM_THREADS; i++) {
//             CalcWriteData(fid, dataspace, datatype);
//         }
//     }

//     /*Finished lock mechanism, destroy it*/
//     omp_destroy_lock(&lock);

//     /*
//      * Close/release resources.
//      */
//     H5Sclose(dataspace);
//     H5Tclose(datatype);
//     H5Fclose(fid);

//     return 0;
// }

// /*Each thread will call this function independantly.  They calculate dataset
//  *and then write it out to hdf, for NUM_MDSET times */
// void CalcWriteData(hid_t fid, hid_t dataspace, hid_t datatype) {
//     double data[NX][NY];
//     hid_t  dataset;
//     char   dname[16];
//     int    tid;
//     int    i, j, k;

//     tid = omp_get_thread_num();

//     for (i = 0; i < NUM_MDSET; i++) {
//         /*Weird calculation to extend computing time*/
//         for (j = 0; j < NX; j++) {
//             for (k = 0; k < NY; k++)
//                 data[j][k] = (pow(sin(tid), 2.0) + pow(cos(tid), 2.0)) * (tid + i) +
//                              (pow(123456789.0, 8.0) - pow(123456789.0, 8.0));
//         }
//         sprintf(dname, "tid%d-dataset%d", tid, i);

//         /*Serialize HDF dataset writing using OpenMP lock*/
//         omp_set_lock(&lock);

//         dataset = H5Dcreate(fid, dname, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//         H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
//         H5Dclose(dataset);

//         /*Release lock*/
//         omp_unset_lock(&lock);
//     }
// }
