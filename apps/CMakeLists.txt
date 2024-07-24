### BOSE GAS MONTECARLO ###

add_executable(bose_met BoseGas_MetropolisMonteCarlo.cpp)
target_link_libraries(bose_met PRIVATE reticolo)

add_executable(bose_hmc BoseGas_HMCMonteCarlo.cpp)
target_link_libraries(bose_hmc PRIVATE reticolo)


### LLR ###

add_executable(LLRBoseGas LLRBoseGas.cpp)
target_link_libraries(LLRBoseGas PRIVATE reticolo)


### LQGR ###

add_executable(WeakFieldEuclideanGR_MCMC WeakFieldEuclideanGR_MCMC.cpp)
target_link_libraries(WeakFieldEuclideanGR_MCMC PRIVATE reticolo)

add_executable(WeakFieldEuclideanGR_HMC WeakFieldEuclideanGR_HMC.cpp)
target_link_libraries(WeakFieldEuclideanGR_HMC PRIVATE reticolo)