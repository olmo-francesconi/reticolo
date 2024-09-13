#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "/Users/olmo/software/source/reticolo/build/apps/test/meas/"
names = {"Metropolis", "HMC"}
MonteCarlo_data = {}
Observables_data = {}

# read data
for name in names:
    print(name)
    f = h5py.File(path + name + ".h5", "r")
    MonteCarlo_data[name] = pd.DataFrame(np.array(f[name + "/MonteCarlo"]))
    Observables_data[name] = pd.DataFrame(np.array(f[name + "/Observables"]))

# plot action history
for name in names:
    MonteCarlo_data[name]["S"].plot(alpha=0.75, lw=0.1, label=name)
plt.title("$Action$")
plt.legend(loc=0)
plt.show()

# plot action history
for name in names:
    MonteCarlo_data[name]["acceptance"].plot(alpha=0.75, lw=0.1, label=name)
plt.title("$Acceptance$")
plt.legend(loc=0)
plt.show()

# # plot Observables history
# for name in names:
#     Observables_data[name]["phi2"].plot(alpha=0.5, lw=0.1)
# plt.show()
# # plot Observables Histogram
# for name in names:
#     Observables_data[name]["phi2"].hist(bins=100, alpha=0.5, lw=0.1)
# plt.show()
# compare autocorrelations
for name in names:
    autocorr = []
    for t in range(0, 200):
        autocorr.append(MonteCarlo_data[name]["S"][50000:].autocorr(t))
    plt.plot(autocorr, label=name)
plt.title("$Autocorrelation$")
plt.legend(loc=0)
plt.show()
