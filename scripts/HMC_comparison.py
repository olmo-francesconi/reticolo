#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "/Users/olmo/software/source/reticolo/build/apps/test/meas/"

MonteCarlo_data = {}
Observables_data = {}

names = ["HMC", "Met"]

for name in names:
    f = h5py.File(path + name + ".h5", "r")
    MonteCarlo_data[name] = pd.DataFrame(np.array(f["MonteCarlo"]))
    Observables_data[name] = pd.DataFrame(np.array(f["Observables"]))

# plot action history
for k, v in MonteCarlo_data.items():
    v["acceptance"].plot(lw=0.5, label=k)
    (v["S_re"] / 4**4).plot(lw=0.5, label=k)

plt.title("$Action$")
plt.legend(loc=0)
plt.show()

# # plot Observables history
# for k, v in Observables_data.items():
#     v["phi2"].plot(lw=0.5, label=k)
# plt.show()
# # plot Observables Histogram
# for k, v in Observables_data.items():
#     v["phi2"].hist(bins=100, alpha=0.5)
# plt.show()

# compare autocorrelations
for k, v in MonteCarlo_data.items():
    autocorr = []
    for t in range(0, 20):
        autocorr.append(v["S_re"][100000:].autocorr(t))
    plt.plot(autocorr, label=k)
plt.title("$Autocorrelation$")
plt.legend(loc=0)

plt.show()
