#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "/Users/olmo/software/source/reticolo/build/apps/BoseGas_"
Simulations = {
    "Metropolis": {
        "dir": "Metropolis",
        "file": "Met_bose.h5",
        "dataset": "met",
        "PropWidth": [0.1 * (i + 1) for i in range(10)],
    },
    "HMC": {
        "dir": "HMC",
        "file": "HMC_bose.h5",
        "dataset": "hmc",
        "Steps": [1, 2, 5, 10],
        "StepSize": [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05],
        # "StepSize": [0.05],
    },
}
MonteCarlo_data = {}
Observables_data = {}

# read data
print("Loading data..")

# info = Simulations["Metropolis"]
# print("- Metropolis")
# f = h5py.File(path + info["dir"] + "/Measurements/" + info["file"], "r")
# for PropW in info["PropWidth"]:
#     name = "{name}_{prop_width:.1f}".format(name=info["dataset"], prop_width=PropW)
#     print("  - ", name)
#     MonteCarlo_data[name] = pd.DataFrame(np.array(f[name + "/MonteCarlo"]))
#     Observables_data[name] = pd.DataFrame(np.array(f[name + "/Observables"]))

info = Simulations["HMC"]
print("- HMC")
f = h5py.File(path + info["dir"] + "/Measurements/" + info["file"], "r")
for Steps in info["Steps"]:
    for StepSize in info["StepSize"]:
        name = "{name}_{steps}_{stepsize:.4f}".format(name=info["dataset"], steps=Steps, stepsize=StepSize)
        print("  - ", name)
        MonteCarlo_data[name] = pd.DataFrame(np.array(f[name + "/MonteCarlo"]))
        Observables_data[name] = pd.DataFrame(np.array(f[name + "/Observables"]))

# plot action history
print("Plotting Action")
for label, data in MonteCarlo_data.items():
    print("  - ", label)
    data["S_re"].plot(lw=0.5, label=label + "_Re")
plt.title("$Action$")
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
print("Plotting Autocorrelations")
hmc_autocorr = {}
for label, data in MonteCarlo_data.items():
    print("  - ", label)
    autocorr = []
    for t in range(0, 50):
        autocorr.append(data["S_re"][-10000:].autocorr(t))
    plt.plot(autocorr, lw=0.5, ls=("--" if "hmc" in label else "-"), label=label)
    if "hmc" in label:
        SplitLbl = label.split("_")
        TrajLen = float(SplitLbl[1]) * float(SplitLbl[2])
        hmc_autocorr[SplitLbl[1] + "_" + SplitLbl[2]] = {
            "Steps": float(SplitLbl[1]),
            "StepSize": float(SplitLbl[2]),
            "TrajLen": float(SplitLbl[1]) * float(SplitLbl[2]),
            "data": autocorr,
        }
plt.title("$Autocorrelation$")
# plt.legend(loc=0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x = []
y = []
z = []
for lbl, data in hmc_autocorr.items():
    x.append(data["Steps"])
    y.append(data["StepSize"])
    z.append(data["data"][4])
ax.scatter(x, y, np.log(z), marker="o")
plt.show()
