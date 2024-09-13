#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load data
path = "/Users/olmo/software/source/reticolo/build/apps/test/meas/"
names = ["HMC", "Met"]

MonteCarlo_data = {}
Observables_data = {}
for name in names:
    f = h5py.File(path + name + ".h5", "r")
    MonteCarlo_data[name] = pd.DataFrame(np.array(f["MonteCarlo"]))
    Observables_data[name] = pd.DataFrame(np.array(f["Observables"]))

# plot action history
fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
fig.subplots_adjust(hspace=0)
for k, v in MonteCarlo_data.items():
    ax[0].plot(v["S_re"] / 4**4, lw=0.5, label=k)
    if k == "HMC":
        ax[1].plot(v["acceptance"].cumsum().div(v["acceptance"].index.to_series() + 1), lw=0.5, label=k)
    else:
        ax[1].plot(v["acceptance"], lw=0.5, label=k)
ax[0].set_ylabel(r"$Action/V$")
ax[1].set_ylabel(r"$Acceptance$")
ax[1].set_xlabel(r"$Iterations$")
ax[0].legend(loc=0)
plt.show()

# plot Observables history
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(hspace=0)
for k, v in Observables_data.items():
    ax[0].plot(v["phi2"], lw=0.5, label=k)
    ax[1].hist(v["phi2"], bins=1000, density=True, alpha=0.75, label=k)
ax[0].set_xlabel(r"$Iterations$")
ax[0].set_ylabel(r"$\phi^2$")
ax[1].set_xlabel(r"$\phi^2$")
ax[0].legend(loc=0)
plt.show()

# compare autocorrelations
fig, ax = plt.subplots(figsize=(12, 6))
for k, v in MonteCarlo_data.items():
    autocorr = []
    for t in range(0, 20):
        autocorr.append(v["S_re"].autocorr(t))
    ax.plot(autocorr, lw=0.5, label=k + "_S")
for k, v in Observables_data.items():
    autocorr = []
    for t in range(0, 20):
        autocorr.append(v["phi2"].autocorr(t))
    ax.plot(autocorr, lw=0.5, label=k + "_phi2")
ax.set_ylabel(r"$Autocorrelation$")
ax.set_xlabel(r"$lag$")
ax.legend(loc=0)
plt.show()
