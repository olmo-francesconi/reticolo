#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load data
path_tmplt = "/Users/olmo/software/source/reticolo/build/apps/LQGR_metropolis/meas/"

Sizes = [
    [6, 3],
    [6, 4],
    [6, 5],
    [6, 6],
    # [6, 7],
    # [6, 8],
]

plot_group_size = 10000

MonteCarlo_data = {}
Observables_data = {}
Volumes = {}
Iterations = {}

print("loading data..")
for t, l in Sizes:
    path = path_tmplt.format(T=t, L=l)
    name = "{T}x{L}^3".format(T=t, L=l)
    print("  {}".format(name))
    f = h5py.File(path + name + ".h5", "r")
    MonteCarlo_data[name] = pd.DataFrame(np.array(f["MonteCarlo"]))
    Observables_data[name] = pd.DataFrame(np.array(f["Observables"]))
    Iterations[name] = pd.DataFrame(np.arange(len(MonteCarlo_data[name])), columns=["iter"])
    Volumes[name] = t * l**3


print("generating history plots..")
# plot action history
fig_action, ax_action = plt.subplots(2, figsize=(12, 6), sharex=True)
fig_action.subplots_adjust(hspace=0)
for k, v in MonteCarlo_data.items():
    print("  {}".format(k))
    grouper = np.arange(len(v)) // plot_group_size
    x = Iterations[k]["iter"].groupby(grouper).mean()
    for data, ax in [[v["S"] / Volumes[k], ax_action[0]], [v["acceptance"], ax_action[1]]]:
        y = data.groupby(grouper).mean()
        y_std = data.groupby(grouper).std()
        ax.plot(x, y, lw=0.5, label=k)
        ax.fill_between(x, y + 2.0 * y_std, y - 2.0 * y_std, alpha=0.1, lw=0.5)
ax_action[0].set_ylabel(r"$Action/V$")
ax_action[1].set_ylabel(r"$Acceptance$")
ax_action[1].set_xlabel(r"$Iterations$")
ax_action[0].legend(loc=0)
ax_action[1].legend(loc=0)


# # plot Observables history
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.subplots_adjust(hspace=0)
# for k, v in Observables_data.items():
#     ax[0].plot(v["phi2"], lw=0.5, label=k)
#     ax[1].hist(v["phi2"], bins=1000, density=True, alpha=0.75, label=k)
# ax[0].set_xlabel(r"$Iterations$")
# ax[0].set_ylabel(r"$\phi^2$")
# ax[1].set_xlabel(r"$\phi^2$")
# ax[0].legend(loc=0)
# plt.show()

print("generating autocorrelation plots..")
# compare autocorrelations
fig_autocorr, ax_autocorr = plt.subplots(figsize=(12, 6))

# print("  early time autocorr..")
# for k, v in MonteCarlo_data.items():
#     print("    {}".format(k))
#     autocorr = []
#     for t in range(0, 100):
#         autocorr.append(v["S"][1000000:2000000].autocorr(t))
#     ax_autocorr.plot(autocorr, ls="-", label=k + "_early")

print("  late time autocorr..")
for k, v in MonteCarlo_data.items():
    print("    {}".format(k))
    autocorr = []
    for t in range(0, 100):
        autocorr.append(v["S"][-1000000:].autocorr(t))
    ax_autocorr.plot(autocorr, ls="--", label=k + "_late")

ax_autocorr.set_ylabel(r"$Autocorrelation$")
ax_autocorr.set_xlabel(r"$lag$")
ax_autocorr.legend(loc=0)

print("showing plots..")
plt.show()
