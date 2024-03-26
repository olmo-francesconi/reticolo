#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Workspace = "/Volumes/Extreme SSD/Physics/LLR/BoseGas_6"

ak_file = h5py.File(Workspace + "/llr/meas/llr.h5", "r")
Datasetnames = ak_file.keys()
intervals = len(Datasetnames)

print("Generating ak values 3D view")
fig_ak = plt.figure(figsize=(12, 6))
ax_ak = plt.axes(projection="3d")
for j in range(intervals):
    data = []
    tmp_data = np.array(ak_file["ak_{}".format(j)])
    data.append(tmp_data)
    ax_ak.plot(np.arange(len(tmp_data)), tmp_data, zs=j, zdir="x")

print("generating history plots..")
# plot action history
MC_Samples = 2000
DeltaS = 0.64
fig_action, ax_action = plt.subplots(2, figsize=(12, 6), sharex=True)
ax_action[1].axhline(0, lw=0.1, c="k", alpha=0.2)
fig_action.subplots_adjust(hspace=0)
for interval in range(intervals):
    ax_action[1].axhline(DeltaS * (interval + 0.5), lw=0.2, ls=":", c="k", alpha=0.5)
    ax_action[1].axhline(DeltaS * (interval + 1), lw=0.2, c="k", alpha=0.5)
    f = h5py.File(Workspace + "/llr/meas/llr_{}.h5".format(interval), "r")
    data = pd.DataFrame(np.array(f["MonteCarlo"]))
    grouper = np.arange(len(data)) // MC_Samples
    x = np.arange(0, len(set(grouper)))
    for lbl, ax in [["S_re", ax_action[0]], ["S_im", ax_action[1]]]:
        y = data[lbl].groupby(grouper).mean()
        y_std = data[lbl].groupby(grouper).std() / np.sqrt(MC_Samples)
        ax.errorbar(x, y, y_std, alpha=0.25, lw=0.5, ls="--", marker="x", markersize=2, capsize=2)
ax_action[0].set_ylabel(r"$S_{Re}$")
ax_action[1].set_ylabel(r"$S_{Im}$")
ax_action[1].set_xlabel(r"$Iterations$")
ax_action[0].legend(loc=0)
ax_action[1].legend(loc=0)


plt.show()
