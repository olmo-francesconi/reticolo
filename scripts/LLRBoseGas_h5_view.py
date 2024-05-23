#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

# Workspace = "/Volumes/Extreme SSD/Physics/LLR/BoseGas_4x4x4x4"
Workspace = "/Users/olmo/Desktop/bose_gas_llr/4_4_4_4"
mu = [0.8]  # [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 2.0]
mu_lbl = ["{:.1f}".format(u) for u in mu]
mu_map = dict(zip(mu_lbl, mu))
print(mu_lbl)
NIntervals = 64
DeltaS = 0.0025 * 8**4

print("Generating ak values plot")
fig_ak, ax_ak = plt.subplots(1, figsize=(12, 6))
# draw intervals borders
ax_ak.axvline(0, lw=0.1, c="k", alpha=0.2)
for interval in range(NIntervals):
    ax_ak.axvline(DeltaS * (interval + 0.5), lw=0.2, ls=":", c="k", alpha=0.2)
    ax_ak.axvline(DeltaS * (interval + 1), lw=0.2, c="k", alpha=0.2)

for u in mu_lbl:
    llr_file = h5py.File(Workspace + "/" + u + "/llr.h5", "r")
    RunNames = llr_file.keys()

    data = []
    for name in RunNames:
        tmp_data = []
        for interval in range(NIntervals):
            tmp_data.append(np.array(llr_file["{}/[{}]ak".format(name, interval)])[-1])
        data.append(tmp_data)
    data = pd.DataFrame(data)
    ax_ak.errorbar(
        np.arange(NIntervals) * DeltaS + 0.5 * DeltaS,
        data.mean(),
        data.std(),
        lw=0.5,
        ls="--",
        marker="x",
        markersize=2,
        capsize=2,
        label=u,
    )

ax_ak.legend(loc=0)
plt.show()


#################
###    OLD    ###
#################

# print("generating history plots..")
# fig_hist, ax_hist = plt.subplots(2, figsize=(12, 6), sharex=True)
# fig_hist.subplots_adjust(hspace=0)
# # draw intervals borders
# ax_hist[1].axhline(0, lw=0.1, c="k", alpha=0.2)
# for interval in range(NIntervals):
#     ax_hist[1].axhline(DeltaS * (interval + 0.5), lw=0.2, ls=":", c="k", alpha=0.2)
#     ax_hist[1].axhline(DeltaS * (interval + 1), lw=0.2, c="k", alpha=0.2)

# for name in ["replica_0"]:
#     for interval in range(NIntervals):
#         avg_data = pd.DataFrame(np.array(llr_file["{}/[{}]mc".format(name, interval)]))
#         var_data = pd.DataFrame(np.array(llr_file["{}/[{}]mc_var".format(name, interval)]))
#         x = np.arange(0, len(avg_data)) + 0.01 * interval
#         for lbl, ax in [["S_re", ax_hist[0]], ["S_im", ax_hist[1]]]:
#             y = avg_data[lbl]
#             yerr = np.sqrt(var_data[lbl])
#             ax.errorbar(x, y, yerr, lw=0.5, alpha=0.5, ls="--", marker="x", markersize=2, capsize=2)

# ax_hist[0].set_ylabel(r"$S_{Re}$")
# ax_hist[1].set_ylabel(r"$S_{Im}$")
# ax_hist[1].set_xlabel(r"$Iterations$")
