#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd


Workspaces = [
    "/Volumes/Extreme SSD/Physics/LLR/BoseGas_4x4x4x4/hmc",
    "/Volumes/Extreme SSD/Physics/LLR/BoseGas_4x4x4x4/met",
    "/Volumes/Extreme SSD/Physics/LLR/BoseGas_4x4x4x4/hmc+met",
]
lbls = {Workspaces[0]: "hmc", Workspaces[1]: "met", Workspaces[2]: "hmc+met"}
NIntervals = 64
fig_autocorr, ax3d_autocorr = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"})
ax_map = dict(zip(Workspaces, ax3d_autocorr))
print("generating autocorrelation plots..")
for Workspace in Workspaces:
    print(Workspace)
    llr_file = h5py.File(Workspace + "/llr.h5", "r")
    RunNames = llr_file.keys()
    lag_max = 50
    z = []
    raw_data = {}
    print("lag: ", end="", flush=True)
    for lag in range(lag_max):
        print(lag, end=", ", flush=True)
        autocorr = []
        for name in RunNames:
            tmp = []
            for interval in range(NIntervals):
                filename = Workspace + "/raw_data/{}/measurements/llr_worker[{:0>3}].h5".format(name, interval)
                raw_data_file = h5py.File(filename, "r")
                if filename not in raw_data.keys():
                    raw_data[filename] = pd.DataFrame(np.array(raw_data_file["RM_0/MonteCarlo"]))
                tmp.append(raw_data[filename]["S_re"].autocorr(lag))
            autocorr.append(tmp)
        df = pd.DataFrame(autocorr)
        z.append(df.mean())
    x = range(NIntervals)
    y = range(lag_max)
    X, Y = np.meshgrid(x, y)
    Z = np.array(z)
    ax_map[Workspace].plot_surface(Y, X, Z, alpha=0.5, label=lbls[Workspace])
    ax_map[Workspace].plot_wireframe(Y, X, Z, lw=0.1, color="k", label=lbls[Workspace])
    ax_map[Workspace].legend(loc=0)
    print("done!")
plt.show()
