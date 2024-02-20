#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = h5py.File("../build/apps/BoseGasLLR/llr/meas/llr.h5", "r")

replicas = 8
intervals = 48

ax = plt.figure().add_subplot(projection="3d")

for j in range(intervals):
    data = []
    for i in range(replicas):
        tmp_data = np.array(f["BoseGasLLR_{0}/ak_{1:0>3d}".format(i, j)])
        data.append(tmp_data)
        shift = 0.05 * (replicas / 2 - i)
        ax.plot(np.arange(len(tmp_data)), tmp_data, zs=j + shift, zdir="x")
    avg_data = np.average(data, axis=0)
    std_data = np.std(data, axis=0)

    ax.plot(np.arange(len(avg_data)), avg_data, zs=j, zdir="x", color="black", lw=0.5)

plt.show()
