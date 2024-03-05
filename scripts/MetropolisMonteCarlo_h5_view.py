#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

steps = np.arange(10, 101, 10)
stepsizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
path = "/Users/olmo/software/source/reticolo/build/apps/test/meas/"

for step in steps:
    for stepsize in stepsizes:
        name = "HMC_{}_{}".format(step, stepsize)
        print(name)
        f = h5py.File(path + name + ".h5", "r")

        data = pd.DataFrame(np.array(f[name + "/MonteCarlo"]))

        autocorr = []
        for t in range(1, 200):
            autocorr.append(data["S_re"][10000:].autocorr(t))

        plt.plot(autocorr, label=name)

    plt.legend(loc=0)
    plt.show()
