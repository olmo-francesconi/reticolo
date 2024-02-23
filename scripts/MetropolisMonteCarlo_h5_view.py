#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = h5py.File(
    "/Users/olmo/software/source/reticolo/build/apps/MetropolisMonteCarlo_out/meas/MetropolisMonteCarlo.h5", "r"
)
data = pd.DataFrame(np.array(f["MetropolisMonteCarlo/MonteCarlo"]))

autocorr = []
for t in range(1, 100):
    autocorr.append(data["S_re"].autocorr(t))

plt.plot(autocorr)
plt.show()
data.plot()
plt.show()
