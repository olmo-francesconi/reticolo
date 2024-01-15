#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import pandas as pd

f = h5py.File("../build/4_4_4_4/meas/llr_worker[000].h5", "r")
# f = h5py.File("../build/16_16_16_16/meas/llr_worker[000].h5", "r")

keys = f.keys()
print(keys)

for key in keys:
    print(f[key].keys())

    data = pd.DataFrame(f[key]["mc"][()])

    data.plot(title=key)

plt.show()
