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

    print(pd.DataFrame(f[key]["mc"]["S_re"]))
    print(pd.DataFrame(f[key]["mc"]["dS_re"]).mean())

    # data.plot(title=key)
    pd.DataFrame(f[key]["mc"]["dS_re"]).plot(kind="hist", bins=100)
    pd.DataFrame(f[key]["mc"]["S_re"]).plot()

plt.show()
