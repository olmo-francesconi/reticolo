#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = h5py.File("../build/4_4_4_4/meas/llr.h5", "r")

runs = f.keys()
# aks = f[runs[0]].keys()
print(runs[1])
# print(aks)

print(np.array(f["run0/ak_001"]))

# for r in runs:
#     ak = f[r].keys()
#     # data = []

#     for a in ak:
#         tmp = pd.DataFrame(f[r][a]).to_numpy()
#         print(pd.DataFrame(f[r][a]))
#         # np.concatenate((data, tmp), axis=0)

#     # print(data)
