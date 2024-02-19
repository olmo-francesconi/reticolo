#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = h5py.File("../build/4_4_4_4/llr/meas/llr.h5", "r")


for i in range(10):
    for j in range(12):
        data = np.array(f["run{0}/ak_{1:0>3d}".format(i, j)])
        plt.plot(data)


plt.show()
