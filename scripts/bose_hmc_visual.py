#!/usr/bin/env python3

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for l in [4, 6, 8, 10, 12]:
    info = {
        "path": "/Volumes/WD BLACK/Physics/bose_hmc/{}^4".format(l),
        "FileName": "HMC",
        "Dataset": "hmc",
        "Steps": [1, 2, 5, 10, 20, 50],
        "TrajLen": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    }
    MonteCarlo_data = {}
    Observables_data = {}

    fig_size = (24, 18)

    # Setup folders
    plot_dir = os.path.join(info["path"], "plots")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # read data
    print("Loading data..")
    for traj_len in info["TrajLen"]:
        f = h5py.File("{}/measurements/{}_{:.3f}.h5".format(info["path"], info["FileName"], traj_len), "r")
        MonteCarlo_data[traj_len] = {}
        Observables_data[traj_len] = {}
        for Steps in info["Steps"]:
            StepSize = traj_len / Steps
            DataSetName = "{name}_{steps}".format(name=info["Dataset"], steps=Steps)
            MonteCarlo_data[traj_len][DataSetName] = pd.DataFrame(np.array(f[DataSetName + "/MonteCarlo"]))
            Observables_data[traj_len][DataSetName] = pd.DataFrame(np.array(f[DataSetName + "/Observables"]))

    # plot action history
    print("Plotting Action")
    fig_autocorr_comp, ax_autocorr_comp = plt.subplots(figsize=fig_size)

    for traj_len in info["TrajLen"]:
        fig, ax = plt.subplots()
        fig_autocorr, ax_autocorr = plt.subplots()

        for label, data in MonteCarlo_data[traj_len].items():
            (data["S_re"] / (l**4)).plot(ax=ax, lw=0.5, label=label)
            autocorr = []
            for t in range(0, 50):
                autocorr.append(data["S_re"][-10000:].autocorr(t))
            ax_autocorr.plot(autocorr, lw=0.5, label=label)
            ax_autocorr_comp.plot(
                np.arange(50) * int(label.split("_")[-1]), autocorr, lw=0.5, label=label + "_" + str(traj_len)
            )

        ax.set_title("Action - traj_len = {}".format(traj_len))
        ax.legend(loc=0)
        fig.savefig(info["path"] + "/plots/action_{:.6f}.pdf".format(traj_len))

        ax_autocorr.set_title("Action autocorrelation - traj_len = {}".format(traj_len))
        ax_autocorr.set_ylim(1e-4, 1.1)
        ax_autocorr.set_yscale("log")
        ax_autocorr.legend(loc=0)
        fig_autocorr.savefig(info["path"] + "/plots/action_{:.6f}_autocorr.pdf".format(traj_len))

    ax_autocorr_comp.set_title("Action autocorrelation - traj_len = {}".format(traj_len))
    ax_autocorr_comp.set_ylim(1e-4, 1.1)
    ax_autocorr_comp.set_yscale("log")
    ax_autocorr_comp.legend(loc=0)
    fig_autocorr_comp.savefig(info["path"] + "/plots/action_autocorr_comp.pdf".format(traj_len))
    plt.close()

    # plot Observables history
    print("Plotting Observables")
    fig_autocorr_comp, ax_autocorr_comp = plt.subplots(figsize=fig_size)

    for traj_len in info["TrajLen"]:
        fig, ax = plt.subplots()
        fig_autocorr, ax_autocorr = plt.subplots()
        for label, data in Observables_data[traj_len].items():
            data["phi2"].plot(ax=ax, lw=0.5, label=label)
            autocorr = []
            for t in range(0, 50):
                autocorr.append(data["phi2"][10000:].autocorr(t))
            ax_autocorr.plot(autocorr, lw=0.5, label=label)
            ax_autocorr_comp.plot(
                np.arange(50) * int(label.split("_")[-1]), autocorr, lw=0.5, label=label + "_" + str(traj_len)
            )

        ax.set_title("phi^2 - traj_len = {}".format(traj_len))
        ax.legend(loc=0)
        fig.savefig(info["path"] + "/plots/phi2_{:.6f}.pdf".format(traj_len))

        ax_autocorr.set_title("phi^2 autocorrelation - traj_len = {}".format(traj_len))
        ax_autocorr.set_ylim(1e-4, 1.1)
        ax_autocorr.set_yscale("log")
        ax_autocorr.legend(loc=0)
        fig_autocorr.savefig(info["path"] + "/plots/phi2_{:.6f}_autocorr.pdf".format(traj_len))

    ax_autocorr_comp.set_title("phi^2 autocorrelation - traj_len = {}".format(traj_len))
    ax_autocorr_comp.set_ylim(1e-4, 1.1)
    ax_autocorr_comp.set_yscale("log")
    ax_autocorr_comp.legend(loc=0)
    fig_autocorr_comp.savefig(info["path"] + "/plots/phi2_autocorr_comp.pdf".format(traj_len))
    plt.close()
