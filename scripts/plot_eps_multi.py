from os import mkdir
from os.path import isdir
from pathlib import Path
from sys import argv
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import re

NN_STRUCTURE_REG = r"^reg_(.*?)_.*"
TYPE_MASS_REG = r"reg_.*?_(.*?).txt"

SUBPLOT_NAMES = ["Position", "Orientation", "Velocity", "Angular Velocity"]

# WEIGHTS = np.array([0.8, 0.4, 0.4, 0.8])
WEIGHTS = np.array([4, 4, 1, 1])

# Make directory to save plots
if not isdir("plots"):
    mkdir("plots")

files = argv[1:]
file_data = {}

mass = None
min_error = None

# Read data from files
for file in files:
    with open(file, "r") as f:
        file_data[file] = [next(f).split()]
        if mass is None:
            # Parse parameters
            mass = float(file_data[file][0][0])
            nsteps = int(file_data[file][0][1])
            nepisodes = int(file_data[file][0][2])
            nruns = int(file_data[file][0][3])
            min_error = list(map(float, next(f).split(",")))
        else:
            next(f)

        runs = []
        for line in f:
            runs.append([list(map(float, x.split(","))) for x in line.split()])
        runs = np.array(runs)

        # Remove rows where optimizer failed
        mask = np.any(runs[:, :, 0] == 1, axis=1)
        print(f"{file} skipped {(mask == True).sum()} runs")
        runs[mask] = np.nan

        # Scale errors for episodes that didn't complete all steps
        cost_weight = nsteps / runs[:, :, 1]
        runs[:, :, 2:] *= cost_weight[:, :, np.newaxis]

        file_data[file].append(np.nanmedian(runs, axis=0))
        file_data[file].append(np.nanpercentile(runs, 25, axis=0))
        file_data[file].append(np.nanpercentile(runs, 75, axis=0))


fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.35)

axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, :]),
]

eps = np.arange(nepisodes)
for file_idx, file in enumerate(files):
    for i in range(4):
        ax = axs[i]  # axs[i // 2, i % 2]
        nn_structure = re.search(NN_STRUCTURE_REG, Path(file).stem).group(1)

        if min_error is not None:
            ax.plot(
                (0, nepisodes - 1),
                2 * (file_data[file][1][0, i + 2],),
                "g-",
                label="Faulty model error",
            )
            ax.plot(
                (0, nepisodes - 1),
                2 * (min_error[i + 2],),
                "m-",
                label="True model error",
            )

        ax.plot(
            eps,
            file_data[file][1][:, i + 2],
            label=f"Hybrid model {nn_structure} error",
        )

        ax.fill_between(
            eps, file_data[file][2][:, i + 2], file_data[file][3][:, i + 2], alpha=0.2
        )
        ax.set_title(SUBPLOT_NAMES[i])
        ax.set(xlabel="Episodes", ylabel="Median Total Error (Scaled)")
        ax.legend()
        ax.grid()

    if min_error is not None:
        axs[4].plot(
            (0, nepisodes - 1),
            2 * (sum(WEIGHTS[j] * file_data[file][1][0, j + 2] for j in range(4)),),
            "g-",
            label="Faulty model error",
        )

        axs[4].plot(
            (0, nepisodes - 1),
            2 * (sum(WEIGHTS[j] * min_error[j + 2] for j in range(4)),),
            "m-",
            label="True model error",
        )

        min_error = None

    axs[4].plot(
        eps,
        file_data[file][1][:, 2:] @ WEIGHTS,
        label=f"Hybrid model {nn_structure} error",
    )

    axs[4].fill_between(
        eps,
        file_data[file][2][:, 2:] @ WEIGHTS,
        file_data[file][3][:, 2:] @ WEIGHTS,
        alpha=0.2,
    )

    axs[4].set_title("Weighted Error Sum")
    axs[4].set(xlabel="Episodes", ylabel="Weighted Sum of Errors")
    axs[4].legend()
    axs[4].grid()

fig.suptitle(f"mass={mass}kg, {nruns} runs, {nsteps} steps/episode")
plt.show()
fig.savefig(f"plots/multi_plot_{re.search(TYPE_MASS_REG, files[0]).group(1)}.png")
