from os import mkdir
from os.path import isdir
from pathlib import Path
from sys import argv
from matplotlib import pyplot as plt
import numpy as np

SUBPLOT_NAMES = ["Position", "Orientation", "Velocity", "Angular Velocity"]

# Make directory to save plots
if not isdir("plots"):
    mkdir("plots")

file = argv[1]
file_data = None

mass = None

# Read data from files
with open(file, "r") as f:
    file_data = [next(f).split()]
    if mass is None:
        # Parse parameters
        mass = float(file_data[0][0])
        nsteps = int(file_data[0][1])
        nepisodes = int(file_data[0][2])
        nruns = int(file_data[0][3])
    file_data.append(list(map(float, next(f).split(","))))

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

    file_data.append(np.nanmedian(runs, axis=0))
    file_data.append(np.nanpercentile(runs, 25, axis=0))
    file_data.append(np.nanpercentile(runs, 75, axis=0))


fig, axs = plt.subplots(2, 2, figsize=(16, 10))

eps = np.arange(nepisodes)
for file_idx, file in enumerate(argv[1:]):
    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.plot(eps, file_data[2][:, i + 2], label="Hybrid model error (per episode)")

        ax.plot(
            (0, nepisodes - 1),
            2 * (file_data[2][0, i + 2],),
            "g-",
            label="Faulty model error",
        )
        ax.plot(
            (0, nepisodes - 1),
            2 * (file_data[1][i + 2],),
            "m-",
            label="Perfect model error",
        )

        ax.fill_between(eps, file_data[3][:, i + 2], file_data[4][:, i + 2], alpha=0.2)
        ax.set_title(SUBPLOT_NAMES[i])
        ax.set(xlabel="Episodes", ylabel="Median Total Error (Scaled)")
        ax.legend()
        ax.grid()

fig.suptitle(f"mass={mass}kg, {nruns} runs, {nsteps} steps/episode")
plt.show()
fig.savefig(f"plots/plot_{Path(file).stem}.png")
