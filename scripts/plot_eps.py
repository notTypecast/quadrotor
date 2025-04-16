from os import mkdir
from os.path import isdir
from sys import argv
from matplotlib import pyplot as plt
import numpy as np

MODEL_NAMES_ORDERED = ["Dropout & Variance", "Dropout without Variance", "No Dropout"]

if len(MODEL_NAMES_ORDERED) != len(argv) - 1:
    print("Number of arguments must match size of model names list")
    print(f"Provided model names: {MODEL_NAMES_ORDERED}")
    exit(101)

SUBPLOT_NAMES = ["Position", "Orientation", "Velocity", "Angular Velocity"]

# Make directory to save plots
if not isdir("plots"):
    mkdir("plots")

files = {}

mass = None

# Read data from files
for file in argv[1:]:
    with open(file, "r") as f:
        files[file] = [next(f).split()]
        if mass is None:
            # Parse parameters
            mass = float(files[file][0][0])
            nsteps = float(files[file][0][1])
            nepisodes = int(files[file][0][2])
            nruns = int(files[file][0][3])

        runs = []
        for line in f:
            runs.append([list(map(float, x.split(","))) for x in line.split()])

        runs = np.array(runs)

        # Remove rows where optimizer failed
        mask = np.any(runs[:, :, 0] == 1, axis=1)
        print(f"{file} skipped {(mask == True).sum()} runs")
        runs[mask] = np.nan

        cost_weight = nsteps / runs[:, :, 1]
        runs[:, :, 2:] *= cost_weight[:, :, np.newaxis]

        files[file].append(np.nanmedian(runs, axis=0))
        files[file].append(np.nanpercentile(runs, 25, axis=0))
        files[file].append(np.nanpercentile(runs, 75, axis=0))

fig, axs = plt.subplots(2, 2, figsize=(16, 10))

eps = np.arange(nepisodes)
for file_idx, file in enumerate(argv[1:]):
    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.plot(eps, files[file][1][:, i + 2], label=MODEL_NAMES_ORDERED[file_idx])
        ax.fill_between(
            eps, files[file][2][:, i + 2], files[file][3][:, i + 2], alpha=0.2
        )
        ax.set_title(SUBPLOT_NAMES[i])
        ax.set(xlabel="Episodes", ylabel="Median Total Error (Scaled)")
        ax.legend()
        ax.grid()

fig.suptitle(f"mass={mass}kg, {nruns} runs")
plt.show()
fig.savefig(f"plots/episode_plot_{mass}.png")
