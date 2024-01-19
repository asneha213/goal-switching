from generate import *
import json
from matplotlib import pyplot as plt


def plot_contingencies():

    experiment = 1
    cats = []
    cars = []
    hats = []
    with open('generate/json/trials_exp_1.json', 'r') as file:
        episodes = json.load(file)

    for block_num in range(18):
        trial_0 = episodes[block_num]['trials'][0]
        cats.extend([trial_0['P']] * 30)
        hats.extend([trial_0['C']] * 30)
        cars.extend([trial_0['B']] * 30)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 7))

    gs = GridSpec(3, 2, width_ratios=[1.5, 1])
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]), \
           plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 1])]

    # Add title boxes to each subplot
    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("Experiment 1", xy=(0.5, 1.15), xycoords='axes fraction',
                fontsize=12, ha='center', bbox=title_box_props)

    axs[0].plot(np.arange(540), cats, label='cats', color='red', marker='o', markersize=3)
    axs[1].plot(np.arange(540), hats, label='hats', color='blue', marker='o', markersize=3)
    axs[2].plot(np.arange(540), cars, label='cars', color='green', marker='o', markersize=3)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_xlim(0, 540)
    axs[1].set_xlim(0, 540)
    axs[2].set_xlim(0, 540)

    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)


    cats = []
    cars = []
    hats = []

    with open('generate/json/trials_exp_2.json', 'r') as file:
        episodes = json.load(file)

    for block_num in range(12):
        trial_0 = episodes[block_num]['trials'][0]
        cats.extend([trial_0['P']] * 30)
        hats.extend([trial_0['C']] * 30)
        cars.extend([trial_0['B']] * 30)

    axs[3].annotate("Experiment 2", xy=(0.5, 1.15), xycoords='axes fraction',
                    fontsize=12, ha='center', bbox=title_box_props)
    axs[3].plot(np.arange(360), cats, label='cats', color='red', marker='o', markersize=3)
    axs[4].plot(np.arange(360), hats, label='hats', color='blue', marker='o', markersize=3)
    axs[5].plot(np.arange(360), cars, label='cars', color='green', marker='o', markersize=3)

    axs[3].legend()
    axs[4].legend()
    axs[5].legend()

    axs[3].set_xlim(0, 360)
    axs[4].set_xlim(0, 360)
    axs[5].set_xlim(0, 360)

    axs[3].set_ylim(0, 1)
    axs[4].set_ylim(0, 1)
    axs[5].set_ylim(0, 1)

    axs[2].set_xlabel("Trial number")
    axs[2].set_ylabel("Token probability")

    plt.show()


if __name__ == "__main__":
    plot_contingencies()