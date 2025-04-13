
from generate import *
import json
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


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

    fig = plt.figure(figsize=(10, 7))

    gs = GridSpec(3, 2, width_ratios=[1.5, 1])
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1,0]), plt.subplot(gs[2,0]), \
           plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 1])]
            #  ,plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])]

    # Add title boxes to each subplot
    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("Experiment 1", xy=(0.5, 1.15), xycoords='axes fraction',
                fontsize=15, ha='center')

    axs[0].plot(np.arange(540), cats, label='cats', color='red', marker='o', markersize=2, linestyle='--')
    axs[1].plot(np.arange(540), hats, label='hats', color='blue', marker='o', markersize=2, linestyle='--')
    axs[2].plot(np.arange(540), cars, label='cars', color='green', marker='o', markersize=2, linestyle='--')

    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    axs[2].legend(fontsize=14)

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
                    fontsize=15, ha='center')

    axs[3].plot(np.arange(360), cats, label='cats', color='red', marker='o', markersize=2, linestyle='--')
    axs[4].plot(np.arange(360), hats, label='hats', color='blue', marker='o', markersize=2, linestyle='--')
    axs[5].plot(np.arange(360), cars, label='cars', color='green', marker='o', markersize=2, linestyle='--')

    axs[3].legend(fontsize=14)
    axs[4].legend(fontsize=14)
    axs[5].legend(fontsize=14)

    axs[3].set_xlim(0, 360)
    axs[4].set_xlim(0, 360)
    axs[5].set_xlim(0, 360)

    axs[3].set_ylim(0, 1)
    axs[4].set_ylim(0, 1)
    axs[5].set_ylim(0, 1)

    experiment = 4
    cats = []
    cars = []
    hats = []

    # with open('generate/json/trials_exp_4.json', 'r') as file:
    #     episodes = json.load(file)
    #
    # for block_num in range(12):
    #     trial_0 = episodes[block_num]['trials'][0]
    #     cats.extend([trial_0['P']] * 30)
    #     hats.extend([trial_0['C']] * 30)
    #     cars.extend([trial_0['B']] * 30)
    #

    # axs[6].annotate("Experiment 3", xy=(0.5, 1.15), xycoords='axes fraction',
    #                 fontsize=13, ha='center')
    # axs[6].plot(np.arange(360), cats, label='cats', color='red', marker='o', markersize=3)
    # axs[7].plot(np.arange(360), hats, label='hats', color='blue', marker='o', markersize=3)
    # axs[8].plot(np.arange(360), cars, label='cars', color='green', marker='o', markersize=3)
    #
    # axs[6].legend()
    # axs[7].legend()
    # axs[8].legend()
    #
    # axs[6].set_xlim(0, 360)
    # axs[7].set_xlim(0, 360)
    # axs[8].set_xlim(0, 360)
    #
    # axs[6].set_ylim(0, 1)
    # axs[7].set_ylim(0, 1)
    # axs[8].set_ylim(0, 1)



    axs[2].set_xlabel("Trial number", fontsize=14)
    axs[2].set_ylabel("Token probability", fontsize=14)

    plt.savefig("figures/task_contingencies.png")

    plt.show()


if __name__ == "__main__":
    plot_contingencies()