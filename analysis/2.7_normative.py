from models import *
from generate import *
import pingouin as pg
from optimize_model import ModelOptimizer
import pickle
from matplotlib import pyplot as plt

import sys


def norm_sims(seed):

    seed = int(sys.argv[1])
    #seed = 100

    walk = "Fast"
    episodes = generate_normative_episode_trials(num_episodes=18, num_trials=30, walk=walk, seed=seed)

    vals = []
    experiment = "normative"

    model_name = 'hierarchy'
    res = ModelOptimizer(experiment=experiment, model_name=model_name, episodes=episodes).get_model_optimal_params()
    print(res)
    vals.append(res[1])

    model_name = 'td_hierarchy'
    res = ModelOptimizer(experiment=experiment, model_name=model_name, episodes=episodes).get_model_optimal_params()
    print(res)
    vals.append(res[1])

    with open("results/normative/" + str(seed) + "_fast_switch_performance.pkl", 'wb') as f:
        pickle.dump(vals, f)


def plot_norm_res():

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    sims = []

    for seed in range(64):
        with open("results/normative/" + str(seed) + "_performance.pkl", 'rb') as f:
            res = pickle.load(f)
        sims.append(res)

    sims = -np.array(sims)

    axs[0].plot([30, 70], [30, 70], 'k--')
    axs[0].plot(sims[:, 0], sims[:, 1], 'o')
    axs[0].set_xlabel("Performance - Prospective")
    axs[0].set_ylabel("Performance - TD-Momentum")
    axs[0].set_title("Gaussian random walks")

    sims = []

    for seed in range(64):
        with open("results/normative/" + str(seed) + "_no_walk_performance.pkl", 'rb') as f:
            res = pickle.load(f)
        sims.append(res)

    sims = -np.array(sims)

    axs[1].plot([30, 60], [30, 60], 'k--')
    axs[1].plot(sims[:, 0], sims[:, 1], 'o')
    axs[1].set_xlabel("Performance - Prospective")
    #axs[1].set_ylabel("Performance - TD-Momentum")
    axs[1].set_title("Slow probability switches")


    sims = []

    for seed in range(64):
        with open("results/normative/" + str(seed) + "_fast_switch_performance.pkl", 'rb') as f:
            res = pickle.load(f)
        sims.append(res)

    sims = -np.array(sims)

    axs[2].plot([30, 50], [30, 50], 'k--')
    axs[2].plot(sims[:, 0], sims[:, 1], 'o')
    axs[2].set_xlabel("Performance - Prospective")
    #axs[1].set_ylabel("Performance - TD-Momentum")
    axs[2].set_title("Fast probability switches")


    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_norm_res()

