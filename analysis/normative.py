from generate import *
from optimize_model import ModelOptimizer
import pickle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec



def normative_simulations(seed):
    """
    Normative simulations
    """

    walk = True
    episodes, token_probs = generate_normative_episode_trials(num_episodes=18, num_trials=30, walk=walk, seed=seed)

    vals = []
    experiment = "normative"

    model_name = 'prospective'
    res = ModelOptimizer(experiment=experiment, model_name=model_name, episodes=episodes).get_model_optimal_params()
    print(res)
    vals.append(res[1])

    model_name = 'momentum'
    res = ModelOptimizer(experiment=experiment, model_name=model_name, episodes=episodes).get_model_optimal_params()
    print(res)
    vals.append(res[1])

    with open("results/normative/" + str(seed) + "_fast_switch_performance.pkl", 'wb') as f:
        pickle.dump(vals, f)


def plot_task_design(axs):
    walk = True
    seed = 150
    episodes, token_probs = generate_normative_episode_trials(num_episodes=18, num_trials=30, walk=walk, seed=seed)

    tokenA = token_probs[0]
    tokenB = token_probs[1]
    tokenC = token_probs[2]

    ax1 = axs[0]
    ax2 = axs[1]

    ax1.plot(tokenA, label="Token A")
    ax1.plot(tokenB, label="Token B")
    ax1.plot(tokenC, label="Token C")

    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Probability")

    ax1.legend()

    seed = 200
    walk = "Slow"
    episodes, token_probs = generate_normative_episode_trials(num_episodes=18, num_trials=30, walk=walk, seed=seed)

    tokenA = token_probs[0]
    tokenB = token_probs[1]
    tokenC = token_probs[2]

    ax2.plot(tokenA, label="Token A")
    ax2.plot(tokenB, label="Token B")
    ax2.plot(tokenC, label="Token C")

    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Probability")

    ax2.set_ylim([0, 1])

    ax2.legend(loc="upper right")


def plot_normative_results():
    fig = plt.figure(figsize=(14, 7))

    gs = GridSpec(2, 3, width_ratios=[1, 0.1, 1], height_ratios=[1, 0.9])

    # Set the height ratios for the subplots

    axs0 = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]

    plot_task_design(axs0)

    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])]
    sims = []

    for seed in range(64):
        with open("results/normative/" + str(seed) + "_performance.pkl", 'rb') as f:
            res = pickle.load(f)
        sims.append(res)

    sims = -np.array(sims)

    axs[0].plot([30, 70], [30, 70], 'k--')

    axs[0].scatter(
        sims[:, 0],
        sims[:, 1],
        c='blue',  # Set the color of the markers
        edgecolors='black',  # Set the color of the marker edges
        s=100,  # Set the marker size
        alpha=0.7,  # Set the transparency of markers
        label='Data Points'
    )

    axs[0].set_xlabel("Performance - Prospective")
    axs[0].set_ylabel("Performance - TD-Momentum")

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)
    axs0[0].annotate("Gaussian random walks", xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=11, ha='center', va='center', bbox=title_box_props)

    sims = []

    for seed in range(64):
        with open("results/normative/" + str(seed) + "_no_walk_performance.pkl", 'rb') as f:
            res = pickle.load(f)
        sims.append(res)

    sims = -np.array(sims)

    axs[1].plot([30, 60], [30, 60], 'k--')
    axs[1].scatter(
        sims[:, 0],
        sims[:, 1],
        c='blue',  # Set the color of the markers
        edgecolors='black',  # Set the color of the marker edges
        s=100,  # Set the marker size
        alpha=0.7,  # Set the transparency of markers
        label='Data Points'
    )
    axs[1].set_xlabel("Performance - Prospective")
    axs[1].set_ylabel("Performance - TD-Momentum")

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)
    axs0[1].annotate("Abrupt probability switches", xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=11, ha='center', va='center', bbox=title_box_props)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_normative_results()

