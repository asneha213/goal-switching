import sys

from fit_behavior import *
from models import *

import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def get_sample_random(model_name):

    if (model_name == "prospective") or (model_name == "momentum"):
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma}

    elif model_name == "hybrid":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        gamma = np.random.uniform(0.6, 1)
        wa = np.random.uniform(0, 1)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a, 'gamma': gamma, 'wa': wa}

    elif model_name == "td_persistence":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_0 = np.random.uniform(-1, 1)
        beta_g = np.random.uniform(0, 10)
        beta_a = np.random.uniform(0, 10)
        params = {'alpha': alpha, 'alpha_c': alpha_c, 'beta_0': beta_0, 'beta_g': beta_g, 'beta_a': beta_a}

    elif model_name == "momentum_just_goal" or model_name == "prospective_just_goal":
        alpha = np.random.uniform(0, 1)
        alpha_c = np.random.uniform(0, 1)
        beta_c = np.random.uniform(0, 10)
        beta_g = np.random.uniform(0, 10)
        gamma = np.random.uniform(0.6, 1)
        params = {'alpha': alpha, 'beta_c': beta_c, 'beta_g': beta_g, 'gamma': gamma, 'alpha_c': alpha_c}



    return params


def simulate_and_recover(experiment, model_name, params, seed):
    runmodel = RunModel(experiment, model_name, params)
    model_res = runmodel.get_model_res()
    fits = BehaviorFits(experiment, seed, model_name, model_res)
    r_params, vals = fits.fit_optuna()
    r_params = list(r_params.values())

    precovery = {}
    precovery['original'] = list(params.values())
    precovery['recovered'] = r_params

    file = open('results_latent/param_recovery/' + model_name + "_" + str(experiment) + "_" + str(seed) + "_parameter_recovery.pkl", "wb")

    pickle.dump(precovery, file)
    return r_params


def recover_params(experiment, model_name, seed):
    np.random.seed(seed)
    params = get_sample_random(model_name)
    simulate_and_recover(experiment, model_name, params, seed)


def recover_model(experiment, model_name, seed):
    modelfits = np.zeros(4)
    np.random.seed(seed)
    params = get_sample_random(model_name)
    runmodel = RunModel(experiment, model_name, params)
    model_res = runmodel.get_model_res()

    models = ['td_persistence', 'momentum', 'prospective', 'hybrid']

    for i in range(len(models)):
        model_fname = models[i]
        fits = BehaviorFits(experiment, seed, model_fname, model_res)
        r_params, vals = fits.fit_optuna()
        modelfits[i] = vals

    file = open('results_latent/model_recovery/' + model_name + "_" + str(seed) + "_" + str(experiment) +  "_model_recovery.pkl",
                "wb")

    pickle.dump(modelfits, file)


def plot_model_recovery(experiment):
    models = ['td_persistence', 'momentum', 'prospective', 'hybrid']

    confusion = np.zeros((len(models), len(models)))

    for i in range(len(models)):
        model_fname = models[i]
        for seed in range(100):

            with open('results_latent/model_recovery/' + model_fname + "_" + str(seed) + "_" + str(
                    experiment) + "_model_recovery.pkl",
                      "rb") as f:

                modelfit = pickle.load(f)
                samples = 540

                modelfit[0] = 2 * modelfit[0] + 3 * np.log(samples)
                modelfit[1] = 2 * modelfit[1] + 4 * np.log(samples)
                modelfit[2] = 2 * modelfit[2] + 4 * np.log(samples)
                modelfit[3] = 2 * modelfit[3] + 5 * np.log(samples)

                confusion[i][np.argmin(modelfit)] += 1

    cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    cm_inv = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]

    models = ['TD-Persistence', 'TD-Momentum', 'Prospective', 'Hybrid']

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.set(font_scale=0.8)

    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=models, yticklabels=models, ax=axs[0])
    axs[0].set_xlabel('Fit')
    axs[0].set_ylabel('Simulated')
    axs[0].set_title('Pr(Fit | Simulated)', fontsize=10)

    sns.heatmap(cm_inv, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=models, yticklabels=False, ax=axs[1])
    axs[1].set_xlabel('Fit')
    axs[1].set_title('Pr(Simulated | Fit)', fontsize=10)

    plt.savefig('figures/model_recovery.png')

    plt.show()


def plot_parameter_recovery():

    original = []
    recovered = []

    num_sims = 1000
    for i in range(num_sims):
        try:
            with open('results_latent/param_recovery/momentum_1_' + str(i) + "_parameter_recovery.pkl", "rb") as f:
                precovery = pickle.load(f)
        except:
            continue


        original.append(precovery['original'])
        recovered.append(precovery['recovered'])

    original = np.array(original)
    recovered = np.array(recovered)

    print(original.shape)

    fig, axs = plt.subplots(1, 6, figsize=(14, 3))

    param_names = ["Learning rate", "Choice kernel", \
                   "Switching cost", "Goal Softmax", "Action softmax", "Discount factor"]

    for k in range(6):
        print(pg.corr(original[:, k], recovered[:, k], method="spearman"))
        axs[k].scatter(original[:, k] , recovered[:, k], s=5)
        axs[k].set_xlabel(param_names[k] + " - Original")
        axs[k].set_ylabel(param_names[k] + " - Recovered")
        if k==3 or k==4:
            axs[k].set_ylim(-1, 11)
            x = np.linspace(0, 10, 100)
        elif k==2:
            axs[k].set_ylim(-1, 1)
            x = np.linspace(-1, 1, 100)
        elif k==5:
            axs[k].set_ylim(0.55, 1.1)
            x = np.linspace(0.5, 1, 100)
        else:
            axs[k].set_ylim(-0.1, 1.1)
            x = np.linspace(0, 1, 100)
        y = x

        r = pg.corr(original[:, k], recovered[:, k], method="spearman")['r'].to_numpy()[0]

        # Create a plot
        axs[k].plot(x, y, color='red', label=f'R={r:.2f}')
        axs[k].legend(fontsize=10, loc='upper left')

    # for k in range(5):
    #     print(pg.corr(original[:, k], recovered[:, k], method="spearman"))
    #     axs[k].scatter(original[:, k] , recovered[:, k], s=5)
    #     axs[k].set_xlabel(param_names[k] + " - Original")
    #     axs[k].set_ylabel(param_names[k] + " - Recovered")
    #     if k==2 or k==3:
    #         axs[k].set_ylim(-1, 11)
    #         x = np.linspace(0, 10, 100)
    #     elif k==4:
    #         axs[k].set_ylim(0.55, 1.1)
    #         x = np.linspace(0.5, 1, 100)
    #     else:
    #         axs[k].set_ylim(-0.1, 1.1)
    #         x = np.linspace(0, 1, 100)
    #     y = x
    #
    #     r = pg.corr(original[:, k], recovered[:, k], method="spearman")['r'].to_numpy()[0]
    #
    #     # Create a plot
    #     axs[k].plot(x, y, color='red', label=f'R={r:.2f}')
    #     axs[k].legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    plt.savefig('figures/parameter_recovery.png')

    plt.show()


if __name__ == "__main__":
    
    machine = 'local'
    #machine = 'server'
    plot = False
    plot = True

    if machine == 'server':
        seed = int(sys.argv[1])
        experiment = 1
        model_name = sys.argv[2]
        param = int(sys.argv[3])
    else:
        seed = 1
        model_name = "momentum"
        model_name = "hybrid"
        model_name = "prospective"
        model_name = "td_persistence"
        param = 1

    if not plot:
        if param == 1:
           recover_params(experiment=1, model_name=model_name, seed=seed)
        else:
           recover_model(experiment=1, model_name=model_name, seed=seed)
    else:
        if param == 1:
            plot_parameter_recovery()
        else:
            plot_model_recovery(experiment=1)

