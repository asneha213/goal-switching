from behavior import *
from models import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pickle


def get_retro_measure_simulation(experiment, model_name, measure_name, optimal=False):
    sub_measures = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        if not optimal:
            with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
            params = model_fits['params']
            params = list(params.values())
        else:
            file_name = "results/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
            with open(file_name, "rb") as f:
                params = pickle.load(f)
                params = list(params[0].values())

        runmodel = RunModel(experiment, model_name, params)
        model_res = runmodel.get_model_res()

        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
        if measure_name == "retro_value":
            measure = subject_measures.get_mean_measure_condition("retro_value")
        elif measure_name == "retro_value_count":
            measure = subject_measures.get_individual_measure(measure_name)

        sub_measures.append(measure)
    return np.array(sub_measures)


def plot_retro_bias_per_condition(experiment, ax, model_name=None, compare=False):

    if model_name is None:
        sub_retro_bias = get_measure_experiment(experiment=experiment, measure_name="retro_value", \
                                            mode="mean_condition")
    else:
        sub_retro_bias = get_retro_measure_simulation(experiment=experiment, model_name=model_name, measure_name="retro_value")

    if compare:
        pros_retro_bias = ModelOptimizer(experiment=experiment, model_name="prospective").simulate_params(
            measure_name="retro_value")
        retro_retro_bias = ModelOptimizer(experiment=experiment, model_name="retrospective").simulate_params(
            measure_name="retro_value")
        data = [pros_retro_bias, sub_retro_bias, retro_retro_bias]
    else:
        data = [sub_retro_bias]

    if (experiment == 1) or (experiment == "instr_1"):
        conditions = ["80-20", "70-30", "60-40"]
    elif experiment == 2:
        conditions = ["75-25", "55-45"]

    if compare:
        mean_values = [np.mean(pros_retro_bias, axis=0), np.mean(sub_retro_bias, axis=0), np.mean(retro_retro_bias, axis=0)]
        std_dev_values = [np.std(pros_retro_bias, axis=0), np.std(sub_retro_bias, axis=0), np.std(retro_retro_bias, axis=0)] / np.sqrt(len(pros_retro_bias))
    else:
        mean_values = [np.mean(sub_retro_bias, axis=0)]
        std_dev_values = np.array([np.std(sub_retro_bias, axis=0)]) / np.sqrt(len(sub_retro_bias))

    if model_name is None:
        if compare:
            models = ["Prospective", "Behavior", "Retrospective"]
        else:
            models = [ "Behavior"]
    elif model_name == "momentum":
        models = ["TD-Momentum"]
    elif model_name == "td_persistence":
        models = ["TD-Persistence"]
    elif model_name == "prospective":
        models = ["Prospective"]
    elif model_name == "hybrid":
        models = ["Hybrid"]

    if (experiment == 1) or (experiment == "instr_1"):
        title = "Experiment 1"
    elif experiment == 2:
        title = "Experiment 2"
    ylabel = "Retrospectively biased choice"
    ylim = [0, 1]

    if compare:
        legend = True
    else:
        legend = False
    plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.18, legend=legend)


def plot_retro_bias_per_condition_instructions(ax):
    pros_retro_bias_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params(
        measure_name="retro_value")

    sub_retro_bias_1 = get_measure_experiment(experiment=1, measure_name="retro_value", \
                                            mode="mean_condition")
    sub_retro_bias_instr = get_measure_experiment(experiment="instr_1", measure_name="retro_value", \
                                            mode="mean_condition")

    data = [pros_retro_bias_1, sub_retro_bias_1, sub_retro_bias_instr]
    mean_values = [np.mean(pros_retro_bias_1, axis=0), np.mean(sub_retro_bias_1, axis=0), np.mean(sub_retro_bias_instr, axis=0)]
    std_dev_values = np.array([np.std(pros_retro_bias_1, axis=0), np.std(sub_retro_bias_1, axis=0), np.std(sub_retro_bias_instr, axis=0)]) / np.sqrt(len(sub_retro_bias_1))

    conditions = ["80-20", "70-30", "60-40"]
    models = ["Prospective", "No instructions", "Instructions"]
    ylabel = "Retrospectively biased choice"
    title = None
    ylim = [0, 1]
    legend = True
    plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                bar_width=0.18, legend=legend)


def plot_retro_bias_relation_to_progress_difference(experiment, ax, model_name=None):
    if model_name is None:
        sub_retro_count = get_measure_experiment(experiment=experiment, measure_name="retro_value_count", \
                                             mode="measure")
    else:
        sub_retro_count = get_retro_measure_simulation(experiment=experiment, model_name=model_name, measure_name="retro_value_count")
    m0 = sub_retro_count[:, 0, :]
    m1 = sub_retro_count[:, 1, :]
    if (experiment == 1) or (experiment == "instr_1"):
        m2 = sub_retro_count[:, 2, :]

    mean_m0 = np.nanmean(m0, axis=0)
    mean_m1 = np.nanmean(m1, axis=0)
    if (experiment == 1) or (experiment == "instr_1"):
        mean_m2 = np.nanmean(m2, axis=0)

    stderr_m0 = np.nanstd(m0, axis=0) / np.sqrt(m0.shape[0])
    stderr_m1 = np.nanstd(m1, axis=0) / np.sqrt(m1.shape[0])
    if (experiment == 1) or (experiment == "instr_1"):
        stderr_m2 = np.nanstd(m2, axis=0) / np.sqrt(m2.shape[0])

    if (experiment == 1) or (experiment == "instr_1"):
        ax.plot(range(7), mean_m0, label="80-20-20", color="red")
        ax.errorbar(x=range(7), y=mean_m0, yerr=stderr_m0, fmt='o', capsize=5, color="red")

        ax.plot(range(7), mean_m1, label="70-30-30", color="blue")
        ax.errorbar(x=range(7), y=mean_m1, yerr=stderr_m1, fmt='o', capsize=5, color="blue")

        ax.plot(range(7), mean_m2, label="60-40-40", color="green")
        ax.errorbar(x=range(7), y=mean_m2, yerr=stderr_m2, fmt='o', capsize=5, color="green")

    elif experiment == 2:
        ax.plot(range(7), mean_m0, label="75-25-25", color="red")
        ax.errorbar(x=range(7), y=mean_m0, yerr=stderr_m0, fmt='o', capsize=5, color="red")

        ax.plot(range(7), mean_m1, label="55-45-45", color="blue")
        ax.errorbar(x=range(7), y=mean_m1, yerr=stderr_m1, fmt='o', capsize=5, color="blue")

    ax.legend(fontsize=11)
    ax.set_xlabel("progress: retrospective - prospective", fontsize=10)
    ylim = [0, 1]
    ax.set_ylim(ylim)


def plot_retrospective_bias(model_name=None, axs=None, print=True):

    if axs is None:
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 5, width_ratios=[1, 0.8, 0.1, 0.7, 0.8])  # Set the height ratios for the subplots

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1], sharey=ax1)
        #plt.subplot(gs[0, 2])
        ax3 = plt.subplot(gs[0, 3], sharey=ax1)
        ax4 = plt.subplot(gs[0, 4], sharey=ax1)
    else:
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        ax4 = axs[3]

    # Experiment 1

    ## Plot retro bias comparison between behavior and model

    ax1.annotate("1", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax2.annotate("2", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax3.annotate("3", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax4.annotate("4", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')

    experiment = 1

    plot_retro_bias_per_condition(experiment=experiment, ax=ax1, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax2, model_name=model_name)

    # Experiment 2

    plot_retro_bias_per_condition(experiment=2, ax=ax3, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax4, model_name=model_name)

    if print:
        plt.tight_layout()
        plt.show()


def plot_retro_bias_compare_prospective_retrospective(model_name=None):
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 0.1, 0.7])  # Set the height ratios for the subplots

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 2], sharey=ax1)

    ax1.annotate("1", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax2.annotate("2", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')

    plot_retro_bias_per_condition(experiment=1, ax=ax1, model_name=model_name, compare=True)

    plot_retro_bias_per_condition(experiment=2, ax=ax2, model_name=model_name, compare=True)

    plt.tight_layout()
    plt.show()


def compare_behavior_model(seed):

    np.random.seed(seed)
    fig = plt.figure(figsize=(17, 24))
    gs = GridSpec(7, 5, width_ratios=[1, 0.8, 0.1, 0.7, 0.8], height_ratios=[1, 0.1, 1, 0.1, 1, 0.1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)
    ax3 = plt.subplot(gs[0, 3], sharey=ax1)
    ax4 = plt.subplot(gs[0, 4], sharey=ax1)

    ax5 = plt.subplot(gs[2, 0])
    ax6 = plt.subplot(gs[2, 1], sharey=ax5)
    ax7 = plt.subplot(gs[2, 3], sharey=ax5)
    ax8 = plt.subplot(gs[2, 4], sharey=ax5)

    ax9 = plt.subplot(gs[4, 0])
    ax10 = plt.subplot(gs[4, 1], sharey=ax9)
    ax11 = plt.subplot(gs[4, 3], sharey=ax9)
    ax12 = plt.subplot(gs[4, 4], sharey=ax9)

    ax13 = plt.subplot(gs[6, 0])
    ax14 = plt.subplot(gs[6, 1], sharey=ax13)
    ax15 = plt.subplot(gs[6, 3], sharey=ax13)
    ax16 = plt.subplot(gs[6, 4], sharey=ax13)



    #
    plot_retro_bias_per_condition(experiment=1, ax=ax1, model_name=None)
    plot_retro_bias_relation_to_progress_difference(experiment=1, ax=ax2, model_name=None)

    plot_retro_bias_per_condition(experiment=2, ax=ax3, model_name=None)
    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax4, model_name=None)

    model_name = "momentum"
    plot_retro_bias_per_condition(experiment=1, ax=ax5, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=1, ax=ax6, model_name=model_name)

    plot_retro_bias_per_condition(experiment=2, ax=ax7, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax8, model_name=model_name)

    model_name = "prospective"
    plot_retro_bias_per_condition(experiment=1, ax=ax9, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=1, ax=ax10, model_name=model_name)

    plot_retro_bias_per_condition(experiment=2, ax=ax11, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax12, model_name=model_name)

    model_name = "td_persistence"
    plot_retro_bias_per_condition(experiment=1, ax=ax13, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=1, ax=ax14, model_name=model_name)

    plot_retro_bias_per_condition(experiment=2, ax=ax15, model_name=model_name)
    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax16, model_name=model_name)



    # Add labels "A" and "B" at the top of the subplots
    ax1.annotate("BEHAVIOR", xy=(-0.3, 0.35), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    ax5.annotate("MOMENTUM", xy=(-0.3, 0.35), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    ax9.annotate("PROSPECTIVE", xy=(-0.3, 0.3), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    ax13.annotate("TD PERSISTENCE", xy=(-0.3, 0.2), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)

    #plt.tight_layout()
    #plt.show()

    plt.savefig("retro_bias_" + str(seed) + ".png")


def plot_retro_bias_progress_instructions():
    pros_retro_bias_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params(
        measure_name="retro_value_count")

    sub_retro_1 = get_measure_experiment(experiment=1, measure_name="retro_value_count", \
                                         mode="measure")
    sub_retro_instr = get_measure_experiment(experiment="instr_1", measure_name="retro_value_count", \
                                             mode="measure")


    pros_retro_mean_1 = [np.mean(pros_retro_bias_1[i], axis=0) for i in range(len(pros_retro_bias_1))]
    sub_retro_mean_1 = [np.mean(sub_retro_1[i], axis=0) for i in range(len(sub_retro_1))]
    sub_retro_mean_instr = [np.mean(sub_retro_instr[i], axis=0) for i in range(len(sub_retro_instr))]

    pros_retro_mean_1 = np.nanmean(pros_retro_mean_1, axis=0)
    retro_mean_1 = np.nanmean(sub_retro_mean_1, axis=0)
    retro_mean_instr = np.nanmean(sub_retro_mean_instr, axis=0)

    pros_retro_std_1 = np.nanstd(pros_retro_mean_1, axis=0) / np.sqrt(len(pros_retro_mean_1))
    retro_std_1 = np.nanstd(sub_retro_mean_1, axis=0) / np.sqrt(len(sub_retro_mean_1))
    retro_std_instr = np.nanstd(sub_retro_mean_instr, axis=0) / np.sqrt(len(sub_retro_mean_instr))


    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 0.8])  # Set the height ratios for the subplots

    ax2 = plt.subplot(gs[0, 0])

    ax1 = plt.subplot(gs[0, 1], sharey=ax2)

    ax1.plot(pros_retro_mean_1, color="blue", label="Prospective")
    ax1.errorbar(x=range(7), y=pros_retro_mean_1, yerr=pros_retro_std_1, fmt='o', capsize=5, color="blue")

    ax1.plot(retro_mean_1, color="red", label="No instructions")
    ax1.errorbar(x=range(7), y=retro_mean_1, yerr=retro_std_1, fmt='o', capsize=5, color="red")

    ax1.plot(retro_mean_instr, color="green", label="Instructions")
    ax1.errorbar(x=range(7), y=retro_mean_instr, yerr=retro_std_instr, fmt='o', capsize=5, color="green")

    ax1.set_xlabel("progress difference: retrospective - prospective")

    ax1.legend()

    plot_retro_bias_per_condition_instructions(ax2)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    model_name = "momentum"
    plot_retrospective_bias(model_name=model_name)
