import numpy as np

from behavior import *
from models import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pickle


def plot_switches_complete():

    # Experiment 1

    """
    Get interaction between performance and condition for subjects
    """

    sub_switches_complete_1 = get_measure_experiment(experiment=1, measure_name="switches_completion", mode="condition")

    sub_switches_complete_2 = get_measure_experiment(experiment=2, measure_name="switches_completion", mode="condition")

    data_1 = None
    data_2 = None

    fig = plt.figure(figsize=(5, 4))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1])  # Set the height ratios for the subplots

    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    conditions = ["80-20", "70-30", "60-40"]
    mean_values = [np.nanmean(sub_switches_complete_1, axis=0)]
    std_dev_values = np.array([np.nanstd(sub_switches_complete_1, axis=0)]) / np.sqrt(len(sub_switches_complete_1))
    models = ["Behavior"]
    title = "Experiment 1"
    ylabel = "Switches after suit completion"
    ylim = [0, 10]

    plot_comparative_bar_plot(axs[0], data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim, bar_width=0.19, draw_data=False)

    conditions = ["75-25", "55-45"]
    mean_values = [np.nanmean(sub_switches_complete_2, axis=0)]
    std_dev_values = np.array([np.nanstd(sub_switches_complete_2, axis=0)]) / np.sqrt(len(sub_switches_complete_2))
    models = ["Behavior"]
    title = "Experiment 2"
    ylabel = "Switches after suit completion"
    ylim = [0, 24]
    plot_comparative_bar_plot(axs[1], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim, bar_width=0.2, draw_data=False)

    #plt.tight_layout()
    plt.show()


def get_retro_measure_simulation(experiment, model_name, measure_name):
    sub_measures = []
    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)
    for subject_num in range(num_subjects):
        if experiment == 1 and subject_num in [0, 6, 11, 21, 23, 27]:
            continue

        subject_id = subject_names[subject_num]
        with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
            model_fits = pickle.load(f)
        params = model_fits['params']
        params = list(params.values())
        runmodel = RunModel(experiment, model_name, params)
        model_res = runmodel.get_model_res()

        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
        if measure_name == "retro_value":
            measure = subject_measures.get_mean_measure_condition("retro_value")
        elif measure_name == "retro_value_count":
            measure = subject_measures.get_individual_measure(measure_name)

        sub_measures.append(measure)
    return np.array(sub_measures)


def plot_retro_bias_comparison_with_prospective_model(experiment, ax, model_name=None, compare=False):
    if model_name is None:
        sub_retro_bias = get_measure_experiment(experiment=experiment, measure_name="retro_value", \
                                            mode="mean_condition")
    else:
        sub_retro_bias = get_retro_measure_simulation(experiment=experiment, model_name=model_name, measure_name="retro_value")

    pros_retro_bias = ModelOptimizer(experiment=experiment, model_name="prospective").simulate_params(measure_name="retro_value")

    if compare:
        data = [pros_retro_bias, sub_retro_bias]
    else:
        data = [sub_retro_bias]

    if experiment == 1:
        conditions = ["80-20", "70-30", "60-40"]
    elif experiment == 2:
        conditions = ["75-25", "55-45"]

    if compare:
        mean_values = [np.mean(pros_retro_bias, axis=0), np.mean(sub_retro_bias, axis=0)]
        std_dev_values = np.array([np.std(pros_retro_bias, axis=0), np.std(sub_retro_bias, axis=0)]) / np.sqrt(len(sub_retro_bias))
    else:
        mean_values = [np.mean(sub_retro_bias, axis=0)]
        std_dev_values = np.array([np.std(sub_retro_bias, axis=0)]) / np.sqrt(len(sub_retro_bias))

    if model_name is None:
        if compare:
            models = ["Prospective", "Behavior"]
        else:
            models = [ "Behavior"]
    elif model_name == "momentum":
        #models = ["Prospective", "TD-Momentum"]
        models = ["TD-Momentum"]
    elif model_name == "td_persistence":
        #models = ["Prospective", "TD-Persistence"]
        models = ["TD-Persistence"]
    elif model_name == "prospective":
        #models = ["Prospective-optimal", "Prospective-behavior"]
        models = ["Prospective"]

    if experiment == 1:
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


def plot_retro_bias_relation_to_progress_difference(experiment, ax, model_name=None):
    if model_name is None:
        sub_retro_count = get_measure_experiment(experiment=experiment, measure_name="retro_value_count", \
                                             mode="measure")
    else:
        sub_retro_count = get_retro_measure_simulation(experiment=experiment, model_name=model_name, measure_name="retro_value_count")
    m0 = sub_retro_count[:, 0, :]
    m1 = sub_retro_count[:, 1, :]
    if experiment == 1:
        m2 = sub_retro_count[:, 2, :]

    mean_m0 = np.nanmean(m0, axis=0)
    mean_m1 = np.nanmean(m1, axis=0)
    if experiment == 1:
        mean_m2 = np.nanmean(m2, axis=0)

    stderr_m0 = np.nanstd(m0, axis=0) / np.sqrt(m0.shape[0])
    stderr_m1 = np.nanstd(m1, axis=0) / np.sqrt(m1.shape[0])
    if experiment == 1:
        stderr_m2 = np.nanstd(m2, axis=0) / np.sqrt(m2.shape[0])


    if experiment == 1:
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


def plot_retro_bias_progress_concat(experiment, ax, model_name=None):
    if model_name is None:
        sub_retro_count = get_measure_experiment(experiment=experiment, measure_name="retro_value_count", \
                                             mode="measure")
    else:
        sub_retro_count = get_retro_measure_simulation(experiment=experiment, model_name=model_name, measure_name="retro_value_count")
    m0 = sub_retro_count[:, 0, :]
    m1 = sub_retro_count[:, 1, :]
    if experiment == 1:
        m2 = sub_retro_count[:, 2, :]


    if experiment == 1:
        m0 = np.concatenate((m0, m1, m2), axis=0)
    elif experiment == 2:
        m0 = np.concatenate((m0, m1), axis=0)

    mean_m = np.nanmean(m0, axis=0)
    stderr_m = np.nanstd(m0, axis=0) / np.sqrt(m0.shape[0])

    ax.plot(range(7), mean_m, label="Behavior", color="red")
    ax.errorbar(x=range(7), y=mean_m, yerr=stderr_m, fmt='o', capsize=5, color="red")


    pros_retro_bias = ModelOptimizer(experiment=experiment, model_name="prospective").simulate_params(measure_name="retro_value_count")
    mean_pros = np.mean(np.nanmean(pros_retro_bias, axis=0), axis=0)
    std_pros = np.std(np.nanmean(pros_retro_bias, axis=0), axis=0) / np.sqrt(pros_retro_bias.shape[0])

    ax.plot(range(7), mean_pros, label="Prospective", color="blue")
    ax.errorbar(x=range(7), y=mean_pros, yerr=std_pros, fmt='o', capsize=5, color="blue")


    ax.legend(fontsize=11)
    ax.set_xlabel("progress: retrospective - prospective", fontsize=10)
    ylim = [0, 1]
    ax.set_ylim(ylim)


def plot_retrospective_bias(model_name=None):

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 5, width_ratios=[1, 0.8, 0.1, 0.7, 0.8])  # Set the height ratios for the subplots

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)
    #plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[0, 3], sharey=ax1)
    ax4 = plt.subplot(gs[0, 4], sharey=ax1)

    # Experiment 1

    ## Plot retro bias comparison between behavior and model

    ax1.annotate("A", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax2.annotate("B", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax3.annotate("C", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax4.annotate("D", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')

    plot_retro_bias_comparison_with_prospective_model(experiment=1, ax=ax1, model_name=model_name)

    ## Plot retro bias relation to difference in progress between the two options

    plot_retro_bias_relation_to_progress_difference(experiment=1, ax=ax2, model_name=model_name)

    plt.tight_layout()

    # Experiment 2

    ## Plot retro bias comparison between behavior and model

    plot_retro_bias_comparison_with_prospective_model(experiment=2, ax=ax3, model_name=model_name)

    ## Plot retro bias relation to difference in progress between the two options

    plot_retro_bias_relation_to_progress_difference(experiment=2, ax=ax4, model_name=model_name)

    plt.show()


def plot_retro_bias_compare_prospective(model_name=None):
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 5, width_ratios=[1, 0.8, 0.1, 0.7, 0.8])  # Set the height ratios for the subplots

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)
    # plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[0, 3], sharey=ax1)
    ax4 = plt.subplot(gs[0, 4], sharey=ax1)

    # Experiment 1

    ## Plot retro bias comparison between behavior and model

    ax1.annotate("A", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax2.annotate("B", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax3.annotate("C", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')
    ax4.annotate("D", xy=(0.5, -0.22), xycoords='axes fraction', fontsize=15, weight='bold')

    plot_retro_bias_comparison_with_prospective_model(experiment=1, ax=ax1, model_name=model_name, compare=True)

    ## Plot retro bias relation to difference in progress between the two options

    plot_retro_bias_progress_concat(experiment=1, ax=ax2, model_name=model_name)

    plt.tight_layout()

    # Experiment 2

    ## Plot retro bias comparison between behavior and model

    plot_retro_bias_comparison_with_prospective_model(experiment=2, ax=ax3, model_name=model_name, compare=True)

    ## Plot retro bias relation to difference in progress between the two options

    plot_retro_bias_progress_concat(experiment=2, ax=ax4, model_name=model_name)

    plt.show()



if __name__ == "__main__":

    #plot_retrospective_bias(model_name="hierarchy")
    plot_retrospective_bias(model_name="momentum")
    #plot_retrospective_bias(model_name="pers_hierarchy")
    #plot_retrospective_bias(model_name=None)
    #plot_retro_bias_compare_prospective(model_name=None)

    #plot_switches_complete()

    # sub_retro_bias_1 = get_measure_experiment(experiment=1, measure_name="retro_value", \
    #                                         mode="mean_condition")
    #
    # sub_retro_bias_2 = get_measure_experiment(experiment=2, measure_name="retro_value", \
    #                                           mode="mean_condition")
    #
    # sub_retro_bias