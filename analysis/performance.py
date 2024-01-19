from behavior import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pingouin as pg


def plot_performance_comparison_models():

    # Prospective
    pros_performance_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params()
    pros_performance_2 = ModelOptimizer(experiment=2, model_name="prospective").simulate_params()

    # Retrospective
    retro_performance_1 = ModelOptimizer(experiment=1, model_name="retrospective").simulate_params()
    retro_performance_2 = ModelOptimizer(experiment=2, model_name="retrospective").simulate_params()

    # Momentum
    mom_performance_1 = ModelOptimizer(experiment=1, model_name="momentum").simulate_params()
    mom_performance_2 = ModelOptimizer(experiment=2, model_name="momentum").simulate_params()

    # Persistence
    pers_performance_1 = ModelOptimizer(experiment=1, model_name="td_persistence").simulate_params()
    pers_performance_2 = ModelOptimizer(experiment=2, model_name="td_persistence").simulate_params()

    # plot bar plot comparing performance of models

    fig = plt.figure(figsize=(10, 5))

    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    data_1 = [pros_performance_1, retro_performance_1, mom_performance_1, pers_performance_1]

    conditions = ["80-20", "70-30", "60-40"]
    mean_values = [np.mean(pros_performance_1, axis=0), np.mean(retro_performance_1, axis=0), np.mean(mom_performance_1, axis=0), np.mean(pers_performance_1, axis=0)]
    std_values = [np.std(pros_performance_1, axis=0), np.std(retro_performance_1, axis=0), np.std(mom_performance_1, axis=0),   np.std(pers_performance_1, axis=0)]

    models = ["Prospective", "Retrospective", "TD-Momentum", "TD-Persistence"]
    title = "Experiment 1"
    ylabel = "Suits collected"
    ylim = [0, 24]

    plot_comparative_bar_plot(ax1, data_1, mean_values, std_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.19)


    data_2 = [pros_performance_2, retro_performance_2, mom_performance_2, pers_performance_2]

    conditions = ["75-25", "55-45"]
    mean_values = [np.mean(pros_performance_2, axis=0), np.mean(retro_performance_2, axis=0), np.mean(mom_performance_2, axis=0), np.mean(pers_performance_2, axis=0)]
    std_values = [np.std(pros_performance_2, axis=0), np.std(retro_performance_2, axis=0), np.std(mom_performance_2, axis=0), np.std(pers_performance_2, axis=0)]

    models = ["Prospective", "Retrospective", "TD-Momentum", "TD-Persistence"]
    title = "Experiment 2"
    ylabel = "Suits collected"
    ylim = [0, 24]

    plot_comparative_bar_plot(ax2, data_2, mean_values, std_values, conditions, models, title, ylabel, ylim,
                                bar_width=0.19)

    plt.tight_layout()
    plt.show()


def plot_performance_experiments():

    # Behavior
    sub_performance_1 = get_measure_experiment(experiment=1, measure_name="num_goals", mode="condition")
    sub_performance_2 = get_measure_experiment(experiment=2, measure_name="num_goals", mode="condition")

    # Prospective
    pros_performance_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params()
    pros_performance_2 = ModelOptimizer(experiment=2, model_name="prospective").simulate_params()

    # Retrospective
    retro_performance_1 = ModelOptimizer(experiment=1, model_name="retrospective").simulate_params()
    retro_performance_2 = ModelOptimizer(experiment=2, model_name="retrospective").simulate_params()


    #Compare overall subject performance with prospective model
    print("Experiment 1")
    tt = pg.ttest(np.mean(sub_performance_1, axis=1), np.mean(pros_performance_1, axis=1))
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    print("Experiment 2")
    tt = pg.ttest(np.mean(sub_performance_2, axis=1), np.mean(pros_performance_2, axis=1))
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    # Compare overall subject performance with retrospective model
    print("Experiment 1")
    tt = pg.ttest(np.mean(sub_performance_1, axis=1), np.mean(retro_performance_1, axis=1))
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    print("Experiment 2")
    tt = pg.ttest(np.mean(sub_performance_2, axis=1), np.mean(retro_performance_2, axis=1))
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    # print interactions for prospective and subject performance

    print("Experiment 1")
    print("Prospective")
    get_anova_conditions(1, pros_performance_1)
    print("Subject")
    get_anova_conditions(1, sub_performance_1)

    print("Experiment 2")
    print("Prospective")
    get_anova_conditions(2, pros_performance_2)
    print("Subject")
    get_anova_conditions(2, sub_performance_2)


    """
    Plot human performance against model performance: 
    """

    data_1 = [pros_performance_1, sub_performance_1, retro_performance_1]
    data_2 = [pros_performance_2, sub_performance_2, retro_performance_2]

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 5, width_ratios=[1.35, 1, 0.05, 1, 1])  # Set the height ratios for the subplots

    axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[3]), plt.subplot(gs[4])]
    conditions = ["80-20", "70-30", "60-40"]
    mean_values =[np.mean(pros_performance_1, axis=0), np.mean(sub_performance_1, axis=0), np.mean(retro_performance_1, axis=0)]
    std_dev_values = np.array([np.std(pros_performance_1, axis=0), np.std(sub_performance_1, axis=0), np.std(retro_performance_1, axis=0)]) / np.sqrt(len(sub_performance_1))
    models = ["Prospective", "Behavior", "Retrospective"]
    title = "Experiment 1"
    ylabel = "Suits collected"
    ylim = [0, 24]

    plot_comparative_bar_plot(axs[0], data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim, bar_width=0.19)

    conditions = ["75-25", "55-45"]
    mean_values = [np.mean(pros_performance_2, axis=0), np.mean(sub_performance_2, axis=0),
                   np.mean(retro_performance_2, axis=0)]
    std_dev_values =  np.array([np.std(pros_performance_2, axis=0), np.std(sub_performance_2, axis=0),
                      np.std(retro_performance_2, axis=0)]) / np.sqrt(len(sub_performance_2))


    title = "Experiment 2"
    ylim = [0, 24]
    plot_comparative_bar_plot(axs[1], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim, bar_width=0.2)


    """
    Plot the distribution of performance for each model
    """

    sub_performance_1 = get_measure_experiment(experiment=1, measure_name="performance", mode="measure")
    pros_perforance_1 = np.sum(pros_performance_1, axis=1)

    retro_perforance_1 = np.sum(retro_performance_1, axis=1)

    measures = [pros_perforance_1[:63], sub_performance_1, retro_perforance_1[:63]]

    xlabel = "Performance"
    ylabel = "Density"
    labels = ["Prospective", "Behavior", "Retrospective"]

    # Add title boxes to each subplot
    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[2].annotate("Experiment 1", xy=(0.5, 1.05), xycoords='axes fraction',
                fontsize=11, ha='center', bbox=title_box_props)

    plot_multiple_histograms(axs[2], measures, labels, xlabel, ylabel, title=None)


    # Experiment 2
    sub_performance_2 = get_measure_experiment(experiment=2, measure_name="performance", mode="measure")
    pros_perforance_2 = np.sum(pros_performance_2, axis=1)
    retro_perforance_2 = np.sum(retro_performance_2, axis=1)

    measures = [ pros_perforance_2, sub_performance_2, retro_perforance_2]

    axs[3].annotate("Experiment 2", xy=(0.5, 1.05), xycoords='axes fraction',
                fontsize=11, ha='center', bbox=title_box_props)

    plot_multiple_histograms(axs[3], measures, labels, xlabel, ylabel, title=None)

    plt.tight_layout()
    plt.show()


def get_mean_and_variance_performance():
    sub_performance_1 = get_measure_experiment(experiment=1, measure_name="num_goals", mode="condition")
    sub_performance_2 = get_measure_experiment(experiment=2, measure_name="num_goals", mode="condition")

    performance_1 = np.sum(sub_performance_1, axis=1)
    performance_2 = np.sum(sub_performance_2, axis=1)

    print("Experiment 1")
    print("Mean", np.mean(performance_1))
    print("Std", np.std(performance_1))
    print("Max", np.max(performance_1))
    print("Min", np.min(performance_1))

    print("Experiment 2")
    print("Mean", np.mean(performance_2))
    print("Std", np.std(performance_2))
    print("Max", np.max(performance_2))
    print("Min", np.min(performance_2))


if __name__ == "__main__":
    #plot_performance_experiments()
    plot_performance_comparison_models()
    #get_mean_and_variance_performance()
    #plot_performance_instructions()