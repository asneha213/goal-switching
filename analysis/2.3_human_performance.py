import numpy as np

from behavior import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pingouin as pg


def plot_performance_experiments():

    # Experiment 1

    """
    Get interaction between performance and condition for subjects
    """

    sub_performance_1 = get_measure_experiment(experiment=1, measure_name="num_goals", mode="condition")
    get_anova_conditions(1, sub_performance_1)

    get_anova_partial_conditions(1, sub_performance_1, 0)
    get_anova_partial_conditions(1, sub_performance_1, 1)

    sub_performance_2 = get_measure_experiment(experiment=2, measure_name="num_goals", mode="condition")
    get_anova_conditions(2, sub_performance_2)

    """
        Get interaction between performance and condition for prospective model
    """

    pros_performance_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params()
    pros_performance_2 = ModelOptimizer(experiment=2, model_name="prospective").simulate_params()

    get_anova_conditions(1, pros_performance_1)
    get_anova_conditions(2, pros_performance_2)

    """
        Get interaction between performance and condition for retrospective model
    """

    retro_performance_1 = ModelOptimizer(experiment=1, model_name="retrospective").simulate_params()
    retro_performance_2 = ModelOptimizer(experiment=2, model_name="retrospective").simulate_params()

    get_anova_conditions(1, retro_performance_1)
    get_anova_conditions(2, retro_performance_1)


    """
     Compare subject performance with prospective model across conditions
    """

    print("Prospective and Behavior")

    print("Experiment 1")

    t_test_80_20 = pg.ttest(sub_performance_1[:, 0], pros_performance_1[:, 0])
    print("80-20; ",  t_test_80_20['T'].to_numpy()[0], t_test_80_20['p-val'].to_numpy()[0])

    t_test_70_30 = pg.ttest(sub_performance_1[:, 1], pros_performance_1[:, 1])
    print("70-30; ",t_test_70_30['T'].to_numpy()[0], t_test_70_30['p-val'].to_numpy()[0])

    t_test_60_40 = pg.ttest(sub_performance_1[:, 2], pros_performance_1[:, 2])
    print("60-40; ",t_test_60_40['T'].to_numpy()[0], t_test_60_40['p-val'].to_numpy()[0])

    print("Experiment 2")

    t_test_75_25 = pg.ttest(sub_performance_2[:, 0], pros_performance_2[:, 0])
    print("75-25; ",t_test_75_25['T'].to_numpy()[0], t_test_75_25['p-val'].to_numpy()[0])

    t_test_55_45 = pg.ttest(sub_performance_2[:, 1], pros_performance_2[:, 1])
    print("55-45; ",t_test_55_45['T'].to_numpy()[0], t_test_55_45['p-val'].to_numpy()[0])

    """
        Compare subject performance with retrospective model across conditions
       """

    print("Prospective and Behavior")

    print("Experiment 1")

    t_test_80_20 = pg.ttest(sub_performance_1[:, 0], retro_performance_1[:, 0])
    print("80-20; ",t_test_80_20['T'].to_numpy()[0], t_test_80_20['p-val'].to_numpy()[0])

    t_test_70_30 = pg.ttest(sub_performance_1[:, 1], retro_performance_1[:, 1])
    print("70-30; ",t_test_70_30['T'].to_numpy()[0], t_test_70_30['p-val'].to_numpy()[0])

    t_test_60_40 = pg.ttest(sub_performance_1[:, 2], retro_performance_1[:, 2])
    print("60-40; ",t_test_60_40['T'].to_numpy()[0], t_test_60_40['p-val'].to_numpy()[0])

    print("Experiment 2")

    t_test_75_25 = pg.ttest(sub_performance_2[:, 0], retro_performance_2[:, 0])
    print("75-25; ",t_test_75_25['T'].to_numpy()[0], t_test_75_25['p-val'].to_numpy()[0])

    t_test_55_45 = pg.ttest(sub_performance_2[:, 1], retro_performance_2[:, 1])
    print("55-45; ",t_test_55_45['T'].to_numpy()[0], t_test_55_45['p-val'].to_numpy()[0])


    """
    Plot human performance against model performance
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
    models = ["Prospective", "Behavior", "Retrospective"]
    title = "Experiment 2"
    ylabel = "Suits collected"
    ylim = [0, 24]
    plot_comparative_bar_plot(axs[1], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim, bar_width=0.2)

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

    from scipy.stats import gaussian_kde
    from scipy.stats import wasserstein_distance

    kde_pros = gaussian_kde(measures[0])
    kde_sub = gaussian_kde(measures[1])
    kde_retro = gaussian_kde(measures[2])

    eval_points = np.linspace(min(measures[1]), max(measures[0]), 1000)

    emd_pros = wasserstein_distance(kde_pros(eval_points), kde_sub(eval_points))
    emd_retro = wasserstein_distance(kde_retro(eval_points), kde_sub(eval_points))

    print("EMD Prospective", emd_pros)
    print("EMD Retrospective", emd_retro)


    h_pros = np.histogram(measures[0], density=False)[0] / np.sum(np.histogram(measures[0], density=False)[0])
    h_sub = np.histogram(measures[1], density=False)[0] / np.sum(np.histogram(measures[1], density=False)[0])
    h_retro = np.histogram(measures[2], density=False)[0] / np.sum(np.histogram(measures[2], density=False)[0])

    # calculate kldivergence

    from scipy.stats import entropy
    epsilon = 1e-10
    kld_pros = entropy(h_pros + epsilon, h_sub + epsilon)
    kld_retro = entropy(h_retro + epsilon, h_sub + epsilon)

    print("KLD Prospective", kld_pros)
    print("KLD Retrospective", kld_retro)




    sub_performance_2 = get_measure_experiment(experiment=2, measure_name="performance", mode="measure")
    pros_perforance_2 = np.sum(pros_performance_2, axis=1)
    retro_perforance_2 = np.sum(retro_performance_2, axis=1)

    measures = [ pros_perforance_2, sub_performance_2, retro_perforance_2]

    xlabel = "Performance"
    ylabel = "Density"
    labels = ["Prospective", "Behavior", "Retrospective"]

    axs[3].annotate("Experiment 2", xy=(0.5, 1.05), xycoords='axes fraction',
                fontsize=11, ha='center', bbox=title_box_props)

    plot_multiple_histograms(axs[3], measures, labels, xlabel, ylabel, title=None)

    kde_pros = gaussian_kde(measures[0])
    kde_sub = gaussian_kde(measures[1])
    kde_retro = gaussian_kde(measures[2])

    eval_points = np.linspace(min(measures[1]), max(measures[0]), 1000)

    emd_pros = wasserstein_distance(kde_pros(eval_points), kde_sub(eval_points))
    emd_retro = wasserstein_distance(kde_retro(eval_points), kde_sub(eval_points))

    print("EMD Prospective", emd_pros)
    print("EMD Retrospective", emd_retro)

    h_pros = np.histogram(measures[0], density=False)[0] / np.sum(np.histogram(measures[0], density=False)[0])
    h_sub = np.histogram(measures[1], density=False)[0] / np.sum(np.histogram(measures[1], density=False)[0])
    h_retro = np.histogram(measures[2], density=False)[0] / np.sum(np.histogram(measures[2], density=False)[0])

    epsilon = 1e-10
    kld_pros = entropy(h_pros + epsilon, h_sub + epsilon)
    kld_retro = entropy(h_retro + epsilon, h_sub + epsilon)

    print("KLD Prospective", kld_pros)
    print("KLD Retrospective", kld_retro)

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
    plot_performance_experiments()
    #get_mean_and_variance_performance()