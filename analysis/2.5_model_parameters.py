import pickle

import pingouin as pg

from behavior import *
from reports import *

import matplotlib.pyplot as plt


def get_model_param_measures(experiment, model_name, param_id, measure_name="performance"):
    param_vals = []
    measure_vals = []

    if experiment == 1:
        num_subjects = 69
        SUBJECT_NAMES = SUBJECT_NAMES_1
    elif experiment == 2:
        num_subjects = 50
        SUBJECT_NAMES = SUBJECT_NAMES_2

    for subject_id in range(num_subjects):

        subject_id = SUBJECT_NAMES[int(subject_id)]

        try:
            with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)

        except:
            continue

        params = model_fits['params']
        measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        param_value = params[param_id]
        param_vals.append(param_value)

        if measure_name == "performance":
            perf = measures.get_task_performance()
            measure_vals.append(perf)
        elif measure_name == "switches_probes":
            switches = measures.get_sum_measure_condition('switches_probes')
            measure_vals.append(np.sum(switches))
        elif measure_name == "switches_actions":
            switches = measures.get_sum_measure_condition('switches_actions')
            measure_vals.append(np.sum(switches))
        elif measure_name == "retro_value":
            retro = measures.get_mean_measure_condition('retro_value')
            measure_vals.append(np.mean(retro))
        elif measure_name == "optimal_action":
            optimal = measures.get_mean_measure_condition('optimal_action')
            measure_vals.append(np.mean(optimal))

    return np.array(param_vals), np.array(measure_vals)


def plot_model_param_measure_correlations(axs, model_name, param_id, \
                                          title=True, measure_name="performance"):
    experiment = 1

    params1, measures1 = get_model_param_measures(experiment=experiment, \
                                                  model_name=model_name, \
                                                  param_id=param_id, measure_name=measure_name)

    experiment = 2
    params2, measures2 = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id=param_id, measure_name=measure_name)

    if param_id == "alpha":
        p_val1 = "6e-7"
        p_val2 = "0.001"
        ylabel = "Task performance"
        xlabel = r"Learning rate: $\alpha$"


    elif param_id == "beta_0g":
        ylabel = "Action switches"
        p_val1 = "2e-6"
        p_val2 = "6e-16"
        xlabel = r"Switch Cost"

    elif param_id == "beta_a":
        ylabel = "Task performance"
        p_val1 = "2e-7"
        p_val2 = "4e-11"
        xlabel = r"Goal switching temperature"

    if title:
        title = "Experiment 1"
    else:
        title = None

    plot_correlation_param_measure(axs[0], params1, measures1, p_val1, \
                                   xlabel, ylabel, \
                                   title)

    if title:
        title = "Experiment 2"
    else:
        title = None

    plot_correlation_param_measure(axs[1], params2, measures2, p_val2, \
                                   xlabel, ylabel, \
                                   title)


def plot_model_parameters_behavior_correlation():
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 4))

    gs = GridSpec(1, 5, width_ratios=[1, 1, 0.01, 1, 1])

    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 3]), plt.subplot(gs[0, 4])]

    gs
    # axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[2, 0]), \
    #        plt.subplot(gs[2, 1]), plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1])]
    #
    plot_model_param_measure_correlations([axs[0], axs[2]], "td_hierarchy", \
                                          "alpha",\
                                          measure_name="performance")

    plot_model_param_measure_correlations([axs[1], axs[3]], "td_hierarchy", \
                                          "beta_0g", \
                                          measure_name="switches_actions", title=False)
    # plot_model_param_measure_correlations([axs[4], axs[5]], "td_hierarchy", \
    #                                       "beta_a", \
    #                                       measure_name="retro_value", title=False)

    plt.tight_layout()
    plt.show()


def plot_model_parameters_behavior_correlation_experiment(experiment):
    fig = plt.figure(figsize=(12, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, width_ratios=[0.1, 1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3])]

    model_name = "momentum"

    alphas, performances = get_model_param_measures(experiment=experiment, \
                                                  model_name=model_name, \
                                                  param_id="alpha", measure_name="performance")

    if experiment == 1:
        pval = "2e-8"
    elif experiment == 2:
        pval = "0.0001"

    plot_correlation_param_measure(axs[0], alphas, performances, pval, \
                                   r"Learning rate: $\alpha$", "Task performance", \
                                   None)



    betas, switches = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_0g", measure_name="switches_actions")


    if experiment == 1:
        pval = "2e-9"
    elif experiment == 2:
        pval = "2e-19"

    plot_correlation_param_measure(axs[1], -betas, switches, pval, \
                                     r"Switch cost: $\beta_0$", "Action switches", \
                                      None)


    betas, retros = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_a", measure_name="retro_value")

    if experiment == 1:
        pval = "4e-7"
    elif experiment == 2:
        pval = "3e-11"

    plot_correlation_param_measure(axs[2], betas, retros, pval, \
                                        r"Action switching temperature: $\beta_a$", "Retrospectively biased choice", \
                                        None)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    if experiment == 1:
        title = "Experiment 1"
    elif experiment == 2:
        title = "Experiment 2"

    axs[0].annotate(title, xy=(-0.3, 0.5), rotation=90, xycoords='axes fraction',
                    fontsize=11, ha='center', va='center', bbox=title_box_props)

    plt.tight_layout()
    plt.show()







if __name__ == "__main__":
    #plot_model_parameters_behavior_correlation()

    plot_model_parameters_behavior_correlation_experiment(1)



