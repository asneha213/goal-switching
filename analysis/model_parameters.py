import pickle

from behavior import *
from reports import *

import matplotlib.pyplot as plt


def get_model_param_measures(experiment, model_name, param_id, measure_name="performance"):
    param_vals = []
    measure_vals = []

    subject_names = get_experiment_subjects(experiment)

    for subject_num in range(len(subject_names)):

        subject_id = subject_names[subject_num]

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


def plot_model_parameters_behavior_correlation_experiment(experiment):
    fig = plt.figure(figsize=(15, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 6, width_ratios=[0.1, 1, 1, 1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]), \
           plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])]

    model_name = "momentum"

    alphas, performances = get_model_param_measures(experiment=experiment, \
                                                  model_name=model_name, \
                                                  param_id="alpha", measure_name="performance")

    print("LR- P", pg.corr(alphas, performances))
    if experiment == 1:
        pval = "2e-6"
    elif experiment == 2:
        pval = "0.002"
    elif experiment == "instr_1":
        pval = "0.01"

    # plot_correlation_param_measure(axs[0], alphas, performances, pval, \
    #                                r"Learning rate: $\alpha$", "Task performance", \
    #                                None)

    betas, switches = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_0", measure_name="switches_actions")

    print(pg.corr(betas, switches))
    if experiment == 1:
        pval = "1e-7"
    elif experiment == 2:
        pval = "5e-12"
    elif experiment == "instr_1":
        pval = "0.01"

    # plot_correlation_param_measure(axs[1], -betas, switches, pval, \
    #                                  r"Switch cost: $\beta_0$", "Action switches", \
    #                                   None)

    betas, retros = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_a", measure_name="retro_value")

    print(pg.corr(betas, retros))
    if experiment == 1:
        pval = "1e-6"
    elif experiment == 2:
        pval = "9e-10"
    elif experiment == "instr_1":
        pval = "0.01"

    # plot_correlation_param_measure(axs[2], betas, retros, pval, \
    #                                     r"Action switching temperature: $\beta_a$", "Retrospectively biased choice", \
    #                                     None)

    betas, switches = get_model_param_measures(experiment=experiment, \
                                               model_name=model_name, \
                                               param_id="alpha_c", measure_name="switches_probes")

    print(pg.corr(betas, switches))
    if experiment == 1:
        pval = "5e-5"
    elif experiment == 2:
        pval = "0.02"
    elif experiment == "instr_1":
        pval = "0.01"

    # plot_correlation_param_measure(axs[3], -betas, switches, pval, \
    #                                r"Switch cost: $\alpha_c$", "Goal switches", \
    #                                None)

    gammas, switches = get_model_param_measures(experiment=experiment, \
                                               model_name=model_name, \
                                               param_id="gamma", measure_name="performance")

    print(pg.corr(gammas, switches))
    if experiment == 1:
        pval = "5e-5"
    elif experiment == 2:
        pval = "0.02"
    elif experiment == "instr_1":
        pval = "0.01"

    # plot_correlation_param_measure(axs[4], gammas, switches, pval, \
    #                                r"Discount factor: $\gamma$", "Action switches", \
    #                                None)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    if experiment == 1:
        title = "Experiment 1"
    elif experiment == 2:
        title = "Experiment 2"
    elif experiment == "instr_1":
        title = "Experiment 3"

    # axs[0].annotate(title, xy=(-0.3, 0.5), rotation=90, xycoords='axes fraction',
    #                 fontsize=11, ha='center', va='center', bbox=title_box_props)
    #
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":

    plot_model_parameters_behavior_correlation_experiment(4)



