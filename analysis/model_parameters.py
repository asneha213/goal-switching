import pickle

from behavior import *
from reports import *

import matplotlib.pyplot as plt



def get_model_parameters(experiment, model_name, param_id):
    params = []
    subject_names = get_experiment_subjects(experiment)

    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        try:
            with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
        except:
            continue
        params.append(model_fits['params'][param_id])
    return np.array(params)



def get_model_param_measures(experiment, model_name, param_id, measure_name="performance"):
    param_vals = []
    measure_vals = []
    ages = []

    #subject_ages = get_demographic_info(experiment)

    subject_names = get_experiment_subjects(experiment)

    for subject_num in range(len(subject_names)):

        subject_id = subject_names[subject_num]

        try:
            with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)

        except:
            continue

        params = model_fits['params']
        measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        param_value = params[param_id]

        #ages.append(int(subject_ages[subject_id]))

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

    return np.array(param_vals), np.array(measure_vals), np.array(ages)




def plot_model_parameters_behavior_correlation_experiment(experiment):
    fig = plt.figure(figsize=(15, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 6, width_ratios=[0.1, 1, 1, 1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]), \
           plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])]

    model_name = "momentum"

    alphas, performances, ages = get_model_param_measures(experiment=experiment, \
                                                  model_name=model_name, \
                                                  param_id="alpha", measure_name="performance")

    print("Learning rates - performance", pg.corr(alphas, performances))

    # plot_correlation_param_measure(axs[0], alphas, performances, pval, \
    #                                r"Learning rate: $\alpha$", "Task performance", \
    #                                None)

    betas, action_switches, ages = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_0", measure_name="switches_actions")

    print("Switching cost --- number of action switches", pg.corr(betas, action_switches))
    # plot_correlation_param_measure(axs[1], -betas, switches, pval, \
    #                                  r"Switch cost: $\beta_0$", "Action switches", \
    #                                   None)

    betas, retro_values, ages = get_model_param_measures(experiment=experiment, \
                                                    model_name=model_name, \
                                                    param_id="beta_a", measure_name="retro_value")

    print("Action-softmax, max progress proportion", pg.corr(betas, retro_values))

    # plot_correlation_param_measure(axs[2], betas, retros, pval, \
    #                                     r"Action switching temperature: $\beta_a$", "Retrospectively biased choice", \
    #                                     None)

    betas, goal_switches, ages = get_model_param_measures(experiment=experiment, \
                                               model_name=model_name, \
                                               param_id="alpha_c", measure_name="switches_probes")

    print("Choice Kernel, number of goal switches", pg.corr(betas, goal_switches))

    # plot_correlation_param_measure(axs[3], -betas, switches, pval, \
    #                                r"Switch cost: $\alpha_c$", "Goal switches", \
    #                                None)

    gammas, performances, ages = get_model_param_measures(experiment=experiment, \
                                               model_name=model_name, \
                                               param_id="gamma", measure_name="performance")

    print("Gamma, performance", pg.corr(gammas, performances))
    # plot_correlation_param_measure(axs[4], gammas, switches, pval, \
    #                                r"Discount factor: $\gamma$", "Action switches", \
    #                                None)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)


    # axs[0].annotate(title, xy=(-0.3, 0.5), rotation=90, xycoords='axes fraction',
    #                 fontsize=11, ha='center', va='center', bbox=title_box_props)
    #
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":

    plot_model_parameters_behavior_correlation_experiment(4)

    # model_name = "prospective"
    # params = get_model_parameters(2, model_name, "gamma")
    # print(np.mean(params))



