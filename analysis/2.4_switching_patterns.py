from fit_behavior import *
from models import *
from behavior import *
from reports import *

import pingouin as pg

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec


def get_proportion_of_divergence(experiment):
    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)
    prop_subs = []
    for subject_num in range(num_subjects):
        subject_id = subject_names[int(subject_num)]
        sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        prop = sub_measures.get_proportion_divergence()
        prop_subs.append(prop)

    return np.mean(prop_subs)


def get_measure_simulation(experiment, subject_id, model_name):

    with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
        model_fits = pickle.load(f)

    params = model_fits['params']
    params = list(params.values())
    runmodel = RunModel(experiment, model_name, params)
    model_res = runmodel.get_model_res()
    subject_measures = SubjectMeasure(subject_id=-1, experiment=experiment, model_res=model_res)
    slot_srate, prob_srate = subject_measures.get_stay_switch_condition_counts()
    return slot_srate, prob_srate


def get_switches_from_max_prob_max_slot(experiment, behavior=True, model_name=None):
    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)

    ss_rates = []
    ps_rates = []

    for subject_num in range(num_subjects):

        subject_id = subject_names[subject_num]

        if experiment == 1 and subject_num in [0, 6, 11, 21, 23, 27]:
            continue

        if behavior:
            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
            slot_srate, prob_srate = subject_measures.get_stay_switch_condition_counts()
        else:
            try:
                with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
            except:
                continue
            slot_srate, prob_srate = get_measure_simulation(experiment, subject_id, model_name)

        sswitch_0_r = slot_srate[0][1]
        sswitch_0_nr = slot_srate[0][0]
        sswitch_1_r = slot_srate[1][1]
        sswitch_1_nr = slot_srate[1][0]

        if experiment == 1:
            sswitch_2_r = slot_srate[2][1]
            sswitch_2_nr = slot_srate[2][0]

        pswitch_0_r = prob_srate[0][1]
        pswitch_0_nr = prob_srate[0][0]
        pswitch_1_r = prob_srate[1][1]
        pswitch_1_nr = prob_srate[1][0]

        if experiment == 1:
            pswitch_2_r = prob_srate[2][1]
            pswitch_2_nr = prob_srate[2][0]

        if experiment == 1:
            reward = [sswitch_0_r, sswitch_1_r, sswitch_2_r, pswitch_0_r, pswitch_1_r, pswitch_2_r]
            no_reward = [sswitch_0_nr, sswitch_1_nr, sswitch_2_nr, pswitch_0_nr, pswitch_1_nr, pswitch_2_nr]

        else:
            reward = [sswitch_0_r, sswitch_1_r, pswitch_0_r, pswitch_1_r]
            no_reward = [sswitch_0_nr, sswitch_1_nr, pswitch_0_nr, pswitch_1_nr]

        ss_rates.append(no_reward)
        ps_rates.append(reward)

    ss_rates = np.array(ss_rates)
    ps_rates = np.array(ps_rates)

    return ss_rates, ps_rates


def plot_stay_switch_rates(axs=None, model_name=None):
    #
    if not axs:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    if model_name is None:
        ss_rates, ps_rates = get_switches_from_max_prob_max_slot(experiment=1, behavior=True)
    else:
        ss_rates, ps_rates = get_switches_from_max_prob_max_slot(experiment=1, behavior=False, model_name=model_name)

    conditions = ['80-20', '70-30', '60-40', '80-20', '70-30', '60-40']
    models = ['No Reward', 'Reward']
    title = "Experiment 1"
    ylabel = "Probability of Switching"
    ylim = [0, 0.53]

    data = [ss_rates, ps_rates]
    mean_values = [np.nanmean(ss_rates, axis=0), np.nanmean(ps_rates, axis=0)]
    std_dev_values = [np.nanstd(ss_rates, axis=0), np.nanstd(ps_rates, axis=0)] / np.sqrt(69)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("RETROSPECTIVE ACTION", xy=(0.25, -0.12), xycoords='axes fraction',
                    fontsize=8, ha='center')
    axs[0].annotate("PROSPECTIVE ACTION", xy=(0.75, -0.12), xycoords='axes fraction',
                    fontsize=8, ha='center')

    axs[0].axvspan(-0.3, 2.9, facecolor='green', alpha=0.3)
    axs[0].axvspan(2.9, 6.2, facecolor='orange', alpha=0.3)

    x_pos = [0, 1, 2, 3.5, 4.5, 5.5]

    if model_name is None:
        p_values_nr = ['ns', '*', '***']
        p_values_r = ['ns', '*', '**']

        axs[0].plot([0, 3.5], [0.28, 0.28], color='black')
        significance_str = p_values_nr[0]
        axs[0].text(1.6, 0.28, f'{significance_str}', ha='center', va='bottom',
                fontsize=11)

        axs[0].plot([0.25, 3.75], [0.3, 0.3], color='black')
        significance_str = p_values_r[0]
        axs[0].text(1.65, 0.3, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)


        axs[0].plot([1, 4.5], [0.35, 0.35], color='black')
        significance_str = p_values_nr[1]
        axs[0].text(2.25, 0.35, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)
        axs[0].plot([1.25, 4.75], [0.37, 0.37], color='black')
        significance_str = p_values_r[1]
        axs[0].text(2.5, 0.37, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)


        axs[0].plot([2, 5.5], [0.40, 0.40], color='black')
        significance_str = p_values_nr[2]
        axs[0].text(3, 0.40, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)
        axs[0].plot([2.25, 5.75], [0.42, 0.42], color='black')
        significance_str = p_values_r[2]
        axs[0].text(3.2, 0.42, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)


    plot_comparative_bar_plot(axs[0], data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.3, x_pos=x_pos, draw_data=False)

    ss_rates, ps_rates = get_switches_from_max_prob_max_slot(experiment=2, behavior=True)
    data = [ss_rates, ps_rates]
    mean_values = [np.nanmean(ss_rates, axis=0), np.nanmean(ps_rates, axis=0)]
    std_dev_values = [np.nanstd(ss_rates, axis=0), np.nanstd(ps_rates, axis=0)] / np.sqrt(50)

    conditions = ['75-25', '55-45', '75-25', '55-45']
    models = ['No Reward', 'Reward']
    title = "Experiment 2"
    ylabel = "Probability of Switching"
    ylim = [0, 0.53]

    axs[1].axvspan(-0.3, 1.9, facecolor='green', alpha=0.3)
    axs[1].axvspan(1.9, 4.2, facecolor='orange', alpha=0.3)
    # Add title boxes to each subplot

    axs[1].annotate("RETROSPECTIVE ACTION", xy=(0.25, -0.12), xycoords='axes fraction',
                fontsize=8, ha='center')
    axs[1].annotate("PROSPECTIVE ACTION", xy=(0.75, -0.12), xycoords='axes fraction',
                    fontsize=8, ha='center')

    if model_name is None:
        p_values_nr = ['*', '****']
        p_values_r = ['*', '****']

        axs[1].plot([0, 2.5], [0.35, 0.35], color='black')
        significance_str = p_values_nr[0]
        axs[1].text(1.6, 0.35, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)

        axs[1].plot([0.25, 2.75], [0.37, 0.37], color='black')
        significance_str = p_values_r[0]
        axs[1].text(1.65, 0.37, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)

        axs[1].plot([1, 3.5], [0.4, 0.4], color='black')
        significance_str = p_values_nr[1]
        axs[1].text(2.25, 0.4, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)
        axs[1].plot([1.25, 3.75], [0.42, 0.42], color='black')
        significance_str = p_values_r[1]
        axs[1].text(2.5, 0.42, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)



    x_pos = [0, 1, 2.5, 3.5]
    plot_comparative_bar_plot(axs[1], data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.3, x_pos=x_pos, draw_data=False)


def get_goal_valuations(experiment):
    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)
    valuations = []
    for subject_num in range(num_subjects):
        subject_id = subject_names[int(subject_num)]
        sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        val = sub_measures.get_goal_valuation()
        valuations.append(val)

    valuations = np.array(valuations)
    return valuations


def plot_goal_valuation(axs=None):

    if not axs:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    experiment = 1

    valuations = get_goal_valuations(experiment)

    prospective_mean = np.mean(valuations[:,0], axis=0)
    prospective_std = np.std(valuations[:,0], axis=0) / np.sqrt(len(valuations))

    retrospective_mean = np.mean(valuations[:,1], axis=0)
    retrospective_std = np.std(valuations[:,1], axis=0) / np.sqrt(len(valuations))

    other_mean = np.mean(valuations[:,2], axis=0)
    other_std = np.std(valuations[:,2], axis=0) / np.sqrt(len(valuations))

    data_1 = [valuations[:,0], valuations[:,1], valuations[:,2]]

    conditions = ["80-20", "70-30", "60-40"]
    mean_values = [prospective_mean, retrospective_mean, other_mean]
    std_dev_values = np.array([prospective_std, retrospective_std, other_std])
    models = ["Prospective", "Retrospective", "Other"]
    title = "Experiment 1"
    ylabel = "Choice probability"
    ylim = [0, 1]

    plot_comparative_bar_plot(axs[0], data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.19, draw_data=False)

    experiment = 2

    valuations = get_goal_valuations(experiment)

    prospective_mean = np.mean(valuations[:, 0], axis=0)
    prospective_std = np.std(valuations[:, 0], axis=0) / np.sqrt(len(valuations))

    retrospective_mean = np.mean(valuations[:, 1], axis=0)
    retrospective_std = np.std(valuations[:, 1], axis=0) / np.sqrt(len(valuations))

    other_mean = np.mean(valuations[:, 2], axis=0)
    other_std = np.std(valuations[:, 2], axis=0) / np.sqrt(len(valuations))

    data_2 = [valuations[:, 0], valuations[:, 1], valuations[:, 2]]

    conditions = ["75-25", "55-45"]
    mean_values = [prospective_mean, retrospective_mean, other_mean]
    std_dev_values = np.array([prospective_std, retrospective_std, other_std])
    models = ["Prospective", "Retrospective", "Other"]
    title = "Experiment 2"
    ylabel = "Choice probability"
    ylim = [0, 1]

    plot_comparative_bar_plot(axs[1], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.2, draw_data=False)


def plot_switching_patterns():
    fig = plt.figure(figsize=(10, 15))
    gs = GridSpec(3, 2, width_ratios=[1.35, 1], height_ratios=[0.6, 0.1, 1])  # Set the height ratios for the subplots
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[4]), plt.subplot(gs[5])]
    plot_stay_switch_rates([axs[2], axs[3]])
    plot_goal_valuation([axs[0], axs[1]])

    # Add labels "A" and "B" at the top of the subplots
    axs[0].annotate("A", xy=(-0.2, 0.85), xycoords='axes fraction', fontsize=15, weight='bold')
    axs[2].annotate("B", xy=(-0.2, 0.85), xycoords='axes fraction', fontsize=15, weight='bold')
    #plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # prop = get_proportion_of_divergence(experiment=2)
    # print(prop)

    #plot_stay_switch_rates()
    #plot_stay_switch_rates(model_name="hierarchy")
    #plot_stay_switch_rates()
    #plot_goal_valuation()

    plot_switching_patterns()




