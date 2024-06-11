from fit_behavior import *
from models import *
from behavior import *
from reports import *

import pingouin as pg

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec


def get_stay_switch_measure_simulation(experiment, subject_id, model_name):

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

    subject_names = get_experiment_subjects(experiment)
    no_reward_rates = []
    reward_rates = []

    for subject_num in range(len(subject_names)):

        subject_id = subject_names[subject_num]

        if behavior:
            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
            # get switch rates from max slot and max prob
            slot_srate, prob_srate = subject_measures.get_stay_switch_condition_counts()
        else:
            try:
                with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
            except:
                continue
            slot_srate, prob_srate = get_stay_switch_measure_simulation(experiment, subject_id, model_name)

        sswitch_0_r = slot_srate[0][1]
        sswitch_0_nr = slot_srate[0][0]
        sswitch_1_r = slot_srate[1][1]
        sswitch_1_nr = slot_srate[1][0]

        if (experiment == 1) or (experiment == "instr_1"):
            sswitch_2_r = slot_srate[2][1]
            sswitch_2_nr = slot_srate[2][0]

        pswitch_0_r = prob_srate[0][1]
        pswitch_0_nr = prob_srate[0][0]
        pswitch_1_r = prob_srate[1][1]
        pswitch_1_nr = prob_srate[1][0]

        if (experiment == 1) or (experiment == "instr_1"):
            pswitch_2_r = prob_srate[2][1]
            pswitch_2_nr = prob_srate[2][0]

        if (experiment == 1) or (experiment == "instr_1"):
            reward = [sswitch_0_r, sswitch_1_r, sswitch_2_r, pswitch_0_r, pswitch_1_r, pswitch_2_r]
            no_reward = [sswitch_0_nr, sswitch_1_nr, sswitch_2_nr, pswitch_0_nr, pswitch_1_nr, pswitch_2_nr]

        else:
            reward = [sswitch_0_r, sswitch_1_r, pswitch_0_r, pswitch_1_r]
            no_reward = [sswitch_0_nr, sswitch_1_nr, pswitch_0_nr, pswitch_1_nr]

        no_reward_rates.append(no_reward)
        reward_rates.append(reward)

    # concatenate max slot and max prob switch rates across conditions
    no_reward_rates = np.array(no_reward_rates)
    reward_rates = np.array(reward_rates)

    return no_reward_rates, reward_rates


def plot_stay_switch_rates(axs=None, model_name=None, experiments=[1, 2], show=False, cache=False):
    #
    if not axs:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    experiment = experiments[0]

    if model_name is None:
        if not cache:
            no_reward_rates, reward_rates = get_switches_from_max_prob_max_slot(experiment=experiment, behavior=True)
            np.save("figures_cache/no_reward_rates_exp" + str(experiment) + ".npy", no_reward_rates)
            np.save("figures_cache/reward_rates_exp" + str(experiment) + ".npy", reward_rates)
        else:
            no_reward_rates = np.load("figures_cache/no_reward_rates_exp" + str(experiment) + ".npy")
            reward_rates = np.load("figures_cache/reward_rates_exp" + str(experiment) + ".npy")
    else:
        if not cache:
            no_reward_rates, reward_rates = get_switches_from_max_prob_max_slot(experiment=experiment, behavior=False, model_name=model_name)
            np.save("figures_cache/no_reward_rates_exp_" + str(experiment) + model_name + ".npy", no_reward_rates)
            np.save("figures_cache/reward_rates_exp_" + str(experiment) + model_name + ".npy", reward_rates)
        else:
            no_reward_rates = np.load("figures_cache/no_reward_rates_exp_" + str(experiment) + model_name + ".npy")
            reward_rates = np.load("figures_cache/reward_rates_exp_" + str(experiment) + model_name + ".npy")


    conditions = ['80-20', '70-30', '60-40', '80-20', '70-30', '60-40']
    models = ['no reward', 'reward']
    title = "Experiment 1"
    ylabel = "Probability of Switching"
    ylim = [0, 0.53]

    data = [no_reward_rates, reward_rates]
    mean_values = [np.nanmean(no_reward_rates, axis=0), np.nanmean(reward_rates, axis=0)]
    std_dev_values = [np.nanstd(no_reward_rates, axis=0), np.nanstd(reward_rates, axis=0)] / np.sqrt(69)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("retrospective choice", xy=(0.25, -0.28), xycoords='axes fraction',
                    fontsize=9, ha='center')
    axs[0].annotate("prospective choice", xy=(0.75, -0.28), xycoords='axes fraction',
                    fontsize=9, ha='center')

    # axs[0].axvspan(-0.3, 2.9, facecolor='green', alpha=0.3)
    # axs[0].axvspan(2.9, 6.2, facecolor='orange', alpha=0.3)

    x_pos = [0, 1, 2, 3.5, 4.5, 5.5]

    p_val_80_20_nr = pg.ttest(no_reward_rates[:,0], no_reward_rates[:,3], paired=True)['p-val']
    p_val_70_30_nr = pg.ttest(no_reward_rates[:,1], no_reward_rates[:,4], paired=True)['p-val']
    p_val_60_40_nr = pg.ttest(no_reward_rates[:,2], no_reward_rates[:,5], paired=True)['p-val']

    p_val_80_20_r = pg.ttest(reward_rates[:,0], reward_rates[:,3], paired=True)['p-val']
    p_val_70_30_r = pg.ttest(reward_rates[:,1], reward_rates[:,4], paired=True)['p-val']
    p_val_60_40_r = pg.ttest(reward_rates[:,2], reward_rates[:,5], paired=True)['p-val']

    switches_from_retro = np.mean(reward_rates[:, 0:3], axis=1)
    switches_from_pros = np.mean(reward_rates[:, 3:], axis=1)

    tt = pg.ttest(switches_from_retro, switches_from_pros, paired=True)

   # print ttest with T, p-val, cohens d with identifier
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    mean_switches_reward = np.mean(np.nanmean(reward_rates, axis=0))
    mean_switches_no_reward = np.mean(np.nanmean(no_reward_rates, axis=0))

    mean_of_mean = np.mean([mean_switches_reward, mean_switches_no_reward])

    print("Mean switching rate: ", mean_of_mean)

    if show:
        p_values_nr = ['ns', '*', '***']
        p_values_r = ['ns', '*', '**']

        axs[0].plot([0, 3.5], [0.26, 0.26], color='black')
        significance_str = p_values_nr[0]
        axs[0].text(1.6, 0.26, f'{significance_str}', ha='center', va='bottom',
                fontsize=8)

        axs[0].plot([0.25, 3.75], [0.29, 0.29], color='black')
        significance_str = p_values_r[0]
        axs[0].text(1.65, 0.29, f'{significance_str}', ha='center', va='bottom',
                    fontsize=8)


        axs[0].plot([1, 4.5], [0.32, 0.32], color='black')
        significance_str = p_values_nr[1]
        axs[0].text(2.25, 0.32, f'{significance_str}', ha='center', va='bottom',
                    fontsize=8)
        axs[0].plot([1.25, 4.75], [0.35, 0.35], color='black')
        significance_str = p_values_r[1]
        axs[0].text(2.5, 0.35, f'{significance_str}', ha='center', va='bottom',
                    fontsize=8)


        axs[0].plot([2, 5.5], [0.40, 0.40], color='black')
        significance_str = p_values_nr[2]
        axs[0].text(3, 0.40, f'{significance_str}', ha='center', va='bottom',
                    fontsize=8)
        axs[0].plot([2.25, 5.75], [0.44, 0.44], color='black')
        significance_str = p_values_r[2]
        axs[0].text(3.2, 0.44, f'{significance_str}', ha='center', va='bottom',
                    fontsize=8)


    plot_comparative_bar_plot(axs[0], data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.3, x_pos=x_pos, draw_data=False)

    experiment = experiments[1]

    if model_name is None:
        no_reward_rates, reward_rates = get_switches_from_max_prob_max_slot(experiment=experiment, behavior=True)
    else:
        no_reward_rates, reward_rates = get_switches_from_max_prob_max_slot(experiment=experiment, behavior=False, model_name=model_name)


    data = [no_reward_rates, reward_rates]
    mean_values = [np.nanmean(no_reward_rates, axis=0), np.nanmean(reward_rates, axis=0)]
    std_dev_values = [np.nanstd(no_reward_rates, axis=0), np.nanstd(reward_rates, axis=0)] / np.sqrt(50)

    conditions = ['75-25', '55-45', '75-25', '55-45']
    models = ['No Reward', 'Reward']
    title = "Experiment 2"
    ylabel = None
    ylim = [0, 0.53]

    # axs[1].axvspan(-0.3, 1.9, facecolor='green', alpha=0.3)
    # axs[1].axvspan(1.9, 4.2, facecolor='orange', alpha=0.3)
    # Add title boxes to each subplot

    axs[1].annotate("retrospective choice", xy=(0.25, -0.28), xycoords='axes fraction',
                fontsize=9, ha='center')
    axs[1].annotate("prospective choice", xy=(0.75, -0.28), xycoords='axes fraction',
                    fontsize=9, ha='center')

    p_val_75_25_nr = pg.ttest(no_reward_rates[:, 0], no_reward_rates[:, 2], paired=True)['p-val']
    p_val_55_45_nr = pg.ttest(no_reward_rates[:, 1], no_reward_rates[:, 3], paired=True)['p-val']

    p_val_75_25_r = pg.ttest(reward_rates[:, 0], reward_rates[:, 2], paired=True)['p-val']
    p_val_55_45_r = pg.ttest(reward_rates[:, 1], reward_rates[:, 3], paired=True)['p-val']

    switches_from_retro = np.mean(reward_rates[:, 0:3], axis=1)
    switches_from_pros = np.mean(reward_rates[:, 3:], axis=1)

    tt = pg.ttest(switches_from_retro, switches_from_pros, paired=True)

    # print ttest with T, p-val, cohens d with identifier
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    mean_switches_reward = np.mean(np.nanmean(reward_rates, axis=0))
    mean_switches_no_reward = np.mean(np.nanmean(no_reward_rates, axis=0))

    mean_of_mean = np.mean([mean_switches_reward, mean_switches_no_reward])

    print("Mean switching rate: ", mean_of_mean)

    if show:
        p_values_nr = ['*', '***']
        p_values_r = ['*', '***']

        axs[1].plot([0, 2.5], [0.32, 0.32], color='black')
        significance_str = p_values_nr[0]
        axs[1].text(1.6, 0.31, f'{significance_str}', ha='center', va='bottom',
                    fontsize=11)

        axs[1].plot([0.25, 2.75], [0.35, 0.35], color='black')
        significance_str = p_values_r[0]
        axs[1].text(1.65, 0.34, f'{significance_str}', ha='center', va='bottom',
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
                              bar_width=0.3, x_pos=x_pos, draw_data=False, legend=False)


    # if show:
    #     plt.tight_layout()
    #     plt.show()


def compare_behavior_model(model_name, cache=False):
    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    #plot_stay_switch_rates([axs[0], axs[1]], model_name=None, show=True)
    plot_stay_switch_rates([axs[0], axs[1]], model_name=model_name, show=True, cache=cache)

    # Add labels "A" and "B" at the top of the subplots
    axs[0].annotate("BEHAVIOR", xy=(-0.2, 0.35), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    #axs[2].annotate("MODEL", xy=(-0.2, 0.55), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)

    #plt.tight_layout()
    plt.show()


def compare_behavior_model_all(seed=120, cache=False):
    np.random.seed(seed)
    fig = plt.figure(figsize=(17, 15))
    gs = GridSpec(7, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.1, 1, 0.1, 1, 0.1, 1])  # Set the height ratios for the subplots

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)

    ax3 = plt.subplot(gs[2, 0])
    ax4 = plt.subplot(gs[2, 1], sharey=ax3)

    ax5 = plt.subplot(gs[4, 0])
    ax6 = plt.subplot(gs[4, 1], sharey=ax5)

    ax7 = plt.subplot(gs[6, 0])
    ax8 = plt.subplot(gs[6, 1], sharey=ax7)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]


    plot_stay_switch_rates([axs[0], axs[1]], model_name=None, cache=cache)
    plot_stay_switch_rates([axs[2], axs[3]], model_name="momentum", cache=cache)
    plot_stay_switch_rates([axs[4], axs[5]], model_name="prospective", cache=cache)
    plot_stay_switch_rates([axs[6], axs[7]], model_name="td_persistence", cache=cache)

    # Add labels "A" and "B" at the top of the subplots
    axs[0].annotate("BEHAVIOR", xy=(-0.2, 0.25), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    axs[2].annotate("MOMENTUM", xy=(-0.2, 0.25), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    axs[4].annotate("PROSPECTIVE", xy=(-0.2, 0.2), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)
    axs[6].annotate("TD PERSISTENCE", xy=(-0.2, 0.15), xycoords='axes fraction', fontsize=15, weight='bold', rotation=90)

    plt.savefig("switching_patterns" + str(seed) + ".png")


if __name__ == "__main__":

    #compare_behavior_model_all(cache=False)
    model_name = None
    compare_behavior_model(model_name=model_name, cache=False)




