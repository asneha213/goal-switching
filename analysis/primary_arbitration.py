from behavior import *
from reports import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def get_pros_retro_divergence(experiment):
    """
    Get the divergence between prospective and retrospective goal valuations
    """

    subject_names = get_experiment_subjects(experiment)
    divergences = []
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[int(subject_num)]
        sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        val = sub_measures.get_proportion_divergence_pros_retro()
        divergences.append(val)

    divergences = np.array(divergences)
    print("Experiment {}: Pros vs. Retro divergence: {}".format(experiment, np.mean(divergences)))
    return divergences


def get_goal_valuations(experiment, measure="retro_value"):
    subject_names = get_experiment_subjects(experiment)
    valuations = []
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[int(subject_num)]
        sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        val = sub_measures.get_goal_valuation(measure=measure)
        valuations.append(val)

    valuations = np.array(valuations)
    return valuations

def plot_goal_valuation_experiment(ax, experiment, cache=False, measure="retro_value", show=True):
    if not ax:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    if not cache:
        valuations = get_goal_valuations(experiment, measure)
        np.save("figures_cache/goal_valuations_" + str(measure) + "_exp_" + str(experiment) + ".npy", valuations)
    else:
        valuations = np.load("figures_cache/goal_valuations_" + str(measure) + "_exp_" + str(experiment) + ".npy")

    prospective_mean = np.mean(valuations[:,0], axis=0)
    prospective_std = np.std(valuations[:,0], axis=0) / np.sqrt(len(valuations))

    retrospective_mean = np.mean(valuations[:,1], axis=0)
    retrospective_std = np.std(valuations[:,1], axis=0) / np.sqrt(len(valuations))

    print("Anova for prospective goal valuations")
    get_anova_conditions(experiment, valuations[:, 0])

    print("Anova for prospective goal valuations")
    get_anova_conditions(experiment, valuations[:, 1])

    other_mean = np.mean(valuations[:,2], axis=0)
    other_std = np.std(valuations[:,2], axis=0) / np.sqrt(len(valuations))

    data_1 = [valuations[:,0], valuations[:,1], valuations[:,2]]

    if experiment == 1 or experiment == "instr_1":
        conditions = ["80-20", "70-30", "60-40"]
    elif experiment == 2:
        conditions = ["75-25", "55-45"]
    elif experiment == 4:
        conditions = ["H disp", "L disp"]
    mean_values = [prospective_mean, retrospective_mean, other_mean]
    std_dev_values = np.array([prospective_std, retrospective_std, other_std])

    if measure == "retro_value":
        models = ["prospective token", "retrospective token", "third token"]
    else:
        models = ["prospective token", "next best token", "third token"]
    if experiment == 4:
        experiment = 3
    title = "Experiment " + str(experiment)

    ylim = [0, 1]

    if experiment == 1:
        legend = True
        ylabel = "Choice probability"
    else:
        legend = False
        ylabel = None

    plot_comparative_bar_plot(ax, data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.19, draw_data=False, legend=legend)

    # Plot a dashed horizontal line at y=0.33
    ax.axhline(y=0.33, color='k', linestyle='--')

    if show:
        plt.tight_layout()
        plt.show()


def plot_goal_valuation(axs=None, show=False, cache=False, measure="retro_value"):
    """
    Plot choice proportions of prospective, retrospective, and other goals for each experiment
    """
    if not axs:
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 3, width_ratios=[1.35, 1, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])]

    plot_goal_valuation_experiment(axs[0], experiment=1, cache=cache, measure=measure, show=False)
    plot_goal_valuation_experiment(axs[1], experiment=2, cache=cache, measure=measure, show=False)
    plot_goal_valuation_experiment(axs[2], experiment=4, cache=cache, measure=measure, show=False)




if __name__ == "__main__":

    #plot_goal_valuation(show=True, measure="retro_value")
    measure = "obvious_choice"
    plot_goal_valuation_experiment(ax=None, experiment=4, cache=True,
                                   measure=measure, show=True)
    # plot_goal_valuation_experiment(ax=None, experiment=4, cache=False,
    #                                measure="retro_value", show=True)