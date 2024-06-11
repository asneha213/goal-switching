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



def plot_goal_valuation(axs=None, show=False, cache=False):
    """
    Plot choice proportions of prospective, retrospective, and other goals for each experiment
    """

    def get_goal_valuations(experiment):
        subject_names = get_experiment_subjects(experiment)
        valuations = []
        for subject_num in range(len(subject_names)):
            subject_id = subject_names[int(subject_num)]
            sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
            val = sub_measures.get_goal_valuation()
            valuations.append(val)

        valuations = np.array(valuations)
        return valuations

    if not axs:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    experiment = 1

    if not cache:
        valuations = get_goal_valuations(experiment)
        np.save("figures_cache/goal_valuations_exp_" + str(experiment) + ".npy", valuations)
    else:
        valuations = np.load("figures_cache/goal_valuations_exp_" + str(experiment) + ".npy")

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

    conditions = ["80-20", "70-30", "60-40"]
    mean_values = [prospective_mean, retrospective_mean, other_mean]
    std_dev_values = np.array([prospective_std, retrospective_std, other_std])
    models = ["prospective token", "retrospective token", "third token"]
    title = "Experiment 1"
    ylabel = "Choice probability"
    ylim = [0, 1]

    plot_comparative_bar_plot(axs[0], data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.19, draw_data=False)

    experiment = 2

    if not cache:
        valuations = get_goal_valuations(experiment)
        np.save("figures_cache/goal_valuations_exp_" + str(experiment) + ".npy", valuations)
    else:
        valuations = np.load("figures_cache/goal_valuations_exp_" + str(experiment) + ".npy")

    prospective_mean = np.mean(valuations[:, 0], axis=0)
    prospective_std = np.std(valuations[:, 0], axis=0) / np.sqrt(len(valuations))

    retrospective_mean = np.mean(valuations[:, 1], axis=0)
    retrospective_std = np.std(valuations[:, 1], axis=0) / np.sqrt(len(valuations))

    print("Anova for prospective goal valuations")
    get_anova_conditions(experiment, valuations[:, 0])

    print("Anova for prospective goal valuations")
    get_anova_conditions(experiment, valuations[:, 1])

    other_mean = np.mean(valuations[:, 2], axis=0)
    other_std = np.std(valuations[:, 2], axis=0) / np.sqrt(len(valuations))

    data_2 = [valuations[:, 0], valuations[:, 1], valuations[:, 2]]

    conditions = ["75-25", "55-45"]
    mean_values = [prospective_mean, retrospective_mean, other_mean]
    std_dev_values = np.array([prospective_std, retrospective_std, other_std])
    models = ["prospective token", "retrospective token", "third token"]
    title = "Experiment 2"
    ylabel = None
    ylim = [0, 1]

    plot_comparative_bar_plot(axs[1], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                              bar_width=0.2, draw_data=False, legend=False)

    if show:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    plot_goal_valuation(show=True)