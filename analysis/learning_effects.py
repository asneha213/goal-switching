from models import *
from reports import *

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pickle


def get_retro_measure_simulation(experiment, model_name, measure_name, optimal=True):
    sub_measures = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        if not optimal:
            with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
            params = model_fits['params']
        else:
            file_name = "results_latent/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
            with open(file_name, "rb") as f:
                params = pickle.load(f)
                #params = list(params[0].values())

        runmodel = RunModel(experiment, model_name, params)
        model_res = runmodel.get_model_res()

        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
        measure = subject_measures.get_mean_measure_condition(measure_name)

        sub_measures.append(measure)
    return np.array(sub_measures)


class LearningEffectsPlotter:
    def __init__(self, legend=True, ylabel=True, measure="retro_value", model_name=None):
        self.legend = legend
        self.ylabel = ylabel
        self.measure = measure
        self.mode = "mean_condition"
        self.model_name = model_name


    def get_conditions(self, experiment):
        if experiment == 1 or experiment == "instr_1":
            conditions = ["80-20", "70-30", "60-40"]
        elif experiment == 2:
            conditions = ["75-25", "55-45"]
        elif experiment == 4:
            conditions = ["H disp", "L disp"]
        return conditions

    def get_ylabel(self, experiment):
        if experiment == 1:
            if self.model_name is None:
                ylabel = "Choosing dominant suit"
            else:
                ylabel = None
        else:
            ylabel = None
        return ylabel

    def get_title(self, experiment):
        if experiment == 1:
            title = "Experiment 1"
        elif experiment == "instr_1":
            title = "Experiment 3 (Instructions)"
        elif experiment == 2:
            title = "Experiment 2"
        elif experiment == 4:
            title = "Experiment 4"

        if self.model_name is None:
            title += ": participants"
        elif self.model_name == "prospective":
            title = "prospective (task-optimized)"
        elif self.model_name == "momentum":
            title += ": TD-momentum"
        return title

    def get_legend(self, experiment):
        if experiment == 1:
            legend = True
        else:
            legend = False
        return legend


    def plot_learning_effects_per_condition(self, experiment, ax, cache=False, optimal=False):

        model_name = self.model_name
        if cache:
            if model_name is None:
                sub_retro_bias_first = np.load("figures_cache/" + self.measure + "_" + str(experiment) + "_first_half" + ".npy")
                sub_retro_bias_second = np.load("figures_cache/" + self.measure + "_" + str(experiment) + "_second_half" + ".npy")
            else:
                sub_retro_bias_first = np.load("figures_cache/" + self.measure + "_" + str(experiment) + "_first_half" + "_" + model_name + "_" + str(optimal) + ".npy")
                sub_retro_bias_second = np.load("figures_cache/" + self.measure + "_" + str(experiment) + "_second_half" + "_" + model_name + "_" + str(optimal) + ".npy")
        else:
            if model_name is None:
                sub_retro_bias_first = get_measure_experiment(experiment=experiment, measure_name=self.measure + "_first", \
                                                              mode=self.mode)
                sub_retro_bias_second = get_measure_experiment(experiment=experiment, measure_name=self.measure + "_second", \
                                                               mode=self.mode)
                np.save("figures_cache/" + self.measure + "_" + str(experiment) + "_first_half" + ".npy", sub_retro_bias_first)
                np.save("figures_cache/" + self.measure + "_" + str(experiment) + "_second_half" + ".npy", sub_retro_bias_second)
            else:
                sub_retro_bias_first = get_retro_measure_simulation(experiment, model_name, self.measure + "_first", optimal)
                sub_retro_bias_second = get_retro_measure_simulation(experiment, model_name, self.measure + "_second", optimal)
                np.save("figures_cache/" + self.measure + "_" + str(experiment) + "_first_half" + "_" + model_name + "_" + str(optimal) + ".npy", sub_retro_bias_first)
                np.save("figures_cache/" + self.measure + "_" + str(experiment) + "_second_half" + "_" + model_name + "_" + str(optimal) + ".npy", sub_retro_bias_second)


        data = [sub_retro_bias_first, sub_retro_bias_second]
        mean_values = [np.mean(sub_retro_bias_first, axis=0), np.mean(sub_retro_bias_second, axis=0)]
        std_dev_values = np.array([np.std(sub_retro_bias_first, axis=0) / np.sqrt(len(sub_retro_bias_first)),
                                   np.std(sub_retro_bias_second, axis=0) / np.sqrt(len(sub_retro_bias_second))])

        conditions = self.get_conditions(experiment)
        models = ["first half", "second half"]
        ylabel = self.get_ylabel(experiment)

        title = self.get_title(experiment)
        ylim = [0, 1.0]
        legend = self.get_legend(experiment)

        if model_name is None:
            plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                      bar_width=0.18, legend=legend, title_font=14)
        else:
            plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                      bar_width=0.18, legend=legend, title_font=14, hatch=["//", "//"])



    def plot_block_wise_learning_effects(self, experiment, ax, cache=False, group=True):
        self.measure = "condition_action_blockwise"
        self.mode = "measure"
        if not cache:
            sub_opt_action = get_measure_experiment(experiment=experiment, measure_name=self.measure, mode=self.mode)
            np.save("figures_cache/" + self.measure + "_" + str(experiment) + ".npy", sub_opt_action)
        else:
            sub_opt_action = np.load("figures_cache/" + self.measure + "_" + str(experiment) + ".npy")

        num_blocks = len(sub_opt_action[0])
        mean_vals = []
        std_err_vals = []
        for i in range(int(num_blocks/3)):
            sub_measures = sub_opt_action[:, i*3:(i+1)*3]
            mean_vals.append(np.mean(sub_measures))
            std_err_vals.append(np.std(sub_measures) / np.sqrt(len(sub_measures)))

        mean_vals = np.array(mean_vals)
        std_err_vals = np.array(std_err_vals)

        mean_vals = np.mean(sub_opt_action, axis=0)
        std_err_vals = np.std(sub_opt_action, axis=0) / np.sqrt(len(sub_opt_action))

        # b0 = sub_opt_action[:, 0:3]
        # if len(mean_vals) == 6:
        #     b1 = sub_opt_action[:, 15:17]
        # elif len(mean_vals) == 4:
        #     b1 = sub_opt_action[:, 9:11]
        #
        # b0 = np.mean(b0, axis=1)
        # b1 = np.mean(b1, axis=1)
        #
        # ttest = pg.ttest(b0, b1, paired=True)
        # print(ttest['p-val'])

        if len(mean_vals) == 6:
            xvals = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17']
        elif len(mean_vals) == 4:
            xvals = ['0-2', '3-5', '6-8', '9-11']

        xvals = np.arange(len(mean_vals))

        ax.plot(xvals, mean_vals, marker='o', linestyle='dashed', markersize=7, color="red")
        ax.errorbar(xvals, mean_vals, yerr=std_err_vals, fmt='o', linestyle='dashed', capsize=5, color="red")
        ax.set_ylim([0, 1.0])
        if experiment == "instr_1":
            experiment = "3 (Instr)"
        ax.set_title("Experiment: " + str(experiment), fontsize=14, fontweight='bold')

        # set xtick fontisze
        ax.tick_params(axis='x', labelsize=14)
        # set ytick fontisze
        ax.tick_params(axis='y', labelsize=14)



def plot_learning_effects_all_experiments_blockwise():
    cache = True
    measure = "condition_action"
    model_name = "momentum"
    model_name = None

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1.35, 1, 1.35, 1])
    plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=model_name)

    ax0 = fig.add_subplot(gs[0])
    plotter.plot_block_wise_learning_effects(1, ax0, cache=cache)
    ax0.set_ylabel("Choosing dominant suit", fontsize=14, fontweight='bold')
    ax0.set_xlabel("Blocks (first-to-last)", fontsize=14, fontweight='bold')

    plotter.plot_block_wise_learning_effects(2, fig.add_subplot(gs[1]), cache=cache)
    plotter.plot_block_wise_learning_effects("instr_1", fig.add_subplot(gs[2]), cache=cache)
    plotter.plot_block_wise_learning_effects(4, fig.add_subplot(gs[3]), cache=cache)


    plt.tight_layout()
    plt.savefig("figures/learning_effects_all_experiments_block_wise_no_group_" + str(model_name) + ".png")
    plt.show()

def plot_learning_effects_all_experiments(model_name, cache=True):
    measure = "condition_action"

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, width_ratios=[1.35, 1, 1])
    plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=model_name)

    ax0 = fig.add_subplot(gs[0])
    plotter.plot_learning_effects_per_condition(1, ax0, cache=cache)
    ax0.set_ylabel("Choosing dominant suit")

    plotter.plot_learning_effects_per_condition(2, fig.add_subplot(gs[1]), cache=cache)
    plotter.plot_learning_effects_per_condition(4, fig.add_subplot(gs[2]), cache=cache)

    plt.tight_layout()
    plt.savefig("figures/learning_effects_all_experiments_" + str(model_name) + ".png")
    plt.show()


def plot_learning_effects_experiments_instructions(cache=True, optimal=False):
    """
    Plot proportion of choosing dominant suit in first half and second half of the blocks in experiment 1 and experiment 1 (with instructions)
    Demonstrates learning effects within blocks under the influence of instructions
    """
    measure = "retro_value"
    measure = "condition_action"

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])


    plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=None)
    plotter.plot_learning_effects_per_condition(1, ax0, cache=cache, optimal=optimal)
    plotter.plot_learning_effects_per_condition("instr_1", ax1, cache=cache, optimal=optimal)
    plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name="prospective")
    plotter.plot_learning_effects_per_condition(1, ax2, cache=cache, optimal=True)
    ax2.legend().remove()
    ax0.set_ylabel("Choosing dominant suit")

    ax0.set_ylim([0, 1.1])
    ax1.set_ylim([0, 1.1])
    ax2.set_ylim([0, 1.1])

    plt.savefig("figures/learning_effects_instructions.png")
    plt.show()



def plot_learning_effects_experiments_12(model_name, cache=True, optimal=False):
    """
    Plot proportion of choosing dominant suit in first half and second half of the blocks in experiment 1 and 2
    Demonstrates learning effects within blocks
    """
    measure = "condition_action"

    if model_name is None:
        fig = plt.figure(figsize=(9, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])
    elif optimal == False:
        fig = plt.figure(figsize=(9, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])
    elif optimal == True:
        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 5, width_ratios=[1.35, 1.35, 0.1, 1.1, 1.2])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    if optimal:
        ax2 = fig.add_subplot(gs[3])
        ax3 = fig.add_subplot(gs[4])


    if model_name is None:
        plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=None)
        plotter.plot_learning_effects_per_condition(1, ax0, cache=cache, optimal=optimal)
        plotter.plot_learning_effects_per_condition(2, ax1, cache=cache, optimal=optimal)
    elif optimal:
        plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=None)
        plotter.plot_learning_effects_per_condition(1, ax0, cache=cache, optimal=False)
        plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=model_name)
        plotter.plot_learning_effects_per_condition(1, ax1, cache=cache, optimal=optimal)
        plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=None)
        plotter.plot_learning_effects_per_condition(2, ax2, cache=cache, optimal=False)
        plotter = LearningEffectsPlotter(legend=True, ylabel=True, measure=measure, model_name=model_name)
        plotter.plot_learning_effects_per_condition(2, ax3, cache=cache, optimal=optimal)


    ax0.set_ylabel("Choosing dominant suit")

    plt.savefig("figures/learning_effects_" + str(model_name) + "_" + str(optimal) + ".png")
    plt.show()



if __name__ == '__main__':

    cache = True


    # Figure 2
    #plot_learning_effects_experiments_12(model_name="prospective", cache=cache, optimal=True)
    # Figure 4
    #plot_learning_effects_experiments_instructions(cache=cache, optimal=False)
    # Supplementary Figure S9
    plot_learning_effects_all_experiments(model_name="momentum", cache=cache)
    # Supplementary Figure S3
    #plot_learning_effects_all_experiments_blockwise()


