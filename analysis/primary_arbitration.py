from models import *
from reports import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def get_goal_valuations_simulation(experiment, model_name, measure_name, optimal=False):
    sub_measures = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        if not optimal:
            with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
            params = model_fits['params']
            params = list(params.values())
        else:
            file_name = "results_latent/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
            with open(file_name, "rb") as f:
                params = pickle.load(f)
                params = list(params[0].values())

        runmodel = RunModel(experiment, model_name, params)
        model_res = runmodel.get_model_res()

        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)

        measure = subject_measures.get_mean_measure_condition("retro_value")
        #subject_measures = ModelMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)

        sub_measures.append(measure)
    return np.array(sub_measures)

def get_goal_valuations(experiment, measure):
    """
    Get choice proportions of the following for each experiment
    1. Prospective, retrospective, and other goals when prospective and retrospective goals diverge (measure = "prospective_retrospective_diverge")
    2. Prospective, next best, and other goals when prospective and retrospective goals converge (measure = "prospective_retrospective_converge")
    3. Dominant, max progress, and other goals when dominant and max progress goals diverge (measure = "maxprogress_dominant_diverge")
    4. Dominant, max progress, and other goals when dominant and max progress goals converge (measure = "maxprogress_dominant_converge")
    """
    subject_names = get_experiment_subjects(experiment)
    valuations = []
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[int(subject_num)]
        if measure == "maxprogress_dominant_diverge" or measure == "maxprogress_dominant_converge":
            sub_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        elif measure == "prospective_retrospective_diverge" or measure == "prospective_retrospective_converge":
            sub_measures = ModelMeasure(subject_id=subject_id, experiment=experiment)
        val = sub_measures.get_goal_valuation(measure=measure)
        valuations.append(val)

    valuations = np.array(valuations)
    return valuations

class GoalValuation:
    """
    Class for plotting goal valuations
    """
    def __init__(self, experiment, measure):
       self.experiment = experiment
       self.measure = measure

    def get_conditions_labels(self):
        if self.experiment == 1 or self.experiment == "instr_1":
            conditions = ["80-20", "70-30", "60-40"]
        elif self.experiment == 2:
            conditions = ["75-25", "55-45"]
        elif self.experiment == 4:
            conditions = ["H disp", "L disp"]
        return conditions

    def get_model_labels(self):
        if self.measure == "maxprogress_dominant_diverge":
            models = ["dominant suit", "max progress suit", "third suit"]
        elif self.measure == "maxprogress_dominant_converge":
            models = ["dominant/ max progress suit", "next best suit", "third suit"]
        elif self.measure == "prospective_retrospective_diverge":
            models = ["prospective suit", "retrospective suit", "third suit"]
        elif self.measure == "prospective_retrospective_converge":
            models = ["prospective/ retrospective suit", "next best suit", "third suit"]
        return models

    def get_title(self):
        if self.experiment == 1 or self.experiment == "instr_1":
            title = "Experiment 1"
        elif self.experiment == 2:
            title = "Experiment 2"
        elif self.experiment == 4:
            title = ""
        else:
            title = None

        return title

    def get_legend(self):
        if self.experiment == 1 or self.experiment == 4:
            legend = True
        else:
            legend = False
        return legend

    def get_ylabel(self):
        #if self.measure == "maxprogress_dominant_diverge" or self.measure == "prospective_retrospective_diverge":
        if self.experiment == 1 or self.experiment == 4:
            ylabel = "Choice probability"
        else:
            ylabel = None
        # else:
        #     ylabel = None
        return ylabel

    def get_colors(self):
        if self.measure == "maxprogress_dominant_diverge":
            colors = ['lightgreen', 'lightcoral', 'lightblue']
        elif self.measure == "maxprogress_dominant_converge":
            colors = ["lightgreen", "sandybrown", "lightblue"]
        elif self.measure == "prospective_retrospective_diverge":
            colors = ["palegreen","lightsalmon" , "lightskyblue"]
        elif self.measure == "prospective_retrospective_converge":
            colors = ["palegreen", "orange" , "lightskyblue"]

        return colors

    def plot_goal_valuation_experiment(self, ax, cache=False, show=True, sim=False):
        """
        Plot for one experiment.
        Use cache to save and load data.
        """
        if not ax:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.axhline(y=0.33, color='k', linestyle='--')

        if not cache:
            if not sim:
                valuations = get_goal_valuations(self.experiment, self.measure)
                np.save("figures_cache/goal_valuations_" + str(self.measure) + "_exp_" + str(self.experiment) + ".npy", valuations)
            else:
                valuations = get_goal_valuations_simulation(self.experiment, "momentum", self.measure)
                np.save("figures_cache/goal_valuations_" + str(self.measure) + "_exp_" + str(self.experiment) + "_sim.npy", valuations)
        else:
            if not sim:
                valuations = np.load("figures_cache/goal_valuations_" + str(self.measure) + "_exp_" + str(self.experiment) + ".npy")
            else:
                valuations = np.load("figures_cache/goal_valuations_" + str(self.measure) + "_exp_" + str(self.experiment) + "_sim.npy")

        prospective_mean = np.mean(valuations[:,0], axis=0)
        prospective_std = np.std(valuations[:,0], axis=0) / np.sqrt(len(valuations))

        retrospective_mean = np.mean(valuations[:,1], axis=0)
        retrospective_std = np.std(valuations[:,1], axis=0) / np.sqrt(len(valuations))

        print("Anova for prospective goal valuations")
        get_anova_conditions(self.experiment, valuations[:, 0])

        print("Anova for retrospective goal valuations")
        get_anova_conditions(self.experiment, valuations[:, 1])

        other_mean = np.mean(valuations[:,2], axis=0)
        other_std = np.std(valuations[:,2], axis=0) / np.sqrt(len(valuations))

        data_1 = [valuations[:,0], valuations[:,1], valuations[:,2]]

        conditions = self.get_conditions_labels()

        mean_values = [prospective_mean, retrospective_mean, other_mean]
        std_dev_values = np.array([prospective_std, retrospective_std, other_std])

        models = self.get_model_labels()

        title = self.get_title()

        ylim = [0, 1]

        legend = self.get_legend()
        ylabel = self.get_ylabel()
        colors = self.get_colors()

        print(self.measure, colors)

        plot_comparative_bar_plot(ax, data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.19, draw_data=False, legend=legend, colors=colors)

        # Plot a dashed horizontal line at y=0.33
        #
        if show:
            plt.tight_layout()
            plt.savefig("figures/goal_valuation_exp_" + str(self.experiment) + "_" + str(self.measure) + ".png")
            plt.show()


def plot_goal_valuations_experiment(experiment, measure, axs=None, show=False, cache=False, sim=False):
    """
    Plot goal valuations for experiments 1 and 2
    """
    if not axs:
        if experiment == 1:
            fig = plt.figure(figsize=(10, 5))
        else:
            fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(1, 2, width_ratios=[1, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]



    if measure.endswith("diverge"):
        GoalValuation(experiment=experiment, measure="maxprogress_dominant_diverge").\
            plot_goal_valuation_experiment(axs[0], cache=cache, show=False, sim=sim)
        GoalValuation(experiment=experiment, measure="prospective_retrospective_diverge").\
            plot_goal_valuation_experiment(axs[1], cache=cache, show=False, sim=sim)
    elif measure.endswith("converge"):
        GoalValuation(experiment=experiment, measure="maxprogress_dominant_converge").plot_goal_valuation_experiment(
            axs[0], cache=cache, show=False, sim=sim)
        GoalValuation(experiment=experiment, measure="prospective_retrospective_converge").plot_goal_valuation_experiment(
        axs[1], cache=cache, show=False, sim=sim)
    #GoalValuation(experiment=2, measure="maxprogress_dominant_diverge").plot_goal_valuation_experiment(axs[2], cache=cache, show=False)
    #GoalValuation(experiment=2, measure="prospective_retrospective_diverge").plot_goal_valuation_experiment(axs[3], cache=cache, show=False)
    #GoalValuation(experiment=2, measure=measure).plot_goal_valuation_experiment(axs[1], cache=cache, show=False)

    plt.savefig("figures/goal_valuation_" + str(experiment) + "_" + str(measure) + "_" + str(sim) + ".png")

    if show:
        #plt.tight_layout()
        plt.show()


def plot_goal_valuations_experiments_12(measure, axs=None, show=False, cache=False, sim=False):
    """
    Plot goal valuations for experiments 1 and 2
    """
    if not axs:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.35, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]


    GoalValuation(experiment=1, measure=measure).\
        plot_goal_valuation_experiment(axs[0], cache=cache, show=False)
    GoalValuation(experiment=2, measure=measure).plot_goal_valuation_experiment(axs[1], cache=cache, show=False)

    plt.savefig("figures/goal_valuation_12" + "_" + str(measure) + ".png")

    if show:
        #plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    measure = "prospective_retrospective_diverge"
    #measure = "maxprogress_dominant_diverge"
    #measure = "prospective_retrospective_converge"
    #measure = "maxprogress_dominant_converge"
    #gv = GoalValuation(experiment=1, measure=measure).plot_goal_valuation_experiment(ax=None, cache=False, show=True)
    #plot_goal_valuations_experiment(4, measure, show=True, cache=False)
    plot_goal_valuations_experiments_12(measure, show=True, cache=False)
