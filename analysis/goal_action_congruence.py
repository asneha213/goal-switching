from behavior import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from reports import *


class PlotGoalActionCongruence:
    def __init__(self):
        pass

    def get_goal_action_congruence(self, experiment):
        measure_name = "goal_action_congruence"

        sub_measures = get_measure_experiment(experiment=experiment,
                                           measure_name=measure_name, \
                                           mode="measure")
        return sub_measures

    def get_switches_around_probes(self, experiment):
        measure_name = "switches_around_probes"

        sub_measures = get_measure_experiment(experiment=experiment,
                                           measure_name=measure_name, \
                                           mode="measure")
        return sub_measures

    def indicate_mean_value(self, ax, xpos, mean):
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        ax.text(mean, xpos, 'Mean: {:.2f}'.format(mean), color='r', fontsize=12, ha='center')

    def plot_goal_action_congruence_all_experiments(self, cache=False):


        fig = plt.figure(figsize=(13, 5))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # Set the height ratios for the subplots

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]), plt.subplot(gs[3])]
        if not cache:
            gac_1 = self.get_goal_action_congruence(1)
            gac_2 = self.get_goal_action_congruence(2)
            gac_3 = self.get_goal_action_congruence("instr_1")
            gac_4 = self.get_goal_action_congruence(4)
            np.save("figures_cache/goal_action_congruence_exp_1.npy", gac_1)
            np.save("figures_cache/goal_action_congruence_exp_2.npy", gac_2)
            np.save("figures_cache/goal_action_congruence_exp_4.npy", gac_4)
            np.save("figures_cache/goal_action_congruence_instr_1.npy", gac_3)
        else:
            gac_1 = np.load("figures_cache/goal_action_congruence_exp_1.npy")
            gac_2 = np.load("figures_cache/goal_action_congruence_exp_2.npy")
            gac_4 = np.load("figures_cache/goal_action_congruence_exp_4.npy")
            gac_3 = np.load("figures_cache/goal_action_congruence_instr_1.npy")

        # plot histogram for experiment 1
        plot_histogram(axs[0], gac_1, "Experiment 1", "Goal-action congruence", "Frequency")
        plot_histogram(axs[1], gac_2, "Experiment 2", None, None)
        plot_histogram(axs[3], gac_4, "Experiment 4", None, None)
        plot_histogram(axs[2], gac_3, "Experiment 3 (Instr)", None, None)

        self.indicate_mean_value(axs[0], 5.5, np.mean(gac_1))
        self.indicate_mean_value(axs[1], 11, np.mean(gac_2))
        self.indicate_mean_value(axs[3], 4.5, np.mean(gac_4))
        self.indicate_mean_value(axs[2], 5.5, np.mean(gac_3))

        plt.tight_layout()

        plt.savefig("figures/goal_action_congruence_all_experiments.png")
        plt.show()


    def plot_switches_around_probes_all_experiments(self, cache: object = False) -> object:


        fig = plt.figure(figsize=(13, 5))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]), plt.subplot(gs[3])]
        if not cache:
            sap_1 = self.get_switches_around_probes(1)
            sap_2 = self.get_switches_around_probes(2)
            sap_3 = self.get_switches_around_probes("instr_1")
            sap_4 = self.get_switches_around_probes(4)
            np.save("figures_cache/switches_around_probes_exp_1.npy", sap_1)
            np.save("figures_cache/switches_around_probes_exp_2.npy", sap_2)
            np.save("figures_cache/switches_around_probes_exp_4.npy", sap_4)
            np.save("figures_cache/switches_around_probes_instr_1.npy", sap_3)
        else:
            sap_1 = np.load("figures_cache/switches_around_probes_exp_1.npy")
            sap_2 = np.load("figures_cache/switches_around_probes_exp_2.npy")
            sap_4 = np.load("figures_cache/switches_around_probes_exp_4.npy")
            sap_3 = np.load("figures_cache/switches_around_probes_instr_1.npy")

        plot_histogram(axs[0], sap_1, "Experiment 1", "Switches around probes", "Frequency")
        plot_histogram(axs[1], sap_2, "Experiment 2", None, None)
        plot_histogram(axs[2], sap_3, "Experiment 3 (Instr)", None, None)
        plot_histogram(axs[3], sap_4, "Experiment 4", None, None)

        self.indicate_mean_value(axs[0], 5.5, np.mean(sap_1))
        self.indicate_mean_value(axs[1], 9, np.mean(sap_2))
        self.indicate_mean_value(axs[3], 8.5, np.mean(sap_4))
        self.indicate_mean_value(axs[2], 4.5, np.mean(sap_3))

        plt.tight_layout()

        plt.savefig("figures/switches_around_probes_all_experiments.png")

        plt.show()



if __name__ == "__main__":
    PlotGoalActionCongruence().plot_goal_action_congruence_all_experiments(cache=True)
    #PlotGoalActionCongruence().plot_switches_around_probes_all_experiments(cache=True)