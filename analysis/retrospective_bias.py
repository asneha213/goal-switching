from models import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec

import pickle



class RetroBiasPlotter:
    """
    Class to plot the retrospective bias patterns 
    (as a relation to task-block condition and progress differences)
    shown by participants and models
    """
    def __init__(self, legend=True, ylabel=True, measure="retro_value", sims_per_subject=1):
        self.legend = legend
        self.ylabel = ylabel
        self.measure = measure
        self.sims_per_subject = sims_per_subject
        if self.measure == "retrospective_choice_model":
            self.mode = "measure"
        else:
            self.mode = "mean_condition"


    def get_condition_labels(self, experiment):
        """
        Get block conditions available for each experiment
        """
        if (experiment == 1) or (experiment == "instr_1"):
            conditions = ["80-20", "70-30", "60-40"]
        elif experiment == 2:
            conditions = ["75-25", "55-45"]
        elif experiment == 'bandit':
            conditions = ["75-25", "55-45"]
        elif experiment == 4:
            conditions = ["H disp", "L Disp"]

        return conditions

    def get_model_labels(self, model_name, compare=False):
        """
        Get model labels for plotting
        """
        if model_name is None:
            if compare:
                models = ["Prospective (task-optimized)", "Participants", "Retrospective (task-optimized)"]
            else:
                models = [ "Participants"]
        elif model_name == "momentum":
            models = ["TD-momentum"]
        elif model_name == "td_persistence":
            models = ["TD-persistence"]
        elif model_name == "prospective":
            models = ["Prospective"]
        elif model_name == "hybrid":
            models = ["Hybrid"]
        elif model_name == "rescorla":
            models = ["Rescorla"]

        return models


    def get_title(self, experiment):
        if experiment == 1 or experiment == "instr_1":
            title = "Experiment 1"
        elif experiment == 2:
            title = "Experiment 2"
        elif experiment == 4:
            title = None
        else:
            title = None
        return title


    def get_ylabel(self, experiment, model_name=None):

        if experiment == 1 or experiment == 4:
            if self.ylabel:
                if self.measure == "retro_value":
                    ylabel = "Choosing max progress"
                else:
                    ylabel = "Choosing dominant/ max progress"
            else:
                ylabel = None
        elif experiment == 2:
            ylabel = None

        else:
            ylabel = None


        return ylabel


    def get_legend(self, experiment, compare=False):
        if compare:
            if experiment == 1:
                legend = self.legend
            else:
                legend = False
        else:
            legend = False
        return legend


    def plot_retro_bias_per_condition(self, experiment, ax, model_name=None, compare=False, cache=False, annot=False, hatch=None):
        """
            Plots a bar plot of retrospective bias per block condition in each experiment
            experiment: experiment number
            model_name: name of model to plot (None for participant data)
            compare: whether to compare with optimal models
            cache: whether to use cached results
            annot: whether to annotate the plot
            hatch: hatch pattern for bars
        """

        if cache:
            if model_name is None:
                sub_retro_bias = np.load("results_cache/" + self.measure + "_" + str(experiment) + "_behavior" + ".npy")
            else:
                sub_retro_bias = np.load("results_cache/" + self.measure + "_"  + str(experiment) + "_" + model_name + "_sims_" + str(self.sims_per_subject) + ".npy")
        else:
            if model_name is None:
                sub_retro_bias = get_model_measure_experiment(experiment=experiment, measure_name=self.measure, \
                                                    mode=self.mode)
                print(np.mean(sub_retro_bias, axis=0))
                np.save("results_cache/" + self.measure + "_" + str(experiment) + "_behavior.npy", sub_retro_bias)
            else:
                sub_retro_bias = get_model_simulation_of_measure(experiment=experiment, model_name=model_name, measure_name=self.measure, sims_per_subject=self.sims_per_subject)
                sub_retro_bias = np.mean(sub_retro_bias, axis=0)
                print(np.mean(sub_retro_bias, axis=0))
                np.save("results_cache/" + self.measure + "_" + str(experiment) + "_" + model_name + "_sims_" + str(self.sims_per_subject) + ".npy", sub_retro_bias)

        if compare:
            if cache:
                pros_retro_bias = np.load("results_cache/" + self.measure + "_"  + str(experiment) + "_prospective_optimal" + ".npy")
                retro_retro_bias = np.load("results_cache/" + self.measure + "_"  + str(experiment) + "_retrospective_optimal" + ".npy")

            else:
                pros_retro_bias = ModelOptimizer(experiment=experiment, model_name="prospective").simulate_params(
                    measure_name=self.measure)
                retro_retro_bias = ModelOptimizer(experiment=experiment, model_name="retrospective").simulate_params(
                    measure_name=self.measure)
                np.save("results_cache/" + self.measure + "_" + str(experiment) + "_prospective_optimal", pros_retro_bias)
                np.save("results_cache/" + self.measure + "_" + str(experiment) + "_retrospective_optimal", retro_retro_bias)

            data = [pros_retro_bias, sub_retro_bias, retro_retro_bias]
        else:
            data = [sub_retro_bias]

        conditions = self.get_condition_labels(experiment)

        if compare:
            mean_values = [np.mean(pros_retro_bias, axis=0), np.mean(sub_retro_bias, axis=0), np.mean(retro_retro_bias, axis=0)]
            std_dev_values = [np.std(pros_retro_bias, axis=0), np.std(sub_retro_bias, axis=0), np.std(retro_retro_bias, axis=0)] / np.sqrt(len(pros_retro_bias))
        else:
            mean_values = [np.mean(sub_retro_bias, axis=0)]
            std_dev_values = np.array([np.std(sub_retro_bias, axis=0)]) / np.sqrt(len(sub_retro_bias))


        models = self.get_model_labels(model_name, compare)

        title = self.get_title(experiment)

        ylabel = self.get_ylabel(experiment, model_name)
        legend = self.get_legend(experiment, compare)

        ylim = [0, 1.1]

        colors = ['gray'] if not compare else None
        bar_width = 0.3 if not compare else 0.12
        legend_loc = "upper right" if self.measure != "retro_value" else "upper left"


        if hatch:
            hatch_val = ["//"] * len(conditions)
        else:
            hatch_val = None

        plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                  bar_width=bar_width, legend=legend, colors=colors, legend_loc=legend_loc, hatch=hatch_val)


    def plot_retro_bias_per_condition_instructions(self, ax, cache=False):
        """
            Plots a bar plot of retrospective bias per block condition comparing experiment 1
            with no instructions and experiment 3 with instructions
            ax: axis to plot on
            cache: whether to use cached results
        """

        if cache:
            pros_retro_bias_1 = np.load("results_cache/pros_retro_bias_instr_1.npy")
            sub_retro_bias_1 = np.load("results_cache/sub_retro_instr_1.npy")
            sub_retro_bias_instr = np.load("results_cache/sub_retro_instrns.npy")
        else:

            pros_retro_bias_1 = ModelOptimizer(experiment=1,
                                               model_name="prospective").simulate_params(
                measure_name="retro_value")

            sub_retro_bias_1 = get_measure_experiment(experiment=1,
                                                      measure_name=self.measure, \
                                                      mode="mean_condition")
            sub_retro_bias_instr = get_measure_experiment(experiment="instr_1",
                                                          measure_name=self.measure, \
                                                          mode="mean_condition")
            np.save("results_cache/pros_retro_bias_instr_1.npy", pros_retro_bias_1)
            np.save("results_cache/sub_retro_instr_1.npy", sub_retro_bias_1)
            np.save("results_cache/sub_retro_instrns.npy", sub_retro_bias_instr)

        data = [pros_retro_bias_1, sub_retro_bias_1, sub_retro_bias_instr]
        mean_values = [np.mean(pros_retro_bias_1, axis=0), np.mean(sub_retro_bias_1, axis=0), np.mean(sub_retro_bias_instr, axis=0)]
        std_dev_values = np.array([np.std(pros_retro_bias_1, axis=0), np.std(sub_retro_bias_1, axis=0), np.std(sub_retro_bias_instr, axis=0)]) / np.sqrt(len(sub_retro_bias_1))

        conditions = ["80-20", "70-30", "60-40"]
        models = ["Prospective", "No instructions", "Instructions"]
        ylabel = "Choosing max_progress"

        title = None
        ylim = [0, 1.1]
        legend = self.legend
        plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                    bar_width=0.18, legend=legend)


    def plot_retro_bias_relation_to_progress_difference(self, experiment, ax, \
                                                        model_name=None, cache=False):
        """
            Plots the preference towards progress as a function of progress difference between max and dominant progress
            experiment: experiment number
            model_name: name of model to plot (None for participant data)
            cache: whether to use cached results
        """
        if cache:
            if model_name is None:
                progress_preference_curve = np.load("results_cache/" + self.measure + "_count_" + str(experiment) + "_behavior" + ".npy")
            else:
                progress_preference_curve = np.load("results_cache/" + self.measure + "_count_" + str(experiment) + "_" + model_name + "_sims_" + str(self.sims_per_subject) + ".npy")
        else:
            if model_name is None:
                progress_preference_curve = get_measure_experiment(experiment=experiment, measure_name=self.measure + "_count", \
                                                     mode="measure")
                np.save("results_cache/" + self.measure + "_count_" + str(experiment) + "_behavior.npy", progress_preference_curve)
            else:
                progress_preference_curve = get_model_simulation_of_measure(experiment=experiment, model_name=model_name, measure_name=self.measure + "_count", sims_per_subject=self.sims_per_subject)
                np.save("results_cache/" + self.measure + "_count_" + str(experiment) + "_" + model_name + "_sims_" + str(self.sims_per_subject) + ".npy", progress_preference_curve)


        if model_name is not None:
            progress_preference_curve = np.mean(progress_preference_curve, axis=0)

        progress_condition_0 = progress_preference_curve[:, 0]
        progress_condition_1 = progress_preference_curve[:, 1]
        if (experiment == 1) or (experiment == "instr_1"):
            progress_condition_2 = progress_preference_curve[:, 2]

        mean_progress_condition_0 = np.nanmean(progress_condition_0, axis=0)
        mean_progress_condition_1 = np.nanmean(progress_condition_1, axis=0)
        if (experiment == 1) or (experiment == "instr_1"):
            mean_progress_condition_2 = np.nanmean(progress_condition_2, axis=0)

        stderr_progress_condition_0 = np.nanstd(progress_condition_0, axis=0) / np.sqrt(progress_condition_0.shape[0])
        stderr_progress_condition_1 = np.nanstd(progress_condition_1, axis=0) / np.sqrt(progress_condition_1.shape[0])
        if (experiment == 1) or (experiment == "instr_1"):
            stderr_progress_condition_2 = np.nanstd(progress_condition_2, axis=0) / np.sqrt(progress_condition_2.shape[0])


        conditions = self.get_condition_labels(experiment)
        ax.plot(range(7), mean_progress_condition_0, label=conditions[0], color="red")
        ax.errorbar(x=range(7), y=mean_progress_condition_0, yerr=stderr_progress_condition_0, fmt='o', capsize=5, color="red")

        ax.plot(range(7), mean_progress_condition_1, label=conditions[1], color="blue")
        ax.errorbar(x=range(7), y=mean_progress_condition_1, yerr=stderr_progress_condition_1, fmt='o', capsize=5, color="blue")

        if (experiment == 1) or (experiment == "instr_1"):
            ax.plot(range(7), mean_progress_condition_2, label=conditions[2], color="green")
            ax.errorbar(x=range(7), y=mean_progress_condition_2, yerr=stderr_progress_condition_2, fmt='o', capsize=5, color="green")

        ax.set_xticks([0, 2, 4, 6])
        if experiment == 4:
            xtick_labels = ["0-14%",  "28-42%",  "56-70%",  "84-100%"]
            fontsize = 11
        else:
            xtick_labels = ["0%", "28%", "57%", "86%"]
            fontsize = 11
        ax.set_xticklabels(xtick_labels, fontsize=fontsize, fontweight='bold')


        if self.legend:
            ax.legend(fontsize=15, loc='lower right')
        if model_name is None:
            if self.measure == "retro_value":
                ax.set_xlabel("max progress - dominant progress", fontsize=13, fontweight='bold')
            else:
                ax.set_xlabel("max/ dominant progress - next best progress", fontsize=13, fontweight='bold')
        ylim = [0, 1.1]
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=13)


    def plot_retrospective_bias_experiment(self, experiment, model_name=None, axs=None, show=True, cache=False, title=None):

        if axs is None:
            if experiment == 4:
                fig = plt.figure(figsize=(7, 5))
                gs = GridSpec(1, 1)
                gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                                           width_ratios=[0.5, 1.2])
            else:
                fig = plt.figure(figsize=(7, 5))
                gs = GridSpec(1, 1)
                gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                                           width_ratios=[1, 1.2])

            ax1 = fig.add_subplot(gs_row1[0])
            ax2 = fig.add_subplot(gs_row1[1])

        else:
            ax1 = axs[0]
            ax2 = axs[1]

        self.plot_retro_bias_per_condition(experiment=experiment, ax=ax1, model_name=model_name, cache=cache)
        self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax2, model_name=model_name, cache=cache)
        if experiment == 4:
            ax2.legend(fontsize=15, loc='upper left')
        # Experiment 2

        if title:
            if model_name == "momentum":
                title = "TD-Momentum"
            elif model_name == "prospective":
                title = "Prospective"
            elif model_name is None:
                title = "Participants"
            elif model_name == "hybrid":
                title = "Hybrid"

            ax1.set_title(title, fontsize=15, fontweight='bold')

        if model_name is not None:
            ax1.set_ylabel(None)

        if show:
            plt.savefig("figures/retro_bias_" + str(experiment) + "_" + str(self.measure) + "_" + str(model_name) + str(title) + ".png")
            plt.tight_layout()
            plt.show()



    def plot_retro_bias_compare_prospective_retrospective(self, axs=None, model_name=None, cache=False):

        if axs is None:
            fig = plt.figure(figsize=(10, 5))
            gs = GridSpec(1, 2, width_ratios=[1, 0.7])  # Set the height ratios for the subplots

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
        else:

            ax1 = axs[0]
            ax2 = axs[1]

        self.plot_retro_bias_per_condition(experiment=1, ax=ax1, model_name=model_name, compare=True, cache=cache)

        self.plot_retro_bias_per_condition(experiment=2, ax=ax2, model_name=model_name, compare=True, cache=cache)

        ax1.set_ylim([0, 1.1])
        ax2.set_ylim([0, 1.1])

        if axs is None:
            plt.tight_layout()
            plt.savefig("figures/retro_bias_compare_prospective_retrospective.png")
            plt.show()


    def plot_retro_bias_experiment_model(self, experiment=1, axs=None, model_name=None, cache=False, show_fig=True):

        if axs is None:
            fig = plt.figure(figsize=(14, 5))
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # Set the height ratios for the subplots

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])
            ax4 = plt.subplot(gs[0, 3])
        else:

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            ax4 = axs[3]

        self.plot_retro_bias_per_condition(experiment=experiment, ax=ax1, model_name=None, compare=False, cache=cache)

        self.plot_retro_bias_per_condition(experiment=experiment, ax=ax2, model_name=model_name, compare=False, cache=cache, hatch=True)
        self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax3, model_name=None, cache=cache)
        self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax4, model_name=model_name, cache=cache)

        ax1.set_ylabel("Choosing max progress", fontsize=15, fontweight='bold')
        ax4.legend().remove()
        ax1.set_title("Experiment " + str(experiment) + ": Participants", fontsize=15, fontweight='bold')
        ax2.set_title(self.get_model_labels(model_name)[0], fontsize=15, fontweight='bold')
        ax3.set_title("Participants", fontsize=15, fontweight='bold')
        ax4.set_title(self.get_model_labels(model_name)[0], fontsize=15, fontweight='bold')



        if axs is None:
            plt.tight_layout()
            plt.savefig("figures/retro_bias_" + str(experiment) + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".png")
        if show_fig:
            plt.show()

    def plot_retro_bias_experiment_4(self, axs=None, cache=False, show_fig=True, hybrid=False):

        experiment = 4

        if axs is None:
            fig = plt.figure(figsize=(21, 5))
            gs = GridSpec(1, 6, width_ratios=[0.9, 0.9, 0.9, 1, 1, 1])  # Set the height ratios for the subplots

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])
            ax4 = plt.subplot(gs[0, 3])
            ax5 = plt.subplot(gs[0, 4])
            ax6 = plt.subplot(gs[0, 5])
        else:

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            ax4 = axs[3]
            ax5 = axs[4]
            ax6 = axs[5]

        self.plot_retro_bias_per_condition(experiment=experiment, ax=ax1, model_name=None, compare=False, cache=cache)
        if not hybrid:
            self.plot_retro_bias_per_condition(experiment=experiment, ax=ax2, model_name="momentum", compare=False, cache=cache, hatch=True)
        else:
            self.plot_retro_bias_per_condition(experiment=experiment, ax=ax2, model_name="hybrid", compare=False, cache=cache, hatch=True)
        self.plot_retro_bias_per_condition(experiment=experiment, ax=ax3, model_name="prospective", compare=False, cache=cache, hatch=True)

        self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax4, model_name=None, cache=cache)
        if not hybrid:
            self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax5, model_name="momentum", cache=cache)
        else:
            self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax5, model_name="hybrid", cache=cache)
        self.plot_retro_bias_relation_to_progress_difference(experiment=experiment, ax=ax6, model_name="prospective", cache=cache)

        ax1.set_ylabel("Choosing max progress", fontsize=15, fontweight='bold')
        ax2.set_ylabel(None)
        ax3.set_ylabel(None)
        ax4.legend().remove()
        ax1.set_title("Experiment " + str(experiment) + ": Participants", fontsize=15, fontweight='bold')
        if not hybrid:
            ax2.set_title("TD-momentum", fontsize=15, fontweight='bold')
            ax5.set_title("TD-momentum", fontsize=15, fontweight='bold')
        else:
            ax2.set_title("Hybrid", fontsize=15, fontweight='bold')
            ax5.set_title("Hybrid", fontsize=15, fontweight='bold')
        ax3.set_title("Prospective", fontsize=15, fontweight='bold')
        ax6.set_title("Prospective", fontsize=15, fontweight='bold')
        ax4.set_title("Participants", fontsize=15, fontweight='bold')



        if axs is None:
            plt.tight_layout()
            if not hybrid:
                plt.savefig("figures/retro_bias_experiment_" + str(experiment) + "_sims_" + str(self.sims_per_subject) + ".png")
            else:
                plt.savefig("figures/retro_bias_experiment_" + str(experiment) + "_hybrid_sims_" + str(self.sims_per_subject) + ".png")

        if show_fig:
            plt.show()

    def plot_retro_bias_progress_instructions(self, axs=None, cache=False):

        if cache:
            pros_retro_bias_1 = np.load("results_cache/pros_retro_bias_1.npy")
            sub_retro_1 = np.load("results_cache/sub_retro_1.npy")
            sub_retro_instr = np.load("results_cache/sub_retro_instr.npy")
        else:

            pros_retro_bias_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params(
                measure_name="retro_value_count")

            sub_retro_1 = get_measure_experiment(experiment=1, measure_name="retro_value_count", \
                                                 mode="measure")
            sub_retro_instr = get_measure_experiment(experiment="instr_1", measure_name="retro_value_count", \
                                                     mode="measure")
            np.save("results_cache/pros_retro_bias_1.npy", pros_retro_bias_1)
            np.save("results_cache/sub_retro_1.npy", sub_retro_1)
            np.save("results_cache/sub_retro_instr.npy", sub_retro_instr)


        pros_retro_mean_1 = [np.mean(pros_retro_bias_1[i], axis=0) for i in range(len(pros_retro_bias_1))]
        sub_retro_mean_1 = [np.mean(sub_retro_1[i], axis=0) for i in range(len(sub_retro_1))]
        sub_retro_mean_instr = [np.mean(sub_retro_instr[i], axis=0) for i in range(len(sub_retro_instr))]

        pros_retro_mean_1 = np.nanmean(pros_retro_mean_1, axis=0)
        retro_mean_1 = np.nanmean(sub_retro_mean_1, axis=0)
        retro_mean_instr = np.nanmean(sub_retro_mean_instr, axis=0)

        pros_retro_std_1 = np.nanstd(pros_retro_mean_1, axis=0) / np.sqrt(len(pros_retro_mean_1))
        retro_std_1 = np.nanstd(sub_retro_mean_1, axis=0) / np.sqrt(len(sub_retro_mean_1))
        retro_std_instr = np.nanstd(sub_retro_mean_instr, axis=0) / np.sqrt(len(sub_retro_mean_instr))


        if axs is None:
            fig = plt.figure(figsize=(10, 5))
            gs = GridSpec(1, 2, width_ratios=[1, 0.68])  # Set the height ratios for the subplots

            ax2 = plt.subplot(gs[0, 0])

            ax1 = plt.subplot(gs[0, 1], sharey=ax2)
            show = True
        else:
            ax1 = axs[0]
            ax2 = axs[1]
            show = False

        ax1.plot(pros_retro_mean_1, color="green", label="Prospective")
        ax1.errorbar(x=range(7), y=pros_retro_mean_1, yerr=pros_retro_std_1, fmt='o',\
                     capsize=5, color="green")

        ax1.plot(retro_mean_1, color="blue", label="No instructions")
        ax1.errorbar(x=range(7), y=retro_mean_1, yerr=retro_std_1, fmt='o', \
                     capsize=5, color="blue")

        ax1.plot(retro_mean_instr, color="red", label="Instructions")
        ax1.errorbar(x=range(7), y=retro_mean_instr, yerr=retro_std_instr, fmt='o', \
                     capsize=5, color="red")

        ax1.set_xlabel("max progress - dominant progress", fontsize=16, fontweight='bold')

        ax1.legend(fontsize=15)

        ax1.set_ylim([0, 1])

        ax1.tick_params(axis='both', which='major', labelsize=15)

        self.plot_retro_bias_per_condition_instructions(ax2, cache=cache)

        plt.savefig("figures/retro_bias_progress_instructions.png")

        if show:
            plt.show()



def plot_manuscript_retrospective_bias_plots():

    cache = True
    sims_per_subject = 1
    

    # """
    # Momentum model simulations for experiments 1 and 2
    # """
    # model_name = "momentum"
    # ## 1 simulation per subject
    # sims_per_subject = 1
    # retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=sims_per_subject)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=1, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=2, axs=None, model_name=model_name, cache=cache, show_fig=False)
    #
    #
    # """
    # Alternate model simulations for experiments 1 and 2
    # """
    #
    # sims_per_subject = 1
    # # Prospective model
    # model_name = "prospective"
    # retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=sims_per_subject)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=1, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=2, axs=None, model_name=model_name, cache=cache, show_fig=False)
    #
    # # Hybrid model
    # model_name = "hybrid"
    # retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=sims_per_subject)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=1, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=2, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # #
    # # # TD-persistence model
    # model_name = "td_persistence"
    # retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=sims_per_subject)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=1, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # retro_plotter.plot_retro_bias_experiment_model(experiment=2, axs=None, model_name=model_name, cache=cache, show_fig=False)
    # #
    #
    # """
    # Momentum simulations for experiment 4
    # """
    #
    # # 1 simulation per subject

    # model_name = "momentum"
    # retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=1)
    # retro_plotter.plot_retro_bias_experiment_4(axs=None, cache=cache, show_fig=False)


    # model_name = "momentum"
    retro_plotter = RetroBiasPlotter(measure="retro_value", sims_per_subject=1)
    retro_plotter.plot_retro_bias_experiment_4(axs=None, cache=cache, show_fig=False, hybrid=True)


if __name__ == "__main__":

    """
    Plotting retrospective bias patterns for the momentum model
    """

    plot_manuscript_retrospective_bias_plots()

    # plot compare retrospective bias patterns for participants comparing with task-optimized prospective and retrospective models
    # retro_plotter = RetroBiasPlotter(measure="retro_value")
    # retro_plotter.plot_retro_bias_compare_prospective_retrospective(axs=None, model_name=None, cache=True)

