from behavior import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import pingouin as pg


class PerformancePlotter:

    def __init__(self, cache=False):
            if not cache:
                self.cache_data()

    def cache_data(self):

        # # # Behavior
        # sub_performance_1 = get_measure_experiment(experiment=1, measure_name="num_goals", mode="condition")
        # sub_performance_2 = get_measure_experiment(experiment=2, measure_name="num_goals", mode="condition")
        # sub_performance_4 = get_measure_experiment(experiment=4, measure_name="num_goals", mode="condition")
        sub_performance_instr_1 = get_measure_experiment(experiment="instr_1", measure_name="num_goals", mode="condition")
        #
        # # Prospective
        # pros_performance_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params()
        # pros_performance_2 = ModelOptimizer(experiment=2, model_name="prospective").simulate_params()
        # pros_performance_4 = ModelOptimizer(experiment=4, model_name="prospective").simulate_params()
        #
        # # Retrospective
        # retro_performance_1 = ModelOptimizer(experiment=1, model_name="retrospective").simulate_params()
        # retro_performance_2 = ModelOptimizer(experiment=2, model_name="retrospective").simulate_params()
        # retro_performance_4 = ModelOptimizer(experiment=4, model_name="retrospective").simulate_params()
        #
        # # Momentum
        # mom_performance_1 = ModelOptimizer(experiment=1, model_name="momentum").simulate_params()
        # mom_performance_2 = ModelOptimizer(experiment=2, model_name="momentum").simulate_params()
        # mom_performance_4 = ModelOptimizer(experiment=4, model_name="momentum").simulate_params()
        #
        # # Persistence
        # pers_performance_1 = ModelOptimizer(experiment=1, model_name="td_persistence").simulate_params()
        # pers_performance_2 = ModelOptimizer(experiment=2, model_name="td_persistence").simulate_params()
        # pers_performance_4 = ModelOptimizer(experiment=4, model_name="td_persistence").simulate_params()

        # np.save("figures_cache/sub_performance_1.npy", sub_performance_1)
        # np.save("figures_cache/sub_performance_2.npy", sub_performance_2)
        # np.save("figures_cache/sub_performance_4.npy", sub_performance_4)
        # np.save("figures_cache/pros_performance_1.npy", pros_performance_1)
        # np.save("figures_cache/pros_performance_2.npy", pros_performance_2)
        # np.save("figures_cache/pros_performance_4.npy", pros_performance_4)
        # np.save("figures_cache/retro_performance_1.npy", retro_performance_1)
        # np.save("figures_cache/retro_performance_2.npy", retro_performance_2)
        # np.save("figures_cache/retro_performance_4.npy", retro_performance_4)
        # np.save("figures_cache/mom_performance_1.npy", mom_performance_1)
        # np.save("figures_cache/mom_performance_2.npy", mom_performance_2)
        # np.save("figures_cache/mom_performance_4.npy", mom_performance_4)
        # np.save("figures_cache/pers_performance_1.npy", pers_performance_1)
        # np.save("figures_cache/pers_performance_2.npy", pers_performance_2)
        # np.save("figures_cache/pers_performance_4.npy", pers_performance_4)

        np.save("figures_cache/sub_performance_instr_1.npy", sub_performance_instr_1)


    def plot_performance_instructions(self):
        sub_performance_instr_1 = np.load("figures_cache/sub_performance_instr_1.npy")
        pros_performance_1 = np.load("figures_cache/pros_performance_1.npy")
        sub_performance_1 = np.load("figures_cache/sub_performance_1.npy")
        # plot bar plot comparing performance of models

        fig = plt.figure(figsize=(7, 6))

        gs = GridSpec(1, 1, figure=fig)

        ax1 = fig.add_subplot(gs[0])

        colors = ['#90ee90', '#add8e6', '#f08080', 'lightgray', ]

        data_1 = [pros_performance_1, sub_performance_1, sub_performance_instr_1]

        conditions = ["80-20", "70-30", "60-40"]
        mean_values = [np.mean(pros_performance_1, axis=0), np.mean(sub_performance_1, axis=0), np.mean(sub_performance_instr_1, axis=0)]
        std_values = [np.std(pros_performance_1, axis=0), np.std(sub_performance_1, axis=0), np.std(sub_performance_instr_1, axis=0)]

        models = ["Prospective (task-optimized)", "Participants: No instructions (experiment 1)", "Participants: Instructions (experiment 3)"]
        title = "Experiment 1"

        ylabel = "Suits collected"
        ylim = [0, 24]

        plot_comparative_bar_plot(ax1, data_1, mean_values, std_values, conditions, models, title, ylabel, ylim,
                                    bar_width=0.14, colors=colors)

        plt.tight_layout()
        plt.savefig("figures/performance_instructions.png")

        plt.show()


    def plot_performance_comparison_models(self):
        pros_performance_1 = np.load("figures_cache/pros_performance_1.npy")
        pros_performance_2 = np.load("figures_cache/pros_performance_2.npy")
        pros_performance_4 = np.load("figures_cache/pros_performance_4.npy")
        retro_performance_1 = np.load("figures_cache/retro_performance_1.npy")
        retro_performance_2 = np.load("figures_cache/retro_performance_2.npy")
        retro_performance_4 = np.load("figures_cache/retro_performance_4.npy")
        mom_performance_1 = np.load("figures_cache/mom_performance_1.npy")
        mom_performance_2 = np.load("figures_cache/mom_performance_2.npy")
        mom_performance_4 = np.load("figures_cache/mom_performance_4.npy")
        pers_performance_1 = np.load("figures_cache/pers_performance_1.npy")
        pers_performance_2 = np.load("figures_cache/pers_performance_2.npy")
        pers_performance_4 = np.load("figures_cache/pers_performance_4.npy")

        # plot bar plot comparing performance of models

        fig = plt.figure(figsize=(10, 5))

        gs = GridSpec(1, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        colors = ['#90ee90', '#add8e6', '#f08080', 'lightgray', ]

        data_1 = [pros_performance_1, retro_performance_1, mom_performance_1, pers_performance_1]

        conditions = ["80-20", "70-30", "60-40"]
        mean_values = [np.mean(pros_performance_1, axis=0), np.mean(retro_performance_1, axis=0), np.mean(mom_performance_1, axis=0), np.mean(pers_performance_1, axis=0)]
        std_values = [np.std(pros_performance_1, axis=0), np.std(retro_performance_1, axis=0), np.std(mom_performance_1, axis=0),   np.std(pers_performance_1, axis=0)]

        models = ["Prospective", "Retrospective", "TD-Momentum", "TD-Persistence"]
        title = "Experiment 1"
        ylabel = "Suits collected"
        ylim = [0, 24]

        plot_comparative_bar_plot(ax1, data_1, mean_values, std_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.19, colors=colors)


        data_2 = [pros_performance_2, retro_performance_2, mom_performance_2, pers_performance_2]

        conditions = ["75-25", "55-45"]
        mean_values = [np.mean(pros_performance_2, axis=0), np.mean(retro_performance_2, axis=0), np.mean(mom_performance_2, axis=0), np.mean(pers_performance_2, axis=0)]

        std_values = [np.std(pros_performance_2, axis=0), np.std(retro_performance_2, axis=0), np.std(mom_performance_2, axis=0), np.std(pers_performance_2, axis=0)]

        models = ["Prospective", "Retrospective", "TD-Momentum", "TD-Persistence"]
        title = "Experiment 2"
        ylabel = "Suits collected"
        ylim = [0, 24]

        plot_comparative_bar_plot(ax2, data_2, mean_values, std_values, conditions, models, title, ylabel, ylim,
                                bar_width=0.19, colors=colors, legend=None)

        data_4 = [pros_performance_4, retro_performance_4, mom_performance_4,
                  pers_performance_4]

        conditions = ["H disp", "L disp"]
        mean_values = [np.mean(pros_performance_4, axis=0), np.mean(retro_performance_4, axis=0),
                          np.mean(mom_performance_4, axis=0), np.mean(pers_performance_4, axis=0)]
        std_values = [np.std(pros_performance_4, axis=0), np.std(retro_performance_4, axis=0),
                        np.std(mom_performance_4, axis=0), np.std(pers_performance_4, axis=0)]

        models = ["Prospective", "Retrospective", "TD-Momentum", "TD-Persistence"]
        title = "Experiment 3"
        ylabel = "Suits collected"
        ylim = [0, 24]

        plot_comparative_bar_plot(ax3, data_4, mean_values, std_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.19, colors=colors, legend=None)

        plt.tight_layout()
        plt.show()


    def plot_performance_experiments(self, axs=None):
            sub_performance_1 = np.load("figures_cache/sub_performance_1.npy")
            sub_performance_2 = np.load("figures_cache/sub_performance_2.npy")
            sub_performance_4 = np.load("figures_cache/sub_performance_4.npy")
            pros_performance_1 = np.load("figures_cache/pros_performance_1.npy")
            pros_performance_2 = np.load("figures_cache/pros_performance_2.npy")
            pros_performance_4 = np.load("figures_cache/pros_performance_4.npy")
            retro_performance_1 = np.load("figures_cache/retro_performance_1.npy")
            retro_performance_2 = np.load("figures_cache/retro_performance_2.npy")
            retro_performance_4 = np.load("figures_cache/retro_performance_4.npy")

            # Compare overall subject performance with prospective model

            print("Experiment 1")
            tt = pg.ttest(np.mean(sub_performance_1, axis=1), np.mean(pros_performance_1, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            print("Experiment 2")
            tt = pg.ttest(np.mean(sub_performance_2, axis=1), np.mean(pros_performance_2, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            print("Experiment 4")
            tt = pg.ttest(np.mean(sub_performance_4, axis=1), np.mean(pros_performance_4, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            # Compare overall subject performance with retrospective model
            print("Experiment 1")
            tt = pg.ttest(np.mean(sub_performance_1, axis=1), np.mean(retro_performance_1, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            print("Experiment 2")
            tt = pg.ttest(np.mean(sub_performance_2, axis=1), np.mean(retro_performance_2, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            print("Experiment 4")
            tt = pg.ttest(np.mean(sub_performance_4, axis=1), np.mean(retro_performance_4, axis=1))
            print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

            # print interactions for prospective and subject performance

            print("Experiment 1")
            print("Prospective")
            get_anova_conditions(1, pros_performance_1)
            print("Subject")
            get_anova_conditions(1, sub_performance_1)

            print("Experiment 2")
            print("Prospective")
            get_anova_conditions(2, pros_performance_2)
            print("Subject")
            get_anova_conditions(2, sub_performance_2)

            print("Experiment 4")
            print("Prospective")
            get_anova_conditions(4, pros_performance_4)
            print("Subject")
            get_anova_conditions(4, sub_performance_4)
            print("Retro")
            get_anova_conditions(4, retro_performance_4)


            """
            Plot human performance against model performance: 
            """

            data_1 = [pros_performance_1, sub_performance_1, retro_performance_1]
            data_2 = [pros_performance_2, sub_performance_2, retro_performance_2]

            if axs is None:
                show = True
                fig = plt.figure(figsize=(16, 5))
                gs = GridSpec(1, 4, width_ratios=[1.4, 1, 1, 1])  # Set the height ratios for the subplots

                axs = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]), \
                       plt.subplot(gs[3])]
            else:
                show = False

            conditions = ["80-20", "70-30", "60-40"]
            mean_values = [np.mean(pros_performance_1, axis=0), np.mean(sub_performance_1, axis=0),
                           np.mean(retro_performance_1, axis=0)]
            std_dev_values = np.array([np.std(pros_performance_1, axis=0), np.std(sub_performance_1, axis=0),
                                       np.std(retro_performance_1, axis=0)]) / np.sqrt(len(sub_performance_1))
            models = ["Prospective (task-optimized)", "Participants", "Retrospective (task-optimized)"]
            title = "Experiment 1"
            ylabel = "Suits collected"
            ylim = [5, 25]

            plot_comparative_bar_plot(axs[0], data_1, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                      bar_width=0.19)

            conditions = ["75-25", "55-45"]
            mean_values = [np.mean(pros_performance_2, axis=0), np.mean(sub_performance_2, axis=0),
                           np.mean(retro_performance_2, axis=0)]
            std_dev_values = np.array([np.std(pros_performance_2, axis=0), np.std(sub_performance_2, axis=0),
                                       np.std(retro_performance_2, axis=0)]) / np.sqrt(len(sub_performance_2))

            title = "Experiment 2"
            ylim = [0, 24]
            ylabel = None
            plot_comparative_bar_plot(axs[2], data_2, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                      bar_width=0.19, legend=None)
            #axs[1].set_yticklabels([])




            """
            Plot the distribution of performance for each model
            """

            sub_performance_1 = get_measure_experiment(experiment=1, measure_name="performance", mode="measure")
            pros_perforance_1 = np.sum(pros_performance_1, axis=1)

            retro_perforance_1 = np.sum(retro_performance_1, axis=1)

            measures = [pros_perforance_1[:63], sub_performance_1, retro_perforance_1[:63]]

            xlabel = "Performance"
            ylabel = "Density"
            labels = ["Prospective (optimal)", "Participants", "Retrospective (optimal)"]

            # Add title boxes to each subplot


            plot_multiple_histograms(axs[1], measures, labels, xlabel, ylabel, title=None, legend=None)

            # Experiment 2
            sub_performance_2 = get_measure_experiment(experiment=2, measure_name="performance", mode="measure")
            pros_perforance_2 = np.sum(pros_performance_2, axis=1)
            retro_perforance_2 = np.sum(retro_performance_2, axis=1)

            measures = [pros_perforance_2, sub_performance_2, retro_perforance_2]

            plot_multiple_histograms(axs[3], measures, labels, xlabel, ylabel=None, title=None, legend=None)

            if show:
                plt.tight_layout()
                plt.savefig("figures/performance_comparison.png")
                plt.show()


    def get_mean_and_variance_performance(self):
        sub_performance_1 = np.load("figures_cache/sub_performance_1.npy")
        sub_performance_2 = np.load("figures_cache/sub_performance_2.npy")

        performance_1 = np.sum(sub_performance_1, axis=1)
        performance_2 = np.sum(sub_performance_2, axis=1)

        print("Experiment 1")
        print("Mean", np.mean(performance_1))
        print("Std", np.std(performance_1))
        print("Max", np.max(performance_1))
        print("Min", np.min(performance_1))

        print("Experiment 2")
        print("Mean", np.mean(performance_2))
        print("Std", np.std(performance_2))
        print("Max", np.max(performance_2))
        print("Min", np.min(performance_2))


if __name__ == "__main__":
    perf_plotter = PerformancePlotter(cache=True)
    #perf_plotter.plot_performance_experiments()
    #perf_plotter.plot_performance_comparison_models()
    #plot_performance_comparison_models()
    #get_mean_and_variance_performance()
    perf_plotter.plot_performance_instructions()
