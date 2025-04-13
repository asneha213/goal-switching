from models import *
from behavior import *
from reports import *

import pingouin as pg
import numpy as np

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec



def get_switching_patterns(experiment, model_name=None, mode="rates", sims_per_subject=1):

    """
    Get switching patterns in the task
    experiment: int (1, 2, 4)
    model_name: str (momentum, rescorla, prospective, td_persistence, hybrid)
    mode: str (rates, progress)
     - rates: switching rates away from retrospective and prospective suits split by reward and no-reward conditions and also by block condition
     - progress: switching rates away from retrospective and prospective suits based on progress differences (low or high) between the two suits
    sims_per_subject: int(1, 30) - number of simulations per subject

    Returns:
    retro_switch_rates: array of shape (num_subjects, num_conditions) - switching rates away from retrospective suit
    pros_switch_rates: array of shape (num_subjects, num_conditions) - switching rates away from prospective suit
    Switching patterns for both 'token received' and 'no token received' conditions are concatenated in the same array
    """

    subject_names = get_experiment_subjects(experiment)

    # switches away from retrospective and prospective suits for each simulation of population
    retro_switch_rates_all_sims = []
    pros_switch_rates_all_sims = []

    for sim in range(sims_per_subject):
        retro_switch_rates = []
        pros_switch_rates = []

        for subject_num in range(len(subject_names)):
            subject_id = subject_names[subject_num]
            if model_name is None:
                subject_measures = ModelMeasure(subject_id=subject_id, experiment=experiment)
                if mode == "rates":
                    retro_switches, pros_switches = subject_measures.get_stay_switch_condition_counts()
                else:
                    retro_switches, pros_switches = subject_measures.get_stay_switch_condition_progress_counts()
            else:
                with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
                    params = model_fits['params']
                    runmodel = RunModel(experiment, model_name, params)
                    model_res = runmodel.get_model_res()
                    subject_measures = ModelMeasure(subject_id=-1,
                                                      experiment=experiment,
                                                      model_res=model_res)
                    if mode == "rates":
                        retro_switches, pros_switches = subject_measures.get_stay_switch_condition_counts()
                    else:
                        retro_switches, pros_switches = subject_measures.get_stay_switch_condition_progress_counts()



            retro_switches_all = retro_switches.flatten(order='F')
            pros_switches_all = pros_switches.flatten(order='F')

            retro_switch_rates.append(retro_switches_all)
            pros_switch_rates.append(pros_switches_all)

        retro_switch_rates = np.array(retro_switch_rates)
        pros_switch_rates = np.array(pros_switch_rates)

        retro_switch_rates_all_sims.append(retro_switch_rates)
        pros_switch_rates_all_sims.append(pros_switch_rates)

    retro_switch_rates_all_sims = np.array(retro_switch_rates_all_sims)
    pros_switch_rates_all_sims = np.array(pros_switch_rates_all_sims)

    retro_switch_rates = np.nanmean(retro_switch_rates_all_sims, axis=0)
    pros_switch_rates = np.nanmean(pros_switch_rates_all_sims, axis=0)

    return retro_switch_rates, pros_switch_rates


class SwitchingPatterns:
    """
    Plot switching patterns in the task
    """
    def __init__(self, experiment, model_name=None, sims_per_subject=1):
        self.experiment = experiment
        self.model_name = model_name
        self.sims_per_subject = sims_per_subject

    def get_figure(self, figsize=(5, 5)):
        if self.experiment != 4:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        return ax

    def get_annotation(self, mode):
        if mode == "rates":
            annot = "A"
        elif mode == "progress":
            annot = "B"
        return annot


    def get_condition_labels(self, mode):
        if mode == "rates" or mode == "compare":
            if self.experiment == 1:
                conditions = ['80-20', '70-30', '60-40', '80-20', '70-30', '60-40']
            elif self.experiment == 2:
                conditions = ['75-25', '55-45', '75-25', '55-45']
            elif self.experiment == 4:
                conditions = ['H disp', 'L disp', 'H disp', 'L disp']
        elif mode == "progress":
            conditions = ["L diff", "H diff", "L diff", "H diff"]

        return conditions

    def get_ylabel(self):
        if self.experiment == 1 or self.experiment == 4:
            ylabel = "probability of switching"
        else:
            ylabel = None
        if self.model_name is not None:
            ylabel = None
        return ylabel

    def get_legend(self):
        if self.experiment == 1 or self.experiment == 4:
            legend = True
        else:
            legend = False
        return legend

    def get_marker_font(self):
        if self.experiment == 1:
            marker_font = 14
        else:
            marker_font = 14
        return marker_font

    def get_bars_xpos(self, mode):
        if mode == "rates" or mode == "compare":
            if self.experiment == 1:
                xpos = [0, 1, 2, 3.5, 4.5, 5.5]
            elif self.experiment == 2:
                xpos = [0, 1, 2.5, 3.5]
            elif self.experiment == 4:
                xpos = [0, 1, 2.5, 3.5]
        elif mode == "progress":
            xpos = [0, 1, 2.5, 3.5]
        return xpos

    def get_colors(self, mode):
        if mode == "rates":
            colors = ["lightsalmon", "palegreen", "lightsalmon", "palegreen"]
        elif mode == "progress":
            colors = ["lightcoral", "lightgreen", "lightcoral", "lightgreen"]
        elif mode == "compare":
            colors = ["lightsalmon", "palegreen"]
        return colors

    def get_ylim(self, mode, experiment):
        if mode == "rates" or mode == "compare":
            ylim = [0, 1.0]
        elif mode == "progress":
            ylim = [0, 1.0]
        if experiment == 4:
            ylim = [0, 1.0]

        return ylim

    def get_title(self):
        if self.experiment == 4:
            title = None
        else:
            title = "Experiment " + str(self.experiment)
        if self.model_name is not None:
            if self.model_name == "momentum":
                title = "TD-momentum"
            elif self.model_name == "rescorla":
                title = "Rescorla-Wagner"
            elif self.model_name == "prospective":
                title = "Prospective"
            elif self.model_name == "td_persistence":
                title = "TD-persistence"
            elif self.model_name == "hybrid":
                title = "Hybrid"
        return title


    def get_model_labels(self):
        if self.model_name is None:
            models = ['away from retrospective suit - participants', 'away from prospective suit - participants']
        elif self.model_name == "momentum":
            models = ['away from retrospective suit',  'away from retrospective suit']
        elif self.model_name == "rescorla":
            models = ['away from retrospective suit - rescorla wagner', 'away from prospective suit - rescorla wagner']
        elif self.model_name == "prospective":
            models = ['away from retrospective suit - prospective', 'away from prospective suit - prospective']
        elif self.model_name == "td_persistence":
            models = ['away from retrospective suit - TD-persistence', 'away from prospective suit - TD-persistence']
        elif self.model_name == "hybrid":
            models = ['away from retrospective suit - hybrid', 'away from prospective suit - hybrid']
        return models


    def get_switch_stats(self, retro_switch_rates, pros_switch_rates, mode="rates"):
        """
        Get significance metrics of the switching rates
        """
        print("Experiment: " + str(self.experiment))
        if mode == "rates":
            """
            t-test between the switching rates away from retrospective and prospective suits for the no-token condition
            """
            print("T-test results under no-token-received condition")
            tt1 = pg.ttest(retro_switch_rates[:, 0], pros_switch_rates[:, 0])
            tt2 = pg.ttest(retro_switch_rates[:, 1], pros_switch_rates[:, 1])
            print(tt1['p-val'])
            print(tt2['p-val'])
            if self.experiment == 1:
                tt3 = pg.ttest(retro_switch_rates[:,2], pros_switch_rates[:,2])
                print(tt3['p-val'])

            """
            t-test between the switching rates away from retrospective and prospective suits for the token condition
            """
            print("T-test results under token-received condition")
            if self.experiment == 1:
                tt1 = pg.ttest(retro_switch_rates[:, 3], pros_switch_rates[:, 3])
                tt2 = pg.ttest(retro_switch_rates[:, 4], pros_switch_rates[:, 4])
                tt3 = pg.ttest(retro_switch_rates[:,5], pros_switch_rates[:,5])
                print(tt1['p-val'])
                print(tt2['p-val'])
                print(tt3['p-val'])
            elif self.experiment == 2 or self.experiment == 4:
                tt1 = pg.ttest(retro_switch_rates[:, 2], pros_switch_rates[:, 2])
                tt2 = pg.ttest(retro_switch_rates[:, 3], pros_switch_rates[:, 3])
                print(tt1['p-val'])
                print(tt2['p-val'])
        else:
            print("T-test results under no-token-received condition")
            tt1 = pg.ttest(retro_switch_rates[:, 0], pros_switch_rates[:, 0])
            tt2 = pg.ttest(retro_switch_rates[:, 1], pros_switch_rates[:, 1])
            print(tt1['p-val'])
            print(tt2['p-val'])

            print("T-test results under token-received condition")
            tt1 = pg.ttest(retro_switch_rates[:, 2], pros_switch_rates[:, 2])
            tt2 = pg.ttest(retro_switch_rates[:, 3], pros_switch_rates[:, 3])
            print(tt1['p-val'])
            print(tt2['p-val'])

        # overall t-test between pros_switch_rates and retro_switch_rates in both token and no-token conditions
        print("T-test results overall")
        tt = pg.ttest(np.nanmean(retro_switch_rates, axis=1), np.nanmean(pros_switch_rates, axis=1))
        print(tt)
        print(tt['p-val'])

        print("No-token vs token")

        if self.experiment == 1:
            tt1 = pg.ttest(np.nanmean(retro_switch_rates[:, 0:3], axis=1), np.nanmean(retro_switch_rates[:, 3:6], axis=1))
            tt2 = pg.ttest(np.nanmean(pros_switch_rates[:, 0:3], axis=1), np.nanmean(pros_switch_rates[:, 3:6], axis=1))
            print(tt1)
            print(tt1['p-val'])
            print(tt2)
            print(tt2['p-val'])
        elif self.experiment == 2:
            tt1 = pg.ttest(np.nanmean(retro_switch_rates[:, 0:2], axis=1), np.nanmean(retro_switch_rates[:, 2:4], axis=1))
            tt2 = pg.ttest(np.nanmean(pros_switch_rates[:, 0:2], axis=1), np.nanmean(pros_switch_rates[:, 2:4], axis=1))
            print(tt1)
            print(tt1['p-val'])
            print(tt2)
            print(tt2['p-val'])


    def get_anova_stats(self, retro_switch_rates, pros_switch_rates):
        """
        Get switching rates interaction with block conditions
        """
        if self.experiment == 1:

            print("prospective:")
            pros_switch_rates_c = np.vstack([pros_switch_rates[:,0:3], pros_switch_rates[:, 3:6]])
            data_dict = {
                            'subjects': list(range(len(pros_switch_rates_c))) * 3,
                            'condition': [0] * len(pros_switch_rates_c[:, 0]) + [
                                1] * len(pros_switch_rates_c[:, 1]) + [2] * len(
                                pros_switch_rates_c[:, 2]),
                            'measures': list(pros_switch_rates_c[:, 0]) + \
                                        list(pros_switch_rates_c[:, 1]) + \
                                        list(pros_switch_rates_c[:, 2])}

        elif self.experiment == 2:
            print("prospective:")
            pros_switch_rates_c = np.vstack([pros_switch_rates[:,0:2], pros_switch_rates[:, 2:4]])
            data_dict = {
                            'subjects': list(range(len(pros_switch_rates_c))) * 2,
                            'condition': [0] * len(pros_switch_rates_c[:, 0]) + [
                                1] * len(pros_switch_rates_c[:, 1]),
                            'measures': list(pros_switch_rates_c[:, 0]) + \
                                        list(pros_switch_rates_c[:, 1])}



        df = pd.DataFrame(data_dict)

        rm_anova_result = pg.rm_anova(data=df, dv='measures', \
                                      within='condition',
                                      subject='subjects')

        print("ANOVA results for Experiment " + str(self.experiment))
        print(rm_anova_result)
        print(rm_anova_result['p-unc'])

    def show_significance_markers(self, ax, retro_switch_rates, pros_switch_rates, mode, show):
        """
        Plot significance markers of t-test comparisons on the bar plots
        """

        if show:
            xpos = self.get_bars_xpos(mode)
            if mode == "rates":
                self.get_switch_stats(retro_switch_rates, pros_switch_rates, mode)
                if self.experiment == 1:
                    draw_significance_marker(ax, 0, 0.25, 0.5, 0.01, 'ns')
                    draw_significance_marker(ax, 1, 1.25, 0.5, 0.01, 'ns')
                    draw_significance_marker(ax, 2, 2.25, 0.58, 0.01, '**')
                    draw_significance_marker(ax, 3.5, 3.75, 0.40, 0.01, 'ns')
                    draw_significance_marker(ax, 4.5, 4.75, 0.52, 0.01, '*')
                    draw_significance_marker(ax, 5.5, 5.75, 0.59, 0.01, '**')
                elif self.experiment == 2:
                    draw_significance_marker(ax, 0, 0.25, 0.5, 0.01, 'ns')
                    draw_significance_marker(ax, 1, 1.25, 0.69, 0.01, '*')
                    draw_significance_marker(ax, 2.5, 2.75, 0.50, 0.01, 'ns')
                    draw_significance_marker(ax, 3.5, 3.75, 0.82, 0.01, '***')
                elif self.experiment == 4:
                    draw_significance_marker(ax, 0, 0.25, 0.60, 0.01, 'ns')
                    draw_significance_marker(ax, 1, 1.25, 0.60, 0.01, '*')
                    draw_significance_marker(ax, 2.5, 2.75, 0.56, 0.01, '**')
                    draw_significance_marker(ax, 3.5, 3.75, 0.64, 0.01, '***')
            else:
                self.get_switch_stats(retro_switch_rates, pros_switch_rates, mode)
                if self.experiment == 1:
                    draw_significance_marker(ax, 0, 0.25, 0.52, 0.01, '***')
                    draw_significance_marker(ax, 1, 1.25, 0.60, 0.01, '***')
                    draw_significance_marker(ax, 2.5, 2.75, 0.44, 0.01, '***')
                    draw_significance_marker(ax, 3.5, 3.75, 0.52, 0.01, '***')
                elif self.experiment == 2:
                    draw_significance_marker(ax, 0, 0.25, 0.58, 0.01, '***')
                    draw_significance_marker(ax, 1, 1.25, 0.64, 0.01, '***')
                    draw_significance_marker(ax, 2.5, 2.75, 0.5, 0.01, '***')
                    draw_significance_marker(ax, 3.5, 3.75, 0.64, 0.01, '***')
                elif self.experiment == 4:
                    draw_significance_marker(ax, 0, 0.25, 0.62, 0.01, 'ns')
                    draw_significance_marker(ax, 1, 1.25, 0.68, 0.01, '***')
                    draw_significance_marker(ax, 2.5, 2.75, 0.58, 0.01, '***')
                    draw_significance_marker(ax, 3.5, 3.75, 0.71, 0.01, '***')


    def plot_stay_switch_rates_experiment(self, ax, experiment, model_name, mode, show=True, cache=False, plot_fig=False):
        """
        Plot switching patterns of an experiment
        show - show significance markers
        cache - save the results in a cache file
        plot_fig - plot the figure
        """
        if ax is None:
            ax = self.get_figure()
        if not cache:
            retro_switch_rates, pros_switch_rates = get_switching_patterns(experiment=experiment, model_name=model_name, mode=mode, sims_per_subject=self.sims_per_subject)

            np.save("results_cache/retro_switch_rates_exp_" + str(experiment) + "_" + \
                     mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy", retro_switch_rates)
            np.save("results_cache/pros_switch_rates_exp_" + str(experiment) + "_" + \
                    mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy", pros_switch_rates)
        else:
            print("Loading from cache")
            print(self.sims_per_subject)
            retro_switch_rates = np.load("results_cache/retro_switch_rates_exp_" + str(experiment) + "_" + \
                                      mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy")
            pros_switch_rates = np.load("results_cache/pros_switch_rates_exp_" + str(experiment) + "_" + \
                                   mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy")


        conditions = self.get_condition_labels(mode)

        models = ['away from retrospective suit', 'away from prospective suit']

        ylabel = self.get_ylabel()
        legend = self.get_legend()


        add_text_annotation(ax, "No token", 0.25, -0.20,
                            fontsize=self.get_marker_font())
        add_text_annotation(ax, "Token", 0.75, -0.20, fontsize=self.get_marker_font())

        data = [retro_switch_rates, pros_switch_rates]

        mean_values = [np.nanmean(retro_switch_rates, axis=0),
                          np.nanmean(pros_switch_rates, axis=0)]

        std_dev_values = np.array([np.nanstd(retro_switch_rates, axis=0),
                            np.nanstd(pros_switch_rates, axis=0)]) / np.sqrt(len(retro_switch_rates))


        print(experiment, model_name)

        if model_name is not None:
            show = False
        else:
            #pass
            self.show_significance_markers(ax, retro_switch_rates, pros_switch_rates, mode, show)
            #self.get_anova_stats(retro_switch_rates, pros_switch_rates)

        colors = self.get_colors(mode)
        ylim = self.get_ylim(mode, experiment)
        xpos = self.get_bars_xpos(mode)
        title = self.get_title()

        if mode == "compare" and self.model_name is not None:
            hatch_val = ['//', '//']
        else:
            hatch_val = [None, None]


        plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.3, x_pos=xpos, draw_data=False, colors=colors, legend=legend, hatch=hatch_val)

        plt.tight_layout()
        plt.savefig("figures/stay_switch_rates_" + str(experiment) + "_" + str(model_name) + "_" + str(mode) + "_sims_" + str(self.sims_per_subject) + ".png")
        if plot_fig:
            plt.show()

    def plot_stay_switch_rates_experiment_model(self, ax, mode, show=True, cache=False, plot_fig=False):
        """
        Plot the model predictions of switching patterns against participant patterns
        """
        experiment = self.experiment
        model_name = self.model_name
        if ax is None:
            fig = plt.figure(figsize=(12, 5))
            gs = GridSpec(1, 2, width_ratios=[1, 1])  # Set the height ratios for the subplots

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
        if not cache:
            retro_switch_rates, pros_switch_rates = get_switching_patterns(experiment=experiment, model_name=model_name, mode=mode, sims_per_subject=self.sims_per_subject)

            np.save("results_cache/retro_switch_rates_exp_" + str(experiment) + "_" + \
                    mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy", retro_switch_rates)
            np.save("results_cache/pros_switch_rates_exp_" + str(experiment) + "_" + \
                    mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy", pros_switch_rates)

        else:
            retro_switch_rates = np.load("results_cache/retro_switch_rates_exp_" + str(experiment) + "_" + \
                                         mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy")
            pros_switch_rates = np.load("results_cache/pros_switch_rates_exp_" + str(experiment) + "_" + \
                                        mode + "_" + str(model_name) + "_sims_" + str(self.sims_per_subject) + ".npy")

        retro_switches_all = retro_switch_rates
        pros_switches_all = pros_switch_rates

        conditions = self.get_condition_labels(mode)
        models = self.get_model_labels()

        ylabel = self.get_ylabel()


        add_text_annotation(ax2, "No token", 0.25, -0.20,
                            fontsize=self.get_marker_font())
        add_text_annotation(ax2, "Token", 0.75, -0.20, fontsize=self.get_marker_font())

        data = [retro_switches_all, pros_switches_all]

        mean_values = [np.nanmean(retro_switches_all, axis=0),
                            np.nanmean(pros_switches_all, axis=0)]

        std_dev_values = np.array([np.nanstd(retro_switches_all, axis=0),
                            np.nanstd(pros_switches_all, axis=0)]) / np.sqrt(len(retro_switches_all))



        print(experiment, model_name)

        colors = self.get_colors(mode)
        ylim = self.get_ylim(mode, experiment)
        xpos = self.get_bars_xpos(mode)
        title = self.get_title()


        plot_comparative_bar_plot(ax2, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.3, x_pos=xpos, draw_data=False, colors=colors, legend=False, hatch=["//", "//"])


        self.plot_stay_switch_rates_experiment(ax1, experiment, model_name=None, mode="rates", show=False, cache=True, plot_fig=False)
        ax1.set_ylabel("probability of switching")
        ax1.set_title("Experiment " + str(experiment) + ": Participants", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("figures/stay_switch_rates_" + str(experiment) + "_" + str(model_name) + "_" + str(mode) + "_sims_" + str(self.sims_per_subject) + ".png")

        if plot_fig:
            plt.show()





    def plot_switching_patterns_experiment_4(self, ax, mode, show=True, cache=False, plot_fig=False):
        """
        Plot switching patterns for experiment 4: comparing participant patterns with model predictions of TD-momentum and prospective models
        """
        if ax is None:
            fig = plt.figure(figsize=(14, 5))
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1])  # Set the height ratios for the subplots

            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])

        self.plot_stay_switch_rates_experiment(ax1, experiment=4, model_name=None, mode=mode, show=False, cache=cache, plot_fig=False)
        self.plot_stay_switch_rates_experiment(ax2, experiment=4, model_name="momentum", mode=mode, show=show, cache=cache, plot_fig=False)
        self.plot_stay_switch_rates_experiment(ax3, experiment=4, model_name="prospective", mode=mode, show=show, cache=cache, plot_fig=False)


        ax1.set_title("Participants", fontsize=16, fontweight='bold')
        ax2.set_title("TD-momentum", fontsize=16, fontweight='bold')
        ax3.set_title("Prospective", fontsize=16, fontweight='bold')
        ax1.legend([])
        ax2.legend([])
        ax2.set_ylabel(None)
        ax3.set_ylabel(None)
        plt.tight_layout()
        plt.savefig("figures/stay_switch_rates_" + str(4) + "_" + str(mode) + "_sims_" + str(self.sims_per_subject) + ".png")

        if plot_fig:
            plt.show()


def plot_switching_patterns_experiment_1_and_2(mode, show=True, cache=False, plot_fig=False, sims_per_subject=1):
    """
    Plot switching patterns for experiment 1 and 2
    """
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[1.33, 1])  # Set the height ratios for the subplots

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    SwitchingPatterns(experiment=1, model_name=None, sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment(ax1, experiment=1, model_name=None, mode=mode, show=show, cache=cache, plot_fig=False)
    SwitchingPatterns(experiment=2, model_name=None, sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment(ax2, experiment=2, model_name=None, mode=mode, show=show, cache=cache, plot_fig=False)

    ax1.set_title("Experiment 1", fontsize=16, fontweight='bold')
    ax2.set_title("Experiment 2", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/stay_switch_rates_" + str(1) + "_" + str(2) + "_" + str(mode) + "_sims_" + str(sims_per_subject) + ".png")

    if plot_fig:
        plt.show()



def plot_switching_patterns_manuscript():
    """
    Manuscript figures of switching patterns
    """

    cache = True
    plot_fig = True


    sims_per_subject = 1
    # ## Switching rates (behavior) away from retrospective and prospective suit in Experiment 1, 2
    #plot_switching_patterns_experiment_1_and_2(mode="rates", show=True, cache=cache, plot_fig=plot_fig, sims_per_subject=sims_per_subject)
    plot_switching_patterns_experiment_1_and_2(mode="progress", show=True, cache=cache, plot_fig=plot_fig, sims_per_subject=sims_per_subject)
    #
    # #
    # ### Participant vs model comparison for Experiment 1, 2
    #
    # ## TD-momentum (sim_per_subject = 1)
    #SwitchingPatterns(experiment=1, model_name="momentum", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # SwitchingPatterns(experiment=2, model_name="momentum", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #
    # ## Prospective (sim_per_subject = 1)
    # SwitchingPatterns(experiment=1, model_name="prospective", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # SwitchingPatterns(experiment=2, model_name="prospective", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #
    # ## TD-persistence (sim_per_subject = 1)
    # SwitchingPatterns(experiment=1, model_name="td_persistence", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # SwitchingPatterns(experiment=2, model_name="td_persistence", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #
    # ## Hybrid (sim_per_subject = 1)
    # SwitchingPatterns(experiment=1, model_name="hybrid", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # SwitchingPatterns(experiment=2, model_name="hybrid", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # #
    # ## Rescorla (sim_per_subject = 1)
    #SwitchingPatterns(experiment=1, model_name="rescorla", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #SwitchingPatterns(experiment=2, model_name="rescorla", sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment_model(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #
    # SwitchingPatterns(experiment=4, model_name=None, sims_per_subject=sims_per_subject).plot_stay_switch_rates_experiment(ax=None, experiment=4, model_name=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    # ## Participant vs model comparison for Experiment 4
    #
    # SwitchingPatterns(experiment=4, model_name=None, sims_per_subject=sims_per_subject).plot_switching_patterns_experiment_4(ax=None, mode="rates", show=True, cache=cache, plot_fig=plot_fig)
    #
    #
    #



if __name__ == "__main__":

    plot_switching_patterns_manuscript()

