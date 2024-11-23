from models import *
from behavior import *
from reports import *

import pingouin as pg

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec



def get_switching_patterns(experiment, model_name=None, mode="rates"):

    subject_names = get_experiment_subjects(experiment)
    no_reward_rates = []
    reward_rates = []

    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        if model_name is None:
            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
            #subject_measures = ModelMeasure(subject_id=subject_id, experiment=experiment)
            if mode == "rates":
                slot_srate, prob_srate = subject_measures.get_stay_switch_condition_counts()
            else:
                slot_srate, prob_srate = subject_measures.get_stay_switch_condition_progress_counts()
        else:
            try:
                with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
                    params = model_fits['params']
                    params = list(params.values())
                    runmodel = RunModel(experiment, model_name, params)
                    model_res = runmodel.get_model_res()
                    subject_measures = SubjectMeasure(subject_id=-1,
                                                      experiment=experiment,
                                                      model_res=model_res)
                    if mode == "rates":
                        slot_srate, prob_srate = subject_measures.get_stay_switch_condition_counts()
                    else:
                        slot_srate, prob_srate = subject_measures.get_stay_switch_condition_progress_counts()

            except:
                continue

        no_reward_switches = np.concatenate((slot_srate[:, 0], prob_srate[:, 0]))
        reward_switches = np.concatenate((slot_srate[:, 1], prob_srate[:, 1]))

        no_reward_rates.append(no_reward_switches)
        reward_rates.append(reward_switches)

    # concatenate max slot and max prob switch rates across conditions
    no_reward_rates = np.array(no_reward_rates)
    reward_rates = np.array(reward_rates)

    return no_reward_rates, reward_rates


class SwitchingPatterns:
    def __init__(self, experiment, model_name=None):
        self.experiment = experiment
        self.model_name = model_name

    def get_figure(self, figsize=(5, 5)):
        if self.experiment != 4:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        return ax

    def get_condition_labels(self, mode):
        if mode == "rates":
            if self.experiment == 1:
                conditions = ['80-20', '70-30', '60-40', '80-20', '70-30', '60-40']
            elif self.experiment == 2:
                conditions = ['75-25', '55-45', '75-25', '55-45']
            elif self.experiment == 4:
                conditions = ['H disp', 'L disp', 'H disp', 'L disp']
        else:
            conditions = ["L diff", "H diff", "L diff", "H diff"]
        return conditions

    def get_ylabel(self):
        if self.experiment == 1 or self.experiment == 4:
            ylabel = "Probability of Switching"
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
        if mode == "rates":
            if self.experiment == 1:
                xpos = [0, 1, 2, 3.5, 4.5, 5.5]
            elif self.experiment == 2:
                xpos = [0, 1, 2.5, 3.5]
            elif self.experiment == 4:
                xpos = [0, 1.3, 2.5, 3.5]
        else:
            xpos = [0, 1.3, 2.8, 4.1]
        return xpos

    def get_colors(self, mode):
        if mode == "rates":
            colors = ['#f08080', '#FAFAD2', '#90ee90', '#add8e6', ]
        else:
            colors = ['#90ee90', '#add8e6', '#f08080', '#FAFAD2', ]
        return colors

    def get_ylim(self, mode):
        if mode == "rates":
            ylim = [0, 0.75]
        else:
            ylim = [0, 0.8]

        return ylim

    def get_title(self):
        if self.experiment == 4:
            title = None
        else:
            title = "Experiment " + str(self.experiment)
        if self.model_name is not None:
            title = None
        return title


    def get_switch_stats(self, no_reward_rates, reward_rates):
        overall = np.concatenate((no_reward_rates, reward_rates), axis=0)
        tt = pg.ttest(np.nanmean(overall[:, 0:3], axis=1), np.nanmean(overall[:, 3:], axis=1),
                      paired=True)
        print(tt)
        print(tt['p-val'], tt['T'], tt['dof'])

    def get_anova_stats(self, no_reward_rates, reward_rates):
        overall = np.concatenate((no_reward_rates, reward_rates), axis=0)
        if self.experiment == 1:
            data_dict = {'subjects': list(range(len(overall))) * 3,
                         'condition': [0] * len(overall[:, 0] - overall[:, 3]) + [
                             1] * len(overall[:, 1] - overall[:, 4]) + [2] * len(
                             overall[:, 2] - overall[:, 5]),
                         'measures': list(overall[:, 0] - overall[:, 3]) + \
                                     list(overall[:, 1] - overall[:, 4]) + \
                                     list(overall[:, 2] - overall[:, 5])}
        else:
            data_dict = {'subjects': list(range(len(overall))) * 2,
                         'condition': [0] * len(overall[:, 0] - overall[:, 2]) + [
                             1] * len(overall[:, 1] - overall[:, 3]),
                         'measures': list(overall[:, 0] - overall[:, 2]) + list(
                             overall[:, 1] - overall[:, 3])}

        df = pd.DataFrame(data_dict)

        rm_anova_result = pg.rm_anova(data=df, dv='measures', \
                                      within='condition',
                                      subject='subjects')

        print(rm_anova_result)

    def show_significance_markers(self, no_reward_rates, reward_rates, mode, show):

        if show:
            xpos = self.get_bars_xpos(mode)
            if mode == "rates":
                #get_switch_stats(no_reward_rates, reward_rates)
                self.get_anova_stats(no_reward_rates, reward_rates)
                if self.experiment == 1:
                    draw_significance_marker(ax, 0, 3.5, 0.45, 0.01, 'ns')
                    draw_significance_marker(ax, 1, 4.5, 0.52, 0.01, '*')
                    draw_significance_marker(ax, 2, 5.5, 0.59, 0.01, '***')
                elif self.experiment == 2:
                    draw_significance_marker(ax, 0, 2.5, 0.5, 0.01, '*')
                    draw_significance_marker(ax, 1, 3.5, 0.6, 0.01, '***')
                elif self.experiment == 4:
                    draw_significance_marker(ax, xpos[0], xpos[2], 0.55, 0.01, '*')
                    draw_significance_marker(ax, xpos[1], xpos[3], 0.62, 0.01,'**')
            else:
                self.get_switch_stats(no_reward_rates, reward_rates)
                if self.experiment == 1:
                    draw_significance_marker(ax, xpos[0], xpos[1], 0.4, 0.01, '**')
                    draw_significance_marker(ax, xpos[2], xpos[3], 0.5, 0.01,
                                             '**')
                elif self.experiment == 2:
                    print(pg.ttest(reward_rates[:, 2], reward_rates[:, 3], paired=True)['p-val'])
                    draw_significance_marker(ax, xpos[0], xpos[1], 0.4, 0.01, '***')
                    draw_significance_marker(ax, xpos[2], xpos[3], 0.6, 0.01,
                                             '**')
                elif self.experiment == 4:
                    draw_significance_marker(ax, xpos[0], xpos[1], 0.4, 0.01, '***')
                    draw_significance_marker(ax, xpos[2], xpos[3], 0.7, 0.01,
                                             '***')

    def plot_stay_switch_rates_experiment(self, ax, mode, show=True, cache=False, plot_fig=False):
        experiment = self.experiment
        model_name = self.model_name
        if ax is None:
            ax = self.get_figure()
        if not cache:
            no_reward_rates, reward_rates = get_switching_patterns(experiment=experiment, model_name=model_name, mode=mode)
            data = np.vstack((no_reward_rates, reward_rates)).T
            data = data[~np.isnan(data).any(axis=1)]
            no_reward_rates = data[:, 0]
            reward_rates = data[:, 1]

            np.save("figures_cache/no_reward_rates_exp_" + str(experiment) + "_" + \
                    mode + "_" + str(model_name) + ".npy", no_reward_rates)
            np.save("figures_cache/reward_rates_exp_" + str(experiment) + "_" + \
                    mode + "_" + str(model_name) + ".npy", reward_rates)
        else:
            no_reward_rates = np.load("figures_cache/no_reward_rates_exp_" + str(experiment) + "_" + \
                                      mode + "_" + str(model_name) + ".npy")
            reward_rates = np.load("figures_cache/reward_rates_exp_" + str(experiment) + "_" + \
                                   mode + "_" + str(model_name) + ".npy")


        conditions = self.get_condition_labels(mode)

        models = ['no token', 'token']

        ylabel = self.get_ylabel()
        legend = self.get_legend()

        add_text_annotation(ax, "retrospective choice", 0.25, -0.20,
                            fontsize=self.get_marker_font())
        add_text_annotation(ax, "prospective choice", 0.75, -0.20, fontsize=self.get_marker_font())

        data = [no_reward_rates, reward_rates]
        mean_values = [np.nanmean(no_reward_rates, axis=0),
                       np.nanmean(reward_rates, axis=0)]
        std_dev_values = [np.nanstd(no_reward_rates, axis=0),
                          np.nanstd(reward_rates, axis=0)] / np.sqrt(len(no_reward_rates))


        print(experiment, model_name)

        if model_name is not None:
            show = False

        #self.show_significance_markers(no_reward_rates, reward_rates, mode, show)
        colors = self.get_colors(mode)
        ylim = self.get_ylim(mode)
        xpos = self.get_bars_xpos(mode)
        title = self.get_title()
        plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, conditions, models, title, ylabel, ylim,
                                  bar_width=0.3, x_pos=xpos, draw_data=False, colors=colors, legend=legend)

        if plot_fig:
            plt.tight_layout()
            plt.savefig("figures/stay_switch_rates_" + str(experiment) + "_" + str(model_name) + "_" + str(mode) + ".png")
            plt.show()


def plot_stay_switch_rates_experiments_1_and_2(axs=None, model_name=None, mode="rates", show=False, cache=False, plot_fig=False):
    #
    if not axs:
        fig = plt.figure(figsize=(12, 5))
        if mode == "rates":
            gs = GridSpec(1, 2, width_ratios=[1.25, 1])
        else:
            gs = GridSpec(1, 2, width_ratios=[1, 1])
        axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    experiment = 1

    SwitchingPatterns(experiment, model_name).plot_stay_switch_rates_experiment(axs[0], mode, show, cache)

    experiment = 2
    SwitchingPatterns(experiment, model_name).plot_stay_switch_rates_experiment(axs[1], mode, show, cache)


    if plot_fig:
        plt.tight_layout()
        plt.savefig("figures/stay_switch_rates_12_" + str(model_name) +  "_" + str(mode) + str(show) + ".png")
        plt.show()


if __name__ == "__main__":

    #compare_behavior_model_all(cache=True)
    model_name = "momentum"
    model_name = "rescorla"
    model_name = None

    plot_stay_switch_rates_experiments_1_and_2(axs=None, model_name=model_name, show=True,
                           cache=False, mode="rates", plot_fig=True)



    #plot_stay_switch_rates_experiment(ax=None, experiment=4, model_name=None, mode="progress", show=True, cache=True, plot_fig=True)

