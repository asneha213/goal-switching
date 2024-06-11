from model_comparisons_aggregate import get_model_BIC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from behavior import *
import pingouin as pg


def plot_bic_diff(ax, bics0, bics1, titles, ylim, title="", ylabel=True):

    # Sample sorted array
    sorted_array = np.sort(bics0 - bics1)

    # Iterate through the array and plot bars
    for idx, value in enumerate(sorted_array):
        if value < 0:
            ax.bar(idx, value, color='blue', align='center')
        else:
            ax.bar(idx, value, color='red', align='center')

    # Set y-axis limits based on the minimum and maximum array values
    ax.set_ylim(min(sorted_array) - 1, max(sorted_array) + 1)

    # Add colored arrows and text annotations for greater and less than zero
    ax.annotate(titles[0], xy=(0.35, 0.3), xycoords='axes fraction',
                xytext=(0.35, 0.1), textcoords='axes fraction',
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='<-', linewidth=2),
                ha='center', va='center', color='blue')

    ax.annotate(titles[1], xy=(0.35, 0.5), xycoords='axes fraction',
                xytext=(0.35, 0.8), textcoords='axes fraction',
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2),
                ha='center', va='center', color='red')

    # Set labels and title
    ax.set_xlabel('subjects', fontsize=11)
    if ylabel is not None:
        ax.set_ylabel('BIC difference', fontsize=11)
    #ax.set_title('Colored Rectangular Bars of a Sorted Array')

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylim(ylim)

    if title != "":
        # Add title boxes to each subplot
        title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

        ax.annotate(title, xy=(0.5, 1.15), xycoords='axes fraction',
                    fontsize=10, ha='center', bbox=title_box_props)


def plot_bics_diffs_two_experiments(axs, bics_exp1_m1, bics_exp1_m2, \
                                    bics_exp2_m1, bics_exp2_m2, titles, ylims, show=True):

    plot_bic_diff(axs[0], bics_exp1_m1, bics_exp1_m2, titles, ylims[0], title="No instructions")
    plot_bic_diff(axs[1], bics_exp2_m1, bics_exp2_m2, titles, ylims[1], title="Instructions", ylabel=None)

    if show:
        plt.tight_layout()
        plt.show()


def plot_model_comparisions_individual(experiments=[1, 2], AIC=False, axs=None):
    experiment = experiments[0]

    bics_td_1, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_nt_1, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    experiment = experiments[1]

    bics_td_2, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_nt_2, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    if axs is None:
        fig = plt.figure(figsize=(8, 4))

        if experiments[1] == 2:
            gs = GridSpec(1, 2, width_ratios=[1.2, 1])
        else:
            gs = GridSpec(1, 2, width_ratios=[1, 1])

        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
        show = True
    else:
        show = False

    plot_bics_diffs_two_experiments(axs, bics_nt_1, bics_td_1, bics_nt_2, bics_td_2, \
                                    ["TD Persistence better", "TD Momentum better"], \
                                    [[-180, 330], [-180, 250]], show=show)


def get_performance_difference_two_groups_instructions():
    experiment = "instr_1"
    AIC = False

    bics_td, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_nt, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    subjects_mom = np.where(bics_td - bics_nt < 0)[0]
    subjects_nt = np.where(bics_td - bics_nt > 0)[0]

    retro_bias_instr = get_measure_experiment(experiment="instr_1", measure_name="retro_value_count", mode="measure")
    retro_bias_mom_mean = np.mean(np.nanmean(retro_bias_instr[subjects_mom], axis=0), axis=0)
    retro_bias_nt_mean = np.mean(np.nanmean(retro_bias_instr[subjects_nt], axis=0), axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.plot(range(7), retro_bias_mom_mean, label="explained by TD-Momentum", color="red")
    ax.plot(range(7), retro_bias_nt_mean, label="explained by TD-Persistence", color="blue")

    ax.legend(fontsize=12)
    ax.set_xlabel("progress: retrospective - prospective", fontsize=11)
    ylim = [0, 1]
    ax.set_ylim(ylim)

    ax.set_ylabel("Retrospectively biased choice", fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_model_comparisions_individual(experiments=[1, "instr_1"])
    #get_performance_difference_two_groups_instructions()
