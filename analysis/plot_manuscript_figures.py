from primary_arbitration import plot_goal_valuation
from retrospective_bias import *
from switching_patterns import plot_stay_switch_rates
from performance import *
from performance_instructions import plot_performance_instructions
from model_comparisons_aggregate import *
from model_comparisons_individual import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def get_behavior_metrics():

    # Create a figure
    fig = plt.figure(figsize=(11,  24))

    # Create GridSpec
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.1, 1.2, 0.1, 1.2])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=[0.12, 0.5, 0.3, 0.12])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[2], width_ratios=[0.25, 0.03, 0.3, 0.05, 0.2, 0.03, 0.3])
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4], width_ratios=[0.05, 0.5, 0.42, 0.05])

    # Top row plots
    ax1 = fig.add_subplot(gs_row1[1])
    ax2 = fig.add_subplot(gs_row1[2])
    plot_goal_valuation(axs=[ax1, ax2], show=False, cache=True)


    # Middle row plots
    ax3 = fig.add_subplot(gs_row2[0])
    ax4 = fig.add_subplot(gs_row2[2])
    ax5 = fig.add_subplot(gs_row2[4])
    ax6 = fig.add_subplot(gs_row2[6])
    plot_retrospective_bias(model_name=None, axs=[ax3, ax4, ax5, ax6], print=False, cache=True)
    ax3.set_title('Experiment 1', fontsize=11)
    ax5.set_title('Experiment 2', fontsize=11)

    # Bottom full-width plot
    ax7 = fig.add_subplot(gs_row3[1])
    ax8 = fig.add_subplot(gs_row3[2])
    plot_stay_switch_rates(axs=[ax7, ax8], model_name=None, experiments=[1, 2], show=True, cache=True)
    ax7.set_title('Experiment 1', fontsize=11)
    ax8.set_title('Experiment 2', fontsize=11)

    # Add labels
    fig.text(0.08, 0.85, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.5, 'B', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.15, 'C', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.18, hspace=0.6)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Goal selection probabilities', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.63, 'Proportion of retrospectively-biased choice', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.32, 'Probability of switching away from current choice', ha='center', fontsize=11, fontweight='bold')

    # Show plot
    plt.show()


def plot_performance_figures():

    # Create a figure
    fig = plt.figure(figsize=(12, 18))

    # Create GridSpec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 0.05, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], width_ratios=[1, 1, 0.1, 1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], width_ratios=[0.1, 0.6, 0.4, 0.1])
    #gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4], width_ratios=[0.5, 0.1, 0.5, 0.5])


    # Top row plots
    ax1 = fig.add_subplot(gs_row1[0])
    ax2 = fig.add_subplot(gs_row1[1])
    ax3 = fig.add_subplot(gs_row1[3])
    ax4 = fig.add_subplot(gs_row1[4])

    perf_plotter = PerformancePlotter(cache=True)
    perf_plotter.plot_performance_experiments(axs=[ax1, ax2, ax3, ax4])

    # Middle row plots
    ax5 = fig.add_subplot(gs_row2[1])
    ax6 = fig.add_subplot(gs_row2[2])
    plot_retro_bias_compare_prospective_retrospective(axs=[ax5, ax6], model_name=None, cache=True)

    # Bottom full-width plot
    # ax7 = fig.add_subplot(gs_row3[0])
    # ax8 = fig.add_subplot(gs_row3[2])
    # ax9 = fig.add_subplot(gs_row3[3])
    # plot_performance_instructions(axs=[ax7])
    # plot_retro_bias_progress_instructions(axs=[ax9, ax8])

    # Add labels
    fig.text(0.08, 0.85, 'A', fontsize=16, fontweight='bold', va='center')
    fig.text(0.08, 0.5, 'B', fontsize=16, fontweight='bold', va='center')
    #fig.text(0.08, 0.15, 'C', fontsize=16, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.15, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Task performance', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.48, 'Proportion of retrospectively-biased choice', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()


def plot_instructions_figure():
    # Create a figure
    fig = plt.figure(figsize=(10, 4))

    # Create GridSpec
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])
    plot_performance_instructions(axs=ax0, cache=True)
    ax0.set_title('Task performance', fontsize=11)
    plot_retro_bias_progress_instructions(axs=[ax2, ax1], cache=True)
    title = ax1.set_title('Proportion of retrospectively-biased choice', fontsize=11)
    title.set_position((0.9, 1.05))

    fig.text(0.08, 0.95, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.4, 0.95, 'B', fontsize=13, fontweight='bold', va='center')

    plt.show()


def plot_model_fits():

    # Create a figure
    fig = plt.figure(figsize=(15, 8))

    # Create GridSpec
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.1, 1, 0.3, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0],\
                                               width_ratios=[0.1, 1, 0.1, 1, 0.1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[2], \
                                               width_ratios=[0.1, 1, 0.1, 1, 0.1, 1])
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[4], \
                                               width_ratios=[ 1, 0.1, 1, 0.1, 1, 0.05, 1])


    # Top row plots
    ax1 = fig.add_subplot(gs_row1[1])
    ax2 = fig.add_subplot(gs_row1[3])
    ax3 = fig.add_subplot(gs_row1[5])

    plot_model_fits_experiment(experiment=1, axs=[ax1, ax2, ax3])

    # Middle row plots
    ax5 = fig.add_subplot(gs_row2[1])
    ax6 = fig.add_subplot(gs_row2[3])
    ax7 = fig.add_subplot(gs_row2[5])
    plot_model_fits_experiment(experiment=2, axs=[ax5, ax6, ax7])

    # Bottom full-width plot
    ax7 = fig.add_subplot(gs_row3[0])
    ax8 = fig.add_subplot(gs_row3[2])
    ax9 = fig.add_subplot(gs_row3[4])
    ax10 = fig.add_subplot(gs_row3[6])

    plot_model_fits_instructions(axs=[ax7, ax8])
    plot_model_comparisions_individual(experiments=[1, "instr_1"], axs=[ax9, ax10])

    # Add labels
    fig.text(0.08, 0.95, 'A', fontsize=16, fontweight='bold', va='center')
    #fig.text(0.08, 0.5, 'B', fontsize=16, fontweight='bold', va='center')
    fig.text(0.08, 0.35, 'B', fontsize=16, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.15, hspace=0.5)

    # Add titles for each row
    #fig.text(0.5, 0.95, 'Task performance', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.48, 'Proportion of retrospectively-biased choice', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    get_behavior_metrics()
    #plot_performance_figures()
    #plot_instructions_figure()
    #plot_model_fits()