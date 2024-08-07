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


def plot_behavior_metrics():

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

    retro_plotter = RetroBiasPlotter()
    retro_plotter.plot_retrospective_bias(model_name=None, axs=[ax3, ax4, ax5, ax6], print=False, cache=True)
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


def plot_goal_valuations():

    # Create a figure
    fig = plt.figure(figsize=(12,  8))

    # Create GridSpec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.01, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], width_ratios=[1.35, 0.05, 1, 0.05, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2], width_ratios=[1.35, 0.05, 1, 0.05, 1])


    # Top row plots
    ax1 = fig.add_subplot(gs_row1[0])
    ax2 = fig.add_subplot(gs_row1[2])
    ax3 = fig.add_subplot(gs_row1[4])
    plot_goal_valuation(axs=[ax1, ax2, ax3], show=False, cache=True)

    # Top row plots
    ax4 = fig.add_subplot(gs_row2[0])
    ax5 = fig.add_subplot(gs_row2[2])
    ax6 = fig.add_subplot(gs_row2[4])
    plot_goal_valuation(axs=[ax4, ax5, ax6], show=False, cache=True, measure="obvious_choice")


    # Add labels
    fig.text(0.10, 0.85, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.10, 0.35, 'B', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.18, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Goal selection probabilities: prospective and retrospective suits diverge', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.49, 'Goal selection probabilities: prospective and retrospective suits are the same', ha='center', fontsize=11, fontweight='bold')

    # Show plot
    plt.show()



def plot_retro_value_figures():

    # Create a figure
    fig = plt.figure(figsize=(15,  24))

    # Create GridSpec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.1, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[0],\
                                               width_ratios=[0.7, 0.8, 0.05, 0.5, 0.8, 0.05,  0.5, 0.9])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[2], \
                                               width_ratios=[0.7, 0.8, 0.05, 0.5, 0.8, 0.05, 0.5, 0.9])

    # Top row plots
    ax1 = fig.add_subplot(gs_row1[0])
    ax2 = fig.add_subplot(gs_row1[1])
    ax3 = fig.add_subplot(gs_row1[3])
    ax4 = fig.add_subplot(gs_row1[4])
    ax5 = fig.add_subplot(gs_row1[6])
    ax6 = fig.add_subplot(gs_row1[7])

    retro_plotter = RetroBiasPlotter()
    retro_plotter.plot_retrospective_bias(model_name=None,
                                          axs=[ax1, ax2, ax3, ax4, ax5, ax6], print=False,
                                          cache=True)
    ax1.set_title('Experiment 1', fontsize=12)
    ax3.set_title('Experiment 2', fontsize=12)
    ax5.set_title('Experiment 3', fontsize=12)


    # Middle row plots
    ax4 = fig.add_subplot(gs_row2[0])
    ax5 = fig.add_subplot(gs_row2[1])
    ax6 = fig.add_subplot(gs_row2[3])
    ax7 = fig.add_subplot(gs_row2[4])
    ax8 = fig.add_subplot(gs_row2[6])
    ax9 = fig.add_subplot(gs_row2[7])

    retro_plotter = RetroBiasPlotter(measure="obvious_choice")
    retro_plotter.plot_retrospective_bias(model_name=None, axs=[ax4, ax5, ax6, ax7, ax8, ax9], print=False, cache=True)
    ax4.set_title('Experiment 1', fontsize=12)
    ax6.set_title('Experiment 2', fontsize=12)
    ax8.set_title('Experiment 3', fontsize=12)


    # Add labels
    fig.text(0.13, 0.9, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.13, 0.4, 'B', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.18, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Retrospective choice probability when prospective and retrospective suits diverge', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.47, 'Prospective/ retrospective choice probability when prospective and retrospective suits are the same', ha='center', fontsize=11, fontweight='bold')

    # Show plot
    plt.show()


def plot_switching_patterns(cache=False):

    # Create a figure
    fig = plt.figure(figsize=(15, 12))

    # Create GridSpec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1 ])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1.3, 1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1,  1,  1])

    # Bottom full-width plo
    ax1 = fig.add_subplot(gs_row1[0])
    ax2 = fig.add_subplot(gs_row1[1])
    ax3 = fig.add_subplot(gs_row1[2])
    plot_stay_switch_rates(axs=[ax1, ax2, ax3], model_name=None,show=False, cache=cache, mode="rates")
    ax1.set_title('Experiment 1', fontsize=12)
    ax2.set_title('Experiment 2', fontsize=12)
    ax3.set_title('Experiment 3', fontsize=12)

    # Bottom full-width plot
    ax4 = fig.add_subplot(gs_row2[0])
    ax5 = fig.add_subplot(gs_row2[1])
    ax6 = fig.add_subplot(gs_row2[2])
    plot_stay_switch_rates(axs=[ax4, ax5, ax6], model_name=None,
                           show=False, cache=cache, mode="progress")
    ax1.set_title('Experiment 1', fontsize=12)
    ax2.set_title('Experiment 2', fontsize=12)
    ax3.set_title('Experiment 3', fontsize=12)

    # Add labels
    fig.text(0.13, 0.9, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.13, 0.4, 'B', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.18, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Increased tendency to switch away from the prospective choice over the retrospective choice', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.48, 'Progress disparity between retrospective and prospective suits effects switching behavior', ha='center', fontsize=11, fontweight='bold')

    # Show plot
    #plt.tight_layout()
    plt.show()


def plot_performance_figures():

    # Create a figure
    fig = plt.figure(figsize=(14, 8))

    # Create GridSpec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 0.05, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[0], width_ratios=[1.5, 1, 1, 0.1, 1, 1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2], width_ratios=[1.5, 0.1, 1, 0.1, 1])
    #gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4], width_ratios=[0.5, 0.1, 0.5, 0.5])


    # Top row plots
    ax1 = fig.add_subplot(gs_row1[0])
    ax2 = fig.add_subplot(gs_row1[1])
    ax3 = fig.add_subplot(gs_row1[2])
    ax4 = fig.add_subplot(gs_row1[4])
    ax5 = fig.add_subplot(gs_row1[5])
    ax6 = fig.add_subplot(gs_row1[6])

    perf_plotter = PerformancePlotter(cache=True)
    perf_plotter.plot_performance_experiments(axs=[ax1, ax2, ax3, ax4, ax5, ax6])

    # Middle row plots
    ax7 = fig.add_subplot(gs_row2[0])
    ax8 = fig.add_subplot(gs_row2[2])
    ax9 = fig.add_subplot(gs_row2[4])
    retro_plotter = RetroBiasPlotter()
    retro_plotter.plot_retro_bias_compare_prospective_retrospective(axs=[ax7, ax8, ax9],\
                                                                    model_name=None, cache=True)

    # Bottom full-width plot


    # Add labels
    fig.text(0.1, 0.9, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.1, 0.4, 'B', fontsize=13, fontweight='bold', va='center')
    #fig.text(0.08, 0.15, 'C', fontsize=16, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.15, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Task performance', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.44, 'Proportion of retrospective choice', ha='center', fontsize=11, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()





def plot_instructions_figure():
    # Create a figure
    fig = plt.figure(figsize=(13, 4))

    # Create GridSpec
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.05, 1, 0.8])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])
    plot_performance_instructions(axs=ax0, cache=True)
    title = ax0.set_title('Task performance', fontsize=12)
    title.set_position((0.6, 0.95))

    retro_plotter = RetroBiasPlotter()
    retro_plotter.plot_retro_bias_progress_instructions(axs=[ax2, ax1], cache=True)
    title = ax1.set_title('Proportion of retrospectively-biased choice', fontsize=12)
    title.set_position((0.9, 0.95))

    fig.text(0.01, 0.95, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.4, 0.95, 'B', fontsize=13, fontweight='bold', va='center')

    #fig.text(0.5, 0.95, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    fig.subplots_adjust(left=0.12, hspace=0.2)

    # Adjust layout to make space for the labels
    plt.tight_layout()
    plt.show()


def plot_model_fits():

    # Create a figure
    fig = plt.figure(figsize=(14, 20))

    # Create GridSpec
    gs = gridspec.GridSpec(7, 1, height_ratios=[1, 0.1, 1, 0.1, 1, 0.5, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0],\
                                               width_ratios=[0.1, 1, 0.1, 1, 0.1, 1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[2], \
                                               width_ratios=[0.1, 1, 0.1, 1, 0.1, 1])
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[4], \
                                               width_ratios=[0.1, 1, 0.1, 1,
                                                             0.1, 1])
    gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[6], \
                                               width_ratios=[ 1, 0.1, 1, 0.1, 0.8, 0.05, 0.8])


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

    # Middle row plots
    ax8 = fig.add_subplot(gs_row3[1])
    ax9 = fig.add_subplot(gs_row3[3])
    ax10 = fig.add_subplot(gs_row3[5])
    plot_model_fits_experiment(experiment=4, axs=[ax8, ax9, ax10])

    # Bottom full-width plot
    ax11 = fig.add_subplot(gs_row4[0])
    ax12 = fig.add_subplot(gs_row4[2])
    ax13 = fig.add_subplot(gs_row4[4])
    ax14 = fig.add_subplot(gs_row4[6])

    plot_model_fits_instructions(axs=[ax11, ax12])
    plot_model_comparisions_individual(experiments=[1, "instr_1"], axs=[ax13, ax14])

    # Add labels
    fig.text(0.08, 0.93, 'A', fontsize=13, fontweight='bold', va='center')
    #fig.text(0.08, 0.5, 'B', fontsize=16, fontweight='bold', va='center')
    fig.text(0.08, 0.25, 'B', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.15, hspace=0.9)

    # Add titles for each row
    #fig.text(0.5, 0.95, 'Task performance', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.48, 'Proportion of retrospectively-biased choice', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    #plt.tight_layout()
    plt.show()


def plot_model_predictions(cache=False, model_name="td_persistence", measure="retro_value"):

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1])

    if measure == "retro_value":
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0], \
                                                   width_ratios=[0.8, 0.8, 0.5, 0.8,  0.5, 0.8])
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[2], \
                                                   width_ratios=[0.8, 0.8, 0.5, 0.8,  0.5, 0.8])
    else:
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1.35, 1, 1])
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1.35, 1, 1])


    def plot_model_predictions_model(model_name, gs):

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2], sharey=ax1)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        if measure == "retro_value":
            ax4 = fig.add_subplot(gs[3], sharey=ax1)
            ax5 = fig.add_subplot(gs[4])
            ax6 = fig.add_subplot(gs[5], sharey=ax5)
            plt.setp(ax4.get_yticklabels(), visible=False)
            plt.setp(ax6.get_yticklabels(), visible=False)

        if measure == "retro_value":
            retro_plotter = RetroBiasPlotter()
            retro_plotter.plot_retrospective_bias(model_name=model_name, axs=[ax1, ax2, ax3, ax4, ax5, ax6], print=False, cache=cache)
        else:
            plot_stay_switch_rates([ax1, ax2, ax3], model_name=model_name, cache=cache, show=False)

    plot_model_predictions_model(model_name=None, gs=gs_row1)
    plot_model_predictions_model(model_name=model_name, gs=gs_row2)

    # Add labels
    fig.text(0.08, 0.9, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.4, 'B', fontsize=13, fontweight='bold', va='center')


    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.13, hspace=0.4)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Behavior', ha='center', fontsize=12, fontweight='bold')
    if model_name == "momentum":
        fig.text(0.5, 0.48, 'TD-Momentum', ha='center', fontsize=12, fontweight='bold')
    elif model_name == "rescorla":
        fig.text(0.5, 0.48, 'Rescorla Wagner', ha='center', fontsize=12, fontweight='bold')
    elif model_name == "prospective":
        fig.text(0.5, 0.48, 'Prospective', ha='center', fontsize=12, fontweight='bold')
    elif model_name == "td_persistence":
        fig.text(0.5, 0.48, 'TD-Persistence', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()

def plot_model_predictions_rescorla():


    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[0], \
                                               width_ratios=[0.5, 0.7, 0.4, 0.7 ,0.1, 1.3, 1.1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[2], \
                                               width_ratios=[0.5, 0.7, 0.4, 0.7 ,0.1, 1.3, 1.1])

    def plot_model_predictions_model(model_name, gs):

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2], sharey=ax1)
        ax4 = fig.add_subplot(gs[3], sharey=ax1)
        ax5 = fig.add_subplot(gs[5])
        ax6 = fig.add_subplot(gs[6], sharey=ax5)

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        plt.setp(ax6.get_yticklabels(), visible=False)

        retro_plotter = RetroBiasPlotter()
        retro_plotter.plot_retrospective_bias(model_name=model_name, axs=[ax1, ax2, ax3, ax4], print=False, cache=True)

        plot_stay_switch_rates([ax5, ax6], model_name=model_name, cache=True)

    plot_model_predictions_model(model_name=None, gs=gs_row1)
    plot_model_predictions_model(model_name="rescorla", gs=gs_row2)

    # Add labels
    fig.text(0.08, 0.85, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.5, 'B', fontsize=13, fontweight='bold', va='center')


    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.13, hspace=0.7)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Behavior', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.48, 'Rescorla', ha='center', fontsize=12, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()


def plot_model_predictions_all(metric='stay_switch'):

    # Set width ratios for each row

    if metric == 'stay_switch':
        fig = plt.figure(figsize=(10, 13))
        gs = gridspec.GridSpec(7, 1, height_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1])

        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1, 0.1, 1])
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1, 0.1, 1])
        gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4], width_ratios=[1, 0.1, 1])
        gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[6], width_ratios=[1, 0.1, 1])
    else:
        fig = plt.figure(figsize=(9, 40))
        gs = gridspec.GridSpec(7, 1, height_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1])

        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], width_ratios=[0.8, 0.7, 0.1, 0.6, 0.6])
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[2], width_ratios=[0.8, 0.7, 0.1, 0.6, 0.6])
        gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[4], width_ratios=[0.8, 0.7, 0.1, 0.6, 0.6])
        gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[6], width_ratios=[0.8, 0.7, 0.1, 0.6, 0.6])


    def plot_model_predictions_model(model_name, gs, metric=metric, ylabel=False, legend=False):
        if metric != "stay_switch":

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            ax3 = fig.add_subplot(gs[3])
            ax4 = fig.add_subplot(gs[4], sharey=ax3)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax4.get_yticklabels(), visible=False)

            retro_plotter = RetroBiasPlotter(legend=legend, ylabel=ylabel)
            retro_plotter.plot_retrospective_bias(model_name=model_name, axs=[ax1, ax2, ax3, ax4], print=False, cache=True)
        else:
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[2])

            plot_stay_switch_rates([ax1, ax2], model_name=model_name, cache=True)


    plot_model_predictions_model(model_name=None, gs=gs_row1)
    plot_model_predictions_model(model_name="momentum", gs=gs_row2)
    plot_model_predictions_model(model_name="prospective", gs=gs_row3)
    plot_model_predictions_model(model_name="td_persistence", gs=gs_row4, ylabel=True, legend=True)

    # Add labels
    fig.text(0.08, 0.85, 'A', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.65, 'B', fontsize=13, fontweight='bold', va='center')

    fig.text(0.08, 0.45, 'C', fontsize=13, fontweight='bold', va='center')
    fig.text(0.08, 0.25, 'D', fontsize=13, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.16, hspace=0.8)

    # Add titles for each row
    fig.text(0.5, 0.92, 'Behavior', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.70, 'TD-Momentum', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.48, 'Prospective', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.25, 'TD-Persistence', ha='center', fontsize=11, fontweight='bold')
    #fig.text(0.5, 0.32, 'Effect of instructions', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()


if __name__ == '__main__':
    #plot_behavior_metrics()
    plot_model_predictions()
    #plot_model_predictions_rescorla()
    #plot_model_predictions_all(metric='retro_bias')

    #plot_goal_valuations()
    #plot_retro_value_figures()
    #plot_switching_patterns()
    #plot_performance_figures()
    #plot_instructions_figure()
    #plot_model_fits()