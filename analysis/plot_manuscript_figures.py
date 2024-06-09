from primary_arbitration import plot_goal_valuation
from retrospective_bias import plot_retrospective_bias
from switching_patterns import plot_stay_switch_rates

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


if __name__ == '__main__':
    # Sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(-x)

    # Create a figure
    fig = plt.figure(figsize=(12, 24))

    # Create GridSpec
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.1, 1, 0.1, 1])

    # Set width ratios for each row
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=[0.1, 0.5, 0.3, 0.1])
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], width_ratios=[0.3, 0.2, 0.25, 0.2])
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4], width_ratios=[0.1, 0.5, 0.4, 0.1])

    # Top row plots
    ax1 = fig.add_subplot(gs_row1[1])
    ax2 = fig.add_subplot(gs_row1[2])
    plot_goal_valuation(axs=[ax1, ax2], show=False)


    # Middle row plots
    ax3 = fig.add_subplot(gs_row2[0])
    ax4 = fig.add_subplot(gs_row2[1])
    ax5 = fig.add_subplot(gs_row2[2])
    ax6 = fig.add_subplot(gs_row2[3])
    plot_retrospective_bias(model_name=None, axs=[ax3, ax4, ax5, ax6], print=False)
    ax3.set_title('Experiment 1', fontsize=11)
    ax5.set_title('Experiment 2', fontsize=11)

    # Bottom full-width plot
    ax7 = fig.add_subplot(gs_row3[1])
    ax8 = fig.add_subplot(gs_row3[2])
    plot_stay_switch_rates(axs=[ax7, ax8], model_name=None, experiments=[1, 2], show=True)
    ax7.set_title('Experiment 1', fontsize=11)
    ax8.set_title('Experiment 2', fontsize=11)

    # Add labels
    fig.text(0.08, 0.85, 'A', fontsize=16, fontweight='bold', va='center')
    fig.text(0.08, 0.5, 'B', fontsize=16, fontweight='bold', va='center')
    fig.text(0.08, 0.15, 'C', fontsize=16, fontweight='bold', va='center')

    # Adjust layout to make space for the labels
    fig.subplots_adjust(left=0.15, hspace=0.7)

    # Add titles for each row
    fig.text(0.5, 0.95, 'Goal selection probabilities', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.63, 'Proportion of retrospectively-biased choice', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.32, 'Probability of switching away from current choice', ha='center', fontsize=12, fontweight='bold')

    # Show plot
    plt.show()
