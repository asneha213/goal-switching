import numpy as np


def draw_significant_bar_plots(ax, model_names, \
                               fit_values, std_values, p_values, \
                               ylabel, title, bar_width=0.2, bar_gap=0.15, colors=None):


    # generate x pos
    x_pos = np.arange(len(model_names)) * (bar_width + bar_gap)

    # Define color scheme
    if not colors:
        colors = ['#7f8c8d', '#95a5a6', '#bdc3c7', '#d0d3d4']

    # Create a bar plot
    ax.bar(x_pos, fit_values, bar_width, yerr=std_values,\
           capsize=4, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')


    # Add significance markers and annotations
    for i in range(len(model_names) - 1):
        if p_values[i].to_numpy()[0] < 0.05:
            diff = fit_values[i + 1] - fit_values[i]
            num_asterisks = int(-np.log10(p_values[i]))  # Calculate the number of asterisks based on the p-value
            significance_str = '*' * num_asterisks

            ax.plot([x_pos[i], x_pos[i+1]], [fit_values[i] * 1.1 ]*2, \
                    color='black')
            ax.text(x_pos[i] + (bar_width + bar_gap)/2, fit_values[i] * 1.13, \
                    f'{significance_str}', ha='center', va='bottom', fontsize=11)

            # ax.plot([i, i + 1], [fit_values[i + 1] + fit_values[i + 1]/2.5 + (10 - (fit_values[i + 1]/10)*(i+1)) + 0.01 ] * 2, color='black')
            # ax.text(i + 0.5, fit_values[i + 1] + fit_values[i + 1]/2.5 + ((10 - (fit_values[i + 1]/10)*(i+1)) + 0.02) , f'{significance_str}', ha='center', va='bottom',
            #          fontsize=11)

    num_asterisks = int(-np.log10(p_values[3]))  # Calculate the number of asterisks based on the p-value
    significance_str = '*' * num_asterisks
    ax.plot([x_pos[0], x_pos[3]], [fit_values[0] * 1.3] * 2, color='black')
    ax.text(x_pos[0] + (bar_width + bar_gap)*1.5, fit_values[0] * 1.33, f'{significance_str}', ha='center', va='bottom',
            fontsize=11)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(fit_values) * 1.6)

    ax.tick_params(axis='x', labelsize=9)# Customize the y-axis limits as needed

    if title:
        # Add title boxes to each subplot
        title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

        ax.annotate(title, xy=(0.5, 1.05), xycoords='axes fraction',
                    fontsize=10, ha='center', bbox=title_box_props)
