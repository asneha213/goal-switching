import numpy as np


def draw_significant_bar_plots(ax, data_points, model_names, \
                               fit_values, std_values, p_values, \
                               ylabel, title, bar_width=0.2, bar_gap=0.15, colors=None):


    # Define color scheme
    if not colors:
        colors = ['#add8e6', '#FAFAD2', '#90ee90', '#f08080']

    bars = ax.barh(model_names, fit_values,  color=colors, edgecolor='black', alpha=0.6, height=0.5)
    # # Draw data points on top of the bars
    # for i, (category, points) in enumerate(data_points.items()):
    #     xs = np.full_like(points, i) + np.random.uniform(-bar_width*0.6, bar_width * 0.6, size=len(points))
    #     # Scatter plot with transparency and light color
    #     ax.scatter(points, xs, alpha=0.4, color='black', edgecolor='white', s=5, linewidth=0.5)

    # Add grid lines for better readability
    ax.xaxis.grid(True, linestyle='--', which='both', color='gray', alpha=0.7)

    # Labeling
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.set_xlim(min(fit_values) * 0.95, max(fit_values) * 1.04)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def get_significance_str(p_value):
        num_asterisks = int(-np.log10(p_value))
        if num_asterisks < 1:
            significance_str = 'ns'
        elif num_asterisks > 3:
            num_asterisks = 3
            significance_str = '*' * num_asterisks
        else:
            significance_str = '*' * num_asterisks
        return significance_str

    sig_strs = [get_significance_str(p) for p in p_values]


    def add_significance_marker(ax, bar1, bar2, width, text, x_offset=1, bar_height=0.2):
        y1, y2 = bar1.get_y() + bar1.get_height() / 2, bar2.get_y() + bar2.get_height() / 2
        x, w = max(bar1.get_width(), bar2.get_width()) + x_offset, width
        ax.plot([x, x + w, x + w, x], [y1, y1, y2, y2], lw=1.5, color='black')
        ax.text(x + w, (y1 + y2) / 2, text, ha='left', va='center', color='black')


    # Add significance markers
    add_significance_marker(ax, bars[0], bars[1], width=2, text=sig_strs[0])
    add_significance_marker(ax, bars[1], bars[2], width=2, text=sig_strs[1])
    add_significance_marker(ax, bars[2], bars[3], width=2, text=sig_strs[2])
    add_significance_marker(ax, bars[0], bars[3], width=2, text=sig_strs[3])


    # # Add significance markers and annotations
    # for i in range(len(model_names) - 1):
    #     if p_values[i].to_numpy()[0] < 0.05:
    #         diff = fit_values[i + 1] - fit_values[i]
    #         num_asterisks = int(-np.log10(p_values[i]))  # Calculate the number of asterisks based on the p-value
    #         significance_str = '*' * num_asterisks
    #
    #         ax.plot([x_pos[i], x_pos[i+1]], [fit_values[i] * 1.1 ]*2, \
    #                 color='black')
    #         ax.text(x_pos[i] + (bar_width + bar_gap)/2, fit_values[i] * 1.13, \
    #                 f'{significance_str}', ha='center', va='bottom', fontsize=11)
    #
    #         # ax.plot([i, i + 1], [fit_values[i + 1] + fit_values[i + 1]/2.5 + (10 - (fit_values[i + 1]/10)*(i+1)) + 0.01 ] * 2, color='black')
    #         # ax.text(i + 0.5, fit_values[i + 1] + fit_values[i + 1]/2.5 + ((10 - (fit_values[i + 1]/10)*(i+1)) + 0.02) , f'{significance_str}', ha='center', va='bottom',
    #         #          fontsize=11)
    #
    # num_asterisks = int(-np.log10(p_values[3]))  # Calculate the number of asterisks based on the p-value
    # significance_str = '*' * num_asterisks
    # ax.plot([x_pos[0], x_pos[3]], [fit_values[0] * 1.3] * 2, color='black')
    # ax.text(x_pos[0] + (bar_width + bar_gap)*1.5, fit_values[0] * 1.33, f'{significance_str}', ha='center', va='bottom',
    #         fontsize=11)

    # ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    # ax.set_ylim(0, max(fit_values) * 1.6)
    #
    # ax.tick_params(axis='x', labelsize=9)# Customize the y-axis limits as needed

    # if title:
    #     # Add title boxes to each subplot
    #     title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)
    #
    #     ax.annotate(title, xy=(0.5, 1.05), xycoords='axes fraction',
    #                 fontsize=10, ha='center', bbox=title_box_props)
