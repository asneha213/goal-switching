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
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=13)

    ax.set_xlim(min(fit_values) * 0.96, max(fit_values) * 1.04)

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


    def add_significance_marker(ax, bar1, bar2, text, x_offset=0.5, bar_height=0.2):
        if text == 'ns':
            return
        y1, y2 = bar1.get_y() + bar1.get_height() / 2, bar2.get_y() + bar2.get_height() / 2
        x = max(bar1.get_width(), bar2.get_width()) + x_offset
        w = x / 250
        ax.plot([x, x + w, x + w, x], [y1, y1, y2, y2], lw=1.5, color='black')
        ax.text(x + w, (y1 + y2) / 2, text, ha='left', va='center', color='black', fontweight='bold')


    # Add significance markers
    add_significance_marker(ax, bars[0], bars[1], text=sig_strs[0])
    add_significance_marker(ax, bars[1], bars[2], text=sig_strs[1])
    add_significance_marker(ax, bars[2], bars[3], text=sig_strs[2])
    add_significance_marker(ax, bars[0], bars[3], text=sig_strs[3])

