from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
import pingouin as pg

import matplotlib

# Setting font properties globally
# matplotlib.rcParams['font.weight'] = 'bold'
# matplotlib.rcParams['font.family'] = 'sans-serif'

def plot_comparative_bar_plot(ax, data, mean_values, std_dev_values, \
                              conditions, models, title, \
                              ylabel, ylim, x_pos=None, \
                              colors=None, bar_width=0.2,\
                              draw_data=True, legend=True,\
                              label_font=12, tick_font=14, \
                            legend_font=11, title_font=12):

    # Sample data

    # if test:
    #     conditions = ['Condition 1', 'Condition 2', 'Condition 3']
    #     models = ['Model A', 'Model B', 'Model C', 'Model D']
    #
    #     # Mean accuracy values for each model and condition
    #     mean_values = np.array([[0.85, 0.90, 0.92],
    #                                 [0.80, 0.88, 0.90],
    #                                 [0.82, 0.85, 0.91],
    #                             [0.82, 0.89, 0.91]])
    #
    #     # Standard deviation values for each model and condition
    #     std_dev_values = np.array([[0.03, 0.02, 0.01],
    #                                [0.02, 0.03, 0.04],
    #                                [0.01, 0.02, 0.03],
    #                                [0.01, 0.06, 0.03]
    #                                ])
    #
    #
    #     title = "Comparison of Models across Conditions"
    #     ylabel = "Accuracy"
    #     ylim = [0.75, 1.0]

    # Number of conditions and models
    num_conditions = len(conditions)
    num_models = len(models)

    # Set width of each bar
    #bar_width = 0.2

    # Generate positions of the bars on the x-axis
    if not x_pos:
        x_pos = np.arange(num_conditions)
    else:
        x_pos = np.array(x_pos)

    # Define color scheme
    if not colors:
        #colors = ['#1f77b4', "red", '#2ca02c', '#808080']
        colors = ['#90ee90', '#add8e6', '#f08080','#FAFAD2', ]


    # Loop through each model
    for i in range(num_models):
        # Calculate the position for the bar
        pos = x_pos + (i * bar_width)

        if draw_data:
            for k in range(len(data[i])):
                ax.plot(pos + np.random.normal(0, 0.01, len(pos)) , \
                        data[i][k] + np.random.normal(0, np.min(data[i][k])/100, len(pos)), \
                        color=colors[i], marker='o', linestyle='None', markersize=2)

        # Bar plot with error bars
        ax.bar(pos, mean_values[i], bar_width, yerr=std_dev_values[i], capsize=4, label=models[i], color=colors[i],
               edgecolor='black', alpha=0.5)

    # Set x-axis labels and tick positions
    ax.set_xticks(x_pos + (bar_width * (num_models - 1)) / 2)
    ax.set_xticklabels(conditions, fontsize=tick_font)

    # Set y-axis label and title
    ax.set_ylabel(ylabel, fontsize=label_font)
    #ax.set_title(title)

    # Add a legend
    if legend:
        ax.legend(fontsize=legend_font, loc='upper left')

    # Adjust y-axis limits
    ax.set_ylim(ylim)

    # Increase font size for xticks and yticks

    ax.tick_params(axis='both', which='major', labelsize=11)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize tick parameters
    ax.tick_params(axis='y', which='both', length=0)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # # Add title boxes to each subplot
    # title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)
    #
    # ax.annotate(title, xy=(0.5, 1.05), xycoords='axes fraction',
    #                 fontsize=title_font, ha='center', bbox=title_box_props)
    ax.set_title(title, fontsize=title_font)




def plot_correlation_param_measure(ax, x, y, p_val, xlabel, ylabel, title):
    # Generate sample data
    #sns.set(style="whitegrid")

    # # Generate sample data
    # np.random.seed(0)
    # x = np.random.rand(50)
    # y = 2 * x + 1 + np.random.randn(50) * 0.2  # Simulated linear relationship with some noise

    # Calculate the correlation coefficient
    corr = pg.corr(x, y)['r'].to_numpy()[0]

    print(pg.corr(x, y)['p-val'])
    # p_val = pg.corr(x, y)['p-val'].to_numpy()[0]
    # p_val = "{:.1e}".format(p_val)

    # Create a scatter plot
    #plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, color='blue', ax=ax)

    # Add a correlation line
    sns.regplot(x=x, y=y, scatter=False, color='red', label=f'r={corr:.2f}, p={p_val}', ax=ax)

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

    ax.legend(fontsize=12)

    # Add title boxes to each subplot
    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    ax.annotate(title, xy=(0.5, 1.05), xycoords='axes fraction',
                fontsize=11, ha='center', bbox=title_box_props)


def plot_multiple_histograms(ax, measures, labels, xlabel, ylabel, title=None, legend=True):
    colors = ['#1f77b4', "red", '#2ca02c', '#808080']
    colors = ['#90ee90', '#add8e6', '#f08080', '#FAFAD2', ]
    for i in range(len(measures)):
        measure = measures[i]
        label = labels[i]
        sns.histplot(measure, kde=True, stat="density", linewidth=0, alpha=0.4, label=label, color=colors[i], ax=ax)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if legend:
        ax.legend(fontsize=9)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_correlation_param_measure(ax)
