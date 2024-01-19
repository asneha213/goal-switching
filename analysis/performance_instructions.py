from behavior import *
from reports import *

from optimize_model import ModelOptimizer
from matplotlib import pyplot as plt

import pingouin as pg


def plot_performance_instructions():
    pros_performance_1 = ModelOptimizer(experiment=1, model_name="prospective").simulate_params()
    sub_performance_1 = get_measure_experiment(experiment=1, measure_name="num_goals", mode="condition")
    sub_performance_instr = get_measure_experiment(experiment="instr_1", measure_name="num_goals", mode="condition")

    # Compare overall subject performance with and without instructions
    tt = pg.ttest(np.mean(sub_performance_1, axis=1), np.mean(sub_performance_instr, axis=1))
    print(f"t({tt['dof']}) = {tt['T']}, p = {tt['p-val']}, d = {tt['cohen-d']}")

    fig, axs = plt.subplots(1, 1, figsize=(9, 5))

    data_ = [pros_performance_1, sub_performance_1, sub_performance_instr]
    conditions = ["80-20", "70-30", "60-40"]
    models = ["Prospective", "No Instructions", "Instructions"]

    mean_values = [ np.mean(pros_performance_1, axis=0), np.mean(sub_performance_1, axis=0), np.mean(sub_performance_instr, axis=0)]
    std_values = [ np.std(pros_performance_1, axis=0), np.std(sub_performance_1, axis=0), np.std(sub_performance_instr, axis=0)] / np.sqrt(len(sub_performance_1))
    ylim = [0, 25]

    plot_comparative_bar_plot(axs, data_, mean_values, std_values,  conditions, models,title=None,
                              ylim= ylim, ylabel="Number of suits", legend=True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_performance_instructions()