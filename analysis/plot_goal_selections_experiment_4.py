import matplotlib.pyplot as plt

from behavior import *
from reports import *
from models import *
from matplotlib.gridspec import GridSpec
import pickle

def get_measure_experiment_optimal(experiment, model_name=None, cache=True, optimal=False):
    if cache:
        try:
            with open('figures_cache/optimal_goals_' + str(experiment) + '_' + str(model_name) + "_" + str(optimal) + '.pkl', 'rb') as f:
                sub_measures_0, sub_measures_1 = pickle.load(f)
            return np.array(sub_measures_0), np.array(sub_measures_1)
        except:
            pass
    sub_measures_0 = []
    sub_measures_1 = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        if model_name is not None:
            if not optimal:
                with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
                params = model_fits['params']
            else:
                file_name = "results/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
                with open(file_name, "rb") as f:
                    params = pickle.load(f)


            runmodel = RunModel(experiment, model_name, params)
            model_res = runmodel.get_model_res()
            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
        else:
            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)
        selects, counts = subject_measures.get_optimal_goal_condition()
        sub_measures_0.append(selects[0] / counts[0][:, None])
        sub_measures_1.append(selects[1] / counts[1][:, None])

    with open('figures_cache/optimal_goals_' + str(experiment) + '_' + str(model_name) + "_" + str(optimal) + '.pkl', 'wb') as f:
        pickle.dump([sub_measures_0, sub_measures_1], f)

    return np.array(sub_measures_0), np.array(sub_measures_1)



def plot_optimal_goals(ax=None, model_name=None, compare=False, optimal=False, cache=True):

    # Load data
    sub_measures_0, sub_measures_1 = get_measure_experiment_optimal(experiment=4, model_name=model_name,\
                                                                    cache=cache, optimal=optimal)

    optimal_goals_0 = np.mean(sub_measures_0, axis=0)
    optimal_goals_1 = np.mean(sub_measures_1, axis=0)

    optimal_goals_std_0 = np.std(sub_measures_0, axis=0)/ np.sqrt(len(sub_measures_0))
    optimal_goals_std_1 = np.std(sub_measures_1, axis=0)/ np.sqrt(len(sub_measures_1))


    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    data = [optimal_goals_0[0], optimal_goals_0[1], optimal_goals_0[2],
            optimal_goals_1[0], optimal_goals_1[ 1], optimal_goals_1[2]]

    # mean_values = np.array([optimal_goals_0[:, 0], optimal_goals_0[:, 1], optimal_goals_0[:, 2],
    #                  optimal_goals_1[:, 0], optimal_goals_1[:, 1], optimal_goals_1[:, 2]])
    #
    # std_values = np.array([optimal_goals_std_0[:, 0], optimal_goals_std_0[:, 1], optimal_goals_std_0[:, 2],
    #                 optimal_goals_std_1[:, 0], optimal_goals_std_1[:, 1], optimal_goals_std_1[:, 2]])


    mean_values = np.array([optimal_goals_0[ 0], optimal_goals_0[1], optimal_goals_0[2],
                            optimal_goals_1[0], optimal_goals_1[1], optimal_goals_1[2]])


    std_values = np.array([optimal_goals_std_0[0], optimal_goals_std_0[1], optimal_goals_std_0[2],
                            optimal_goals_std_1[0], optimal_goals_std_1[1], optimal_goals_std_1[2]])

    conditions = ['Cat', 'Hat', 'Car', 'Cat', 'Hat', 'Car']
    # conditions = ['SH', 'H0', 'BR']
    goals = ['Cat', 'Hat', 'Car']
    if model_name is None or optimal == True:
        ylabel = "Probability of selection"
    else:
        ylabel = None
    ylim = [0, 0.8]
    if compare:
        if model_name == "momentum":
            title = "TD-Momentum"
        elif model_name == "prospective":
            title = "Prospective (fit to participants)"
        elif model_name == "hybrid":
            title = "Hybrid"
        elif model_name is None:
            title = "Participants"
    else:
        title = "Participants"

    if optimal:
        if model_name == "prospective":
            title = "Optimal prospective agent"
        elif model_name == "momentum":
            title = "Optimal TD-momentum agent"

    x_pos = [0, 1, 2, 3.5, 4.5, 5.5]

    plot_comparative_bar_plot(ax, data, mean_values.T, std_values.T, conditions,
                              goals, title, ylabel, ylim,
                              bar_width=0.25, legend=False, x_pos=x_pos)

    if model_name is None or optimal==True:
        ax.legend(fontsize=13, loc='upper left', title="Choosing")

    ax.annotate("H disp", xy=(0.25, -0.16),
                    xycoords='axes fraction',
                    fontsize=13, fontweight='bold', ha='center')
    ax.annotate("L disp", xy=(0.75, -0.16),
                    xycoords='axes fraction',
                    fontsize=13, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig("figures/optimal_goals_" + str(model_name) + str(title) + str(optimal)+ ".png")
    plt.show()


if __name__ == "__main__":
    model_name = None
    model_name = "momentum"
    #model_name = "prospective"
    model_name = "hybrid"
    plot_optimal_goals(model_name=model_name, cache=False, optimal=False, compare=True)
