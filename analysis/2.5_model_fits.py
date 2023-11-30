import pickle

from behavior import *
from reports import *

import matplotlib.pyplot as plt


def get_model_BIC(experiment, model_name, AIC=False):
    bics = []
    lls = []

    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)

    for subject_num in range(num_subjects):

        subject_id = subject_names[int(subject_num)]
    

        try:
            with open('results/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
        except:
            continue

        fits = model_fits['fits']
        params = list(model_fits['params'].values())

        bic = 2 * fits + len(params) * np.log(num_samples)

        #print(subject_id, model_name, bic)
        
        # if subject_id == "5d4434c68225f9001642fcd3":
        #     print(bic)

        if not AIC:
            if model_name != "td_persistence":
                bic = 2 * fits + len(params) * np.log(num_samples)
            else:
                bic = 2 * fits + (len(params) - 1) * np.log(num_samples)
        else:
            if model_name != "td_persistence":
                bic = 2 * fits + 2 * len(params)
            else:
                bic = 2 * fits + 2 * (len(params) - 1)
        bics.append(bic)
        lls.append(fits)

    return np.array(bics), lls


def plot_bic_diff(ax, bics0, bics1, titles, ylim, title=""):

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
    ax.set_xlabel('subjects')
    ax.set_ylabel('BIC difference')
    #ax.set_title('Colored Rectangular Bars of a Sorted Array')

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylim(ylim)

    if title != "":
        # Add title boxes to each subplot
        title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

        ax.annotate(title, xy=(0.5, 1.15), xycoords='axes fraction',
                    fontsize=10, ha='center', bbox=title_box_props)


def plot_bics_diffs_two_experiments(bics_exp1_m1, bics_exp1_m2, \
                                    bics_exp2_m1, bics_exp2_m2, titles, ylims):
    from matplotlib.gridspec import GridSpec



    fig = plt.figure(figsize=(8, 4))

    gs = GridSpec(1, 2, width_ratios=[1.2, 1])

    axs = [plt.subplot(gs[0,0]), plt.subplot(gs[0, 1])]

    plot_bic_diff(axs[0], bics_exp1_m1, bics_exp1_m2, titles, ylims[0], title="Experiment 1")

    plot_bic_diff(axs[1], bics_exp2_m1, bics_exp2_m2, titles, ylims[1], title="Experiment 2")

    plt.tight_layout()
    plt.show()


def plot_model_comparisions_individual():
    experiment = 1

    AIC = False

    bics_td_1, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_pros_1, ll_pros_1 = get_model_BIC(experiment, 'prospective', AIC)
    bics_nt_1, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    experiment = 2

    bics_td_2, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'prospective', AIC)
    bics_nt_2, _ = get_model_BIC(experiment, 'td_persistence', AIC)


    plot_bics_diffs_two_experiments(bics_pros_1, bics_td_1, bics_pros_2, bics_td_2, \
                                    ["Prospective better", "TD Momentum better"], \
                                    [[-150, 300], [-50, 80]])

    plot_bics_diffs_two_experiments(bics_nt_1, bics_td_1, bics_nt_2, bics_td_2, \
                                    ["TD Persistence better", "TD Momentum better"], \
                                    [[-150, 300], [-120, 275]])


def plot_model_fits():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    experiment = 1

    AIC = True

    bics_td_1, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_pros_1, ll_pros_1 = get_model_BIC(experiment, 'prospective', AIC)
    bics_h_1, ll_h_1 = get_model_BIC(experiment, 'hybrid', AIC)
    bics_nt_1, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    experiment = 2

    bics_td_2, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'prospective', AIC)
    bics_h_2, ll_h_2 = get_model_BIC(experiment, 'hybrid', AIC)
    bics_nt_2, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    bics_prospective = [bics_pros_1, bics_pros_2]
    bics_hybrid = [bics_h_1, bics_h_2]
    bics_momentum = [bics_td_1, bics_td_2]
    bics_persistence = [bics_nt_1, bics_nt_2]


    fits_exp_1 = [np.mean(bics_hybrid[0]), np.mean(bics_prospective[0]), np.mean(bics_persistence[0]), np.mean(bics_momentum[0])]
    fits_exp_2 = [np.mean(bics_persistence[1]), np.mean(bics_prospective[1]), np.mean(bics_hybrid[1]), np.mean(bics_momentum[1])]

    std_fits_exp_1 = [np.std(bics_hybrid[0]), np.std(bics_prospective[0]), np.std(bics_persistence[0]), np.std(bics_momentum[0])] / np.sqrt(len(bics_hybrid[0]))
    std_fits_exp_2 = [np.std(bics_persistence[1]), np.std(bics_prospective[1]), np.std(bics_hybrid[1]), np.std(bics_momentum[1])] / np.sqrt(len(bics_persistence[1]))


    #fits_exp_1 = [np.sum(bics_hybrid[0]), np.sum(bics_prospective[0]), np.sum(bics_persistence[0]), np.sum(bics_momentum[0])]
    #fits_exp_2 = [np.sum(bics_persistence[1]), np.sum(bics_prospective[1]), np.sum(bics_hybrid[1]), np.sum(bics_momentum[1])]

    data1 = None
    data2 = None

    pvals1 = [pg.ttest(bics_hybrid[0], bics_prospective[0], paired=True)['p-val'],\
              pg.ttest(bics_prospective[0], bics_persistence[0], paired=True)['p-val'], \
              pg.ttest(bics_persistence[0], bics_momentum[0], paired=True)['p-val'], \
              pg.ttest(bics_hybrid[0], bics_momentum[0], paired=True)['p-val']]
    pvals2 = [pg.ttest(bics_persistence[1], bics_prospective[1], paired=True)['p-val'],\
              pg.ttest(bics_prospective[1], bics_hybrid[1], paired=True)['p-val'], \
              pg.ttest(bics_hybrid[1], bics_momentum[1], paired=True)['p-val'], \
              pg.ttest(bics_persistence[1], bics_momentum[1], paired=True)['p-val']]

    if AIC:
        ylabel = "AIC scores"
    else:
        ylabel = "BIC scores"

    draw_significant_bar_plots(axs[0], data1, ['Hybrid', 'Prospective', 'TD-Persistence', 'TD-Momentum'],
                               fits_exp_1, std_fits_exp_1, pvals1, ylabel, "Experiment 1")

    draw_significant_bar_plots(axs[1], data2, ['TD-Persistence', 'Prospective', 'Hybrid', 'TD-Momentum'],
                               fits_exp_2, std_fits_exp_2, pvals2, ylabel, "Experiment 2")


    plt.tight_layout()
    plt.show()


def get_cv_scores(experiment, model_name):
    cvs = []

    num_subjects, subject_names, num_samples = get_experiment_logistics(experiment)

    for subject_num in range(num_subjects):

        subject_id = subject_names[int(subject_num)]


        try:
            with open('results/' + model_name + "_" + str(experiment) + "/cv_" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)

        except:
            continue

        cvs.append(np.mean(model_fits))

    return np.array(cvs)


def plot_cv_scores():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    experiment = 2

    td_cvs = get_cv_scores(experiment=experiment, model_name="momentum")
    h_cvs = get_cv_scores(experiment=experiment, model_name="hybrid")
    p_cvs = get_cv_scores(experiment=experiment, model_name="td_persistence")
    pros_cvs = get_cv_scores(experiment=experiment, model_name="prospective")

    p_val_1 = pg.ttest(pros_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], h_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], paired=True)['p-val']
    p_val_2 = pg.ttest(pros_cvs[~np.isinf(pros_cvs)], p_cvs[~np.isinf(pros_cvs)], paired=True)['p-val']
    p_val_3 = pg.ttest(td_cvs[~np.isinf(p_cvs)], p_cvs[~np.isinf(p_cvs)], paired=True)['p-val']
    p_val_4 = pg.ttest(td_cvs[~np.isinf(td_cvs)], h_cvs[~np.isinf(h_cvs)], paired=True)['p-val']

    pvals = [p_val_1, p_val_2, p_val_3, p_val_4]

    mean_scores = [np.mean(h_cvs[~np.isinf(h_cvs)]), np.mean(pros_cvs[~np.isinf(pros_cvs)]),
                   np.mean(p_cvs[~np.isinf(h_cvs)]), np.mean(td_cvs[~np.isinf(h_cvs)])]
    std_scores = [np.std(h_cvs[~np.isinf(h_cvs)]), np.std(pros_cvs[~np.isinf(pros_cvs)]), np.std(p_cvs[~np.isinf(h_cvs)]),
                  np.std(td_cvs[~np.isinf(h_cvs)])] / np.sqrt(len(h_cvs))

    data = None

    draw_significant_bar_plots(axs[0], data, ['Hybrid', 'Prospective', 'TD-Persistence', 'TD-Momentum'],
                                 mean_scores, std_scores, pvals, "CV scores", "Experiment 1")

    experiment = 2

    td_cvs = get_cv_scores(experiment=experiment, model_name="momentum")
    h_cvs = get_cv_scores(experiment=experiment, model_name="hybrid")
    p_cvs = get_cv_scores(experiment=experiment, model_name="td_persistence")
    pros_cvs = get_cv_scores(experiment=experiment, model_name="prospective")

    p_val_2 = pg.ttest(pros_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], h_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], paired=True)['p-val']
    p_val_1 = pg.ttest(pros_cvs[~np.isinf(pros_cvs)], p_cvs[~np.isinf(pros_cvs)], paired=True)['p-val']
    p_val_3 = pg.ttest(td_cvs[~np.isinf(p_cvs)], h_cvs[~np.isinf(h_cvs)], paired=True)['p-val']
    p_val_4 = pg.ttest(td_cvs[~np.isinf(td_cvs)], p_cvs[~np.isinf(p_cvs)], paired=True)['p-val']

    pvals = [p_val_1, p_val_2, p_val_3, p_val_4]


    mean_scores = [np.mean(p_cvs[~np.isinf(p_cvs)]), np.mean(pros_cvs[~np.isinf(pros_cvs)]), np.mean(h_cvs[~np.isinf(h_cvs)]), np.mean(td_cvs[~np.isinf(h_cvs)])]
    std_scores = [np.std(p_cvs[~np.isinf(p_cvs)]), np.std(pros_cvs[~np.isinf(pros_cvs)]), np.std(h_cvs[~np.isinf(h_cvs)]), np.std(td_cvs[~np.isinf(h_cvs)])] / np.sqrt(len(h_cvs))

    data = None

    draw_significant_bar_plots(axs[1], data, ['TD-Persistence', 'Prospective', 'Hybrid', 'TD-Momentum'],
                               mean_scores, std_scores, pvals, "CV scores", "Experiment 1")

    plt.tight_layout()
    plt.show()


def plot_model_fits_prospective():
    fig = plt.figure(figsize=(11, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[0.1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])]
    experiment = 2

    AIC = True

    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'prospective', AIC)

    bics_dl_2, _ = get_model_BIC(experiment, 'prospective_dl', AIC)
    bics_m_2, _ = get_model_BIC(experiment, 'prospective_momentum', AIC)
    bics_dlm_2, _ = get_model_BIC(experiment, 'prospective_dl_momentum', AIC)

    fits_exp_2 = [np.mean(bics_dlm_2), np.mean(bics_m_2), np.mean(bics_dl_2), np.mean(bics_pros_2)]
    std_fits_exp_2 = [np.std(bics_dlm_2), np.std(bics_m_2), np.std(bics_dl_2), np.std(bics_pros_2)] / np.sqrt(len(bics_dlm_2))


    data2 = None

    if AIC:
        ylabel = "AIC scores"
    else:
        ylabel = "BIC scores"

    pvals2 = [pg.ttest(bics_dlm_2, bics_m_2, paired=True)['p-val'], pg.ttest(bics_m_2, bics_dl_2, paired=True)['p-val'], pg.ttest(bics_dl_2, bics_pros_2, paired=True)['p-val'], pg.ttest(bics_dlm_2, bics_pros_2, paired=True)['p-val']]

    draw_significant_bar_plots(axs[0], data2, ['Pros+DL+M', 'Pros+Mo', 'Pros+DL', 'Prospective'],
                                 fits_exp_2, std_fits_exp_2, pvals2, ylabel, None)

    AIC = False

    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'hierarchy', AIC)

    bics_dl_2, _ = get_model_BIC(experiment, 'dl_hierarchy', AIC)
    bics_m_2, _ = get_model_BIC(experiment, 'momentum_hierarchy', AIC)
    bics_dlm_2, _ = get_model_BIC(experiment, 'dl_momentum_hierarchy', AIC)

    fits_exp_2 = [np.sum(bics_dlm_2), np.sum(bics_m_2), np.sum(bics_dl_2), np.sum(bics_pros_2)]

    data2 = None

    if AIC:
        ylabel = "AIC scores"
    else:
        ylabel = "BIC scores"

    pvals2 = [pg.ttest(bics_dlm_2, bics_m_2, paired=True)['p-val'], pg.ttest(bics_m_2, bics_dl_2, paired=True)['p-val'],
              pg.ttest(bics_dl_2, bics_pros_2, paired=True)['p-val'],
              pg.ttest(bics_dlm_2, bics_pros_2, paired=True)['p-val']]

    draw_significant_bar_plots(axs[1], data2, ['Pros+DL+M', 'Pros+Mo', 'Pros+DL', 'Prospective'],
                               fits_exp_2, None, pvals2, ylabel, None)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)
    axs[0].annotate("Experiment 2", xy=(-0.2, 0.5), rotation=90, xycoords='axes fraction',
                    fontsize=11, ha='center', va='center', bbox=title_box_props)

    plt.tight_layout()
    plt.show()


def plot_cv_score(ax, experiment):

    td_cvs = get_cv_scores(experiment=experiment, model_name="momentum")
    h_cvs = get_cv_scores(experiment=experiment, model_name="hybrid")
    p_cvs = get_cv_scores(experiment=experiment, model_name="td_persistence")
    pros_cvs = get_cv_scores(experiment=experiment, model_name="prospective")

    if experiment == 1:

        p_val_1 = pg.ttest(pros_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))],
                           h_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], paired=True)['p-val']
        p_val_2 = pg.ttest(pros_cvs[~np.isinf(pros_cvs)], p_cvs[~np.isinf(pros_cvs)], paired=True)['p-val']
        p_val_3 = pg.ttest(td_cvs[~np.isinf(p_cvs)], p_cvs[~np.isinf(p_cvs)], paired=True)['p-val']
        p_val_4 = pg.ttest(td_cvs[~np.isinf(td_cvs)], h_cvs[~np.isinf(h_cvs)], paired=True)['p-val']


        mean_scores = [np.mean(h_cvs[~np.isinf(h_cvs)]), np.mean(pros_cvs[~np.isinf(pros_cvs)]),
                       np.mean(p_cvs[~np.isinf(h_cvs)]), np.mean(td_cvs[~np.isinf(h_cvs)])]
        std_scores = [np.std(h_cvs[~np.isinf(h_cvs)]), np.std(pros_cvs[~np.isinf(pros_cvs)]),
                      np.std(p_cvs[~np.isinf(h_cvs)]),
                      np.std(td_cvs[~np.isinf(h_cvs)])] / np.sqrt(len(h_cvs))
        models = ['Hybrid', 'Prospective', 'TD-Persistence', 'TD-Momentum']

    elif experiment == 2:
        p_val_2 = pg.ttest(pros_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))],
                           h_cvs[(~np.isinf(h_cvs)) & (~np.isinf(pros_cvs))], paired=True)['p-val']
        p_val_1 = pg.ttest(pros_cvs[~np.isinf(pros_cvs)], p_cvs[~np.isinf(pros_cvs)], paired=True)['p-val']
        p_val_3 = pg.ttest(td_cvs[~np.isinf(p_cvs)], h_cvs[~np.isinf(h_cvs)], paired=True)['p-val']
        p_val_4 = pg.ttest(td_cvs[~np.isinf(td_cvs)], p_cvs[~np.isinf(p_cvs)], paired=True)['p-val']



        mean_scores = [np.mean(p_cvs[~np.isinf(p_cvs)]), np.mean(pros_cvs[~np.isinf(pros_cvs)]),
                       np.mean(h_cvs[~np.isinf(h_cvs)]), np.mean(td_cvs[~np.isinf(h_cvs)])]
        std_scores = [np.std(p_cvs[~np.isinf(p_cvs)]), np.std(pros_cvs[~np.isinf(pros_cvs)]),
                      np.std(h_cvs[~np.isinf(h_cvs)]), np.std(td_cvs[~np.isinf(h_cvs)])] / np.sqrt(len(h_cvs))

        models = ['TD-Persistence', 'Prospective', 'Hybrid', 'TD-Momentum']

    pvals = [p_val_1, p_val_2, p_val_3, p_val_4]
    data = None

    draw_significant_bar_plots(ax, data, models,
                               mean_scores, std_scores, pvals, "CV scores", None)


def plot_aic_bic(ax, experiment, AIC=False):

    bics_momentum, _ = get_model_BIC(experiment, 'momentum', AIC)
    bics_prospective, ll_pros = get_model_BIC(experiment, 'prospective', AIC)
    bics_hybrid, ll_h = get_model_BIC(experiment, 'hybrid', AIC)
    bics_persistence, _ = get_model_BIC(experiment, 'td_persistence', AIC)

    if experiment == 1:
        fits_exp= [np.mean(bics_hybrid), np.mean(bics_prospective), np.mean(bics_persistence),
                  np.mean(bics_momentum)]
        std_fits_exp = [np.std(bics_hybrid), np.std(bics_prospective), np.std(bics_persistence),
                        np.std(bics_momentum)] / np.sqrt(len(bics_hybrid))
        pvals = [pg.ttest(bics_hybrid, bics_prospective, paired=True)['p-val'], \
                 pg.ttest(bics_prospective, bics_persistence, paired=True)['p-val'], \
                 pg.ttest(bics_persistence, bics_momentum, paired=True)['p-val'], \
                 pg.ttest(bics_hybrid, bics_momentum, paired=True)['p-val']]
        models = ['Hybrid', 'Prospective', 'TD-Persistence', 'TD-Momentum']

    elif experiment == 2:
        fits_exp = [np.mean(bics_persistence), np.mean(bics_prospective), np.mean(bics_hybrid),
                    np.mean(bics_momentum)]
        std_fits_exp = [np.std(bics_persistence), np.std(bics_prospective), np.std(bics_hybrid),
                        np.std(bics_momentum)] / np.sqrt(len(bics_hybrid))
        pvals = [pg.ttest(bics_persistence, bics_prospective, paired=True)['p-val'], \
                    pg.ttest(bics_prospective, bics_hybrid, paired=True)['p-val'], \
                    pg.ttest(bics_hybrid, bics_momentum, paired=True)['p-val'], \
                    pg.ttest(bics_persistence, bics_momentum, paired=True)['p-val']]
        models = ['TD-Persistence', 'Prospective', 'Hybrid', 'TD-Momentum']

    data = None


    if AIC:
        ylabel = "AIC scores"
    else:
        ylabel = "BIC scores"

    draw_significant_bar_plots(ax, data, models,
                               fits_exp, std_fits_exp, pvals, ylabel, None)


def plot_model_fits_experiment(experiment=2):
    fig = plt.figure(figsize=(15, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, width_ratios=[0.1, 1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3])]

    #fig, axs = plt.subplots(1, 3, figsize=(14, 3))

    plot_aic_bic(axs[0], experiment, AIC=True)
    plot_aic_bic(axs[1], experiment, AIC=False)
    plot_cv_score(axs[2], experiment)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("Experiment 1", xy=(-0.3, 0.5), rotation=90, xycoords='axes fraction',
                fontsize=11, ha='center', va='center', bbox=title_box_props)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #plot_model_fits_prospective()
    plot_model_fits()

    #plot_cv_scores()
    #plot_model_fits_experiment(experiment=2)