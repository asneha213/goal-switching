from behavior import *
from reports import *

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_model_BIC(experiment, model_name, AIC=False):
    """
    Get AIC/BIC scores for a given model and experiment
    :param experiment: int(1, 2, 4) or str("instr_1")
    :param model_name: str("momentum", "prospective", "hybrid", "td_persistence")
    """
    bics = []
    lls = []


    subject_names = get_experiment_subjects(experiment)
    details = get_experiment_trial_details(experiment)
    num_samples = details['num_samples']

    for subject_num in range(len(subject_names)):
        if subject_num == 44:
            continue

        subject_id = subject_names[int(subject_num)]
        try:
            with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                model_fits = pickle.load(f)
        except:
            print(model_name, subject_num)
            continue

        fits = model_fits['fits']
        params = list(model_fits['params'].values())

        num_params = len(params)

        if not AIC:
            bic = 2 * fits + num_params * np.log(num_samples)
        else:
            bic = 2 * fits + 2 * num_params

        bics.append(bic)
        lls.append(fits)

    return np.array(bics), np.array(lls)


def log_likelihood_ratio_test(ll_null, ll_hypothesis, df):
    from scipy.stats import chi2
    llr = 2 * (-ll_hypothesis - (-ll_null))
    p = [1 - chi2.cdf(x, df) for x in llr]
    return np.array(p)


def plot_aic_bic(ax, experiment, AIC=False):
    bics_momentum, ll_m = get_model_BIC(experiment, 'momentum', AIC)
    bics_prospective, ll_pros = get_model_BIC(experiment, 'prospective', AIC)
    bics_hybrid, ll_h = get_model_BIC(experiment, 'hybrid', AIC)
    bics_persistence, ll_p = get_model_BIC(experiment, 'td_persistence', AIC)

    data = {
        "momentum": bics_momentum,
        "prospective": bics_prospective,
        "hybrid": bics_hybrid,
        "td_persistence": bics_persistence
    }

    p = log_likelihood_ratio_test(ll_p, ll_m, 1)
    print("Proportion passing llr test:", len(np.where(p < 0.05)[0]) / len(ll_p))

    if experiment == "instr_1":
        fits_exp = [np.mean(bics_hybrid), np.mean(bics_prospective), np.mean(bics_persistence),
                    np.mean(bics_momentum)]
        std_fits_exp = [np.std(bics_hybrid), np.std(bics_prospective), np.std(bics_persistence),
                        np.std(bics_momentum)] / np.sqrt(len(bics_hybrid))
        pvals = [pg.ttest(bics_hybrid, bics_prospective, paired=True)['p-val'], \
                 pg.ttest(bics_prospective, bics_persistence, paired=True)['p-val'], \
                 pg.ttest(bics_persistence, bics_momentum, paired=True)['p-val'], \
                 pg.ttest(bics_hybrid, bics_momentum, paired=True)['p-val']]
        models = ['Hybrid', "Prospective", 'TD-Persistence', 'TD-Momentum']
        colors = ['#add8e6', '#FAFAD2', '#90ee90', '#f08080']

    elif experiment == 1 or experiment == 2 or experiment == 4:
        fits_exp = [np.mean(bics_persistence), np.mean(bics_hybrid), np.mean(bics_prospective),
                    np.mean(bics_momentum)]
        std_fits_exp = [np.std(bics_persistence), np.std(bics_hybrid), np.std(bics_prospective),
                        np.std(bics_momentum)] / np.sqrt(len(bics_hybrid))
        pvals = [pg.ttest(bics_persistence, bics_hybrid, paired=True)['p-val'], \
                    pg.ttest(bics_hybrid, bics_prospective, paired=True)['p-val'], \
                    pg.ttest(bics_prospective, bics_momentum, paired=True)['p-val'], \
                    pg.ttest(bics_persistence, bics_momentum, paired=True)['p-val']]


        models = ['TD-Persistence', 'Hybrid', 'Prospective', 'TD-Momentum']
        colors = ['#90ee90', '#add8e6', '#FAFAD2', '#f08080']


    print("Experiment:", experiment)
    print("T-test results:", pvals)


    if AIC:
        title = "AIC metric"
    else:
        title = "BIC metric"

    ylabel = None

    draw_significant_bar_plots(ax, data, models, fits_exp, std_fits_exp, \
                               pvals, ylabel, title, colors=colors)


def get_cv_scores(experiment, model_name):
    """
    Get cross-validated scores for a given model and experiment
    """
    cvs = []

    subject_names = get_experiment_subjects(experiment)

    for subject_num in range(len(subject_names)):

        subject_id = subject_names[int(subject_num)]
        cv_scores = []

        for seed in range(30):
            try:
                with open('results_latent/' + model_name + "_" + str(experiment) + "/cv_" + str(subject_id) + "_seed_" + str(seed) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
                    cv_scores.append(model_fits)
            except:
                continue
        cv_scores = np.array(cv_scores)

        cvs.append(np.nanmean(cv_scores))
        #print(cvs)

    return np.array(cvs)


def plot_cv_score(ax, experiment):
    td_cvs_uf = get_cv_scores(experiment=experiment, model_name="momentum")
    h_cvs_uf = get_cv_scores(experiment=experiment, model_name="hybrid")
    p_cvs_uf = get_cv_scores(experiment=experiment, model_name="td_persistence")
    pros_cvs_uf = get_cv_scores(experiment=experiment, model_name="prospective")

    td_cvs = td_cvs_uf[(~np.isnan(h_cvs_uf)) & (~np.isnan(pros_cvs_uf)) & ~np.isinf(h_cvs_uf) & ~np.isinf(pros_cvs_uf)]
    h_cvs = h_cvs_uf[(~np.isnan(h_cvs_uf)) & (~np.isnan(pros_cvs_uf)) & ~np.isinf(h_cvs_uf) & ~np.isinf(pros_cvs_uf)]
    p_cvs = p_cvs_uf[(~np.isnan(h_cvs_uf)) & (~np.isnan(pros_cvs_uf)) & ~np.isinf(h_cvs_uf) & ~np.isinf(pros_cvs_uf)]
    pros_cvs = pros_cvs_uf[(~np.isnan(h_cvs_uf)) & (~np.isnan(pros_cvs_uf)) & ~np.isinf(h_cvs_uf) & ~np.isinf(pros_cvs_uf)]

    data = {
        "momentum": td_cvs,
        "prospective": pros_cvs,
        "hybrid": h_cvs,
        "td_persistence": p_cvs
    }

    if (experiment == 1) or (experiment == 2):


        p_val_1 = pg.ttest(p_cvs, h_cvs, paired=True)['p-val']
        p_val_2 = pg.ttest(pros_cvs, p_cvs, paired=True)['p-val']
        p_val_3 = pg.ttest(td_cvs, pros_cvs, paired=True)['p-val']
        p_val_4 = pg.ttest(td_cvs, h_cvs, paired=True)['p-val']

        mean_scores = [np.nanmean(p_cvs),np.nanmean(pros_cvs), np.nanmean(h_cvs), np.nanmean(td_cvs)]
        std_scores = [np.nanstd(p_cvs), np.nanstd(pros_cvs), np.nanstd(h_cvs), np.nanstd(td_cvs)] / np.sqrt(len(h_cvs))
        models = ['TD-Persistence', 'Prospective', 'Hybrid', 'TD-Momentum']
        colors = ['#90ee90', '#FAFAD2', '#add8e6', '#f08080']


    elif (experiment == "instr_1"):

        p_val_1 = pg.ttest(p_cvs, h_cvs, paired=True)['p-val']
        p_val_2 = pg.ttest(pros_cvs, p_cvs, paired=True)['p-val']
        p_val_3 = pg.ttest(td_cvs, pros_cvs, paired=True)['p-val']
        p_val_4 = pg.ttest(td_cvs, h_cvs, paired=True)['p-val']

        mean_scores = [np.nanmean(h_cvs), np.nanmean(p_cvs), np.nanmean(pros_cvs), np.nanmean(td_cvs)]
        std_scores = [np.nanstd(h_cvs), np.nanstd(p_cvs), np.nanstd(pros_cvs),
                      np.nanstd(td_cvs)] / np.sqrt(len(h_cvs))
        models = ['Hybrid', 'TD-Persistence', 'Prospective', 'TD-Momentum']
        colors = ['#add8e6', '#90ee90', '#FAFAD2', '#f08080']


    elif  (experiment == 4):

        p_val_1 = pg.ttest(pros_cvs, p_cvs, paired=True)['p-val']
        p_val_2 = pg.ttest(h_cvs, pros_cvs, paired=True)['p-val']
        p_val_3 = pg.ttest(td_cvs, h_cvs, paired=True)['p-val']
        p_val_4 = pg.ttest(td_cvs, p_cvs, paired=True)['p-val']

        mean_scores = [np.nanmean(pros_cvs), np.nanmean(p_cvs), np.nanmean(h_cvs), np.nanmean(td_cvs)]
        std_scores = [np.nanstd(pros_cvs), np.nanstd(p_cvs), np.nanstd(h_cvs),
                        np.nanstd(td_cvs)] / np.sqrt(len(h_cvs))
        models = ['Prospective', 'TD-Persistence', 'Hybrid', 'TD-Momentum']
        colors = ['#FAFAD2', '#90ee90', '#add8e6', '#f08080']


    pvals = [p_val_1, p_val_2, p_val_3, p_val_4]

    draw_significant_bar_plots(ax, data, models,
                               mean_scores, std_scores, pvals, None, "CV metric", colors=colors)


def plot_model_fits_prospective():
    fig = plt.figure(figsize=(11, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[0.1, 1, 1])

    axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])]
    experiment = 1

    AIC = True

    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'prospective', AIC)
    bics_dl_2, _ = get_model_BIC(experiment, 'prospective_dl', AIC)
    bics_m_2, _ = get_model_BIC(experiment, 'prospective_momentum', AIC)
    bics_dlm_2, _ = get_model_BIC(experiment, 'prospective_dl_momentum', AIC)
    bics_hp_2, _ = get_model_BIC(experiment, 'prospective_hyperbolic', AIC)

    fits_exp_2 = [np.mean(bics_m_2), np.mean(bics_dlm_2), np.mean(bics_dl_2), np.mean(bics_hp_2), np.mean(bics_pros_2)]
    std_fits_exp_2 = [np.std(bics_m_2), np.std(bics_dlm_2), np.std(bics_dl_2), np.std(bics_hp_2), np.std(bics_pros_2)] / np.sqrt(len(bics_dlm_2))


    if AIC:
        ylabel = "AIC scores"
    else:
        ylabel = "BIC scores"

    pvals2 = [pg.ttest(bics_m_2, bics_dlm_2, paired=True)['p-val'], pg.ttest(bics_dlm_2, bics_dl_2, paired=True)['p-val'],
                pg.ttest(bics_dl_2, bics_hp_2, paired=True)['p-val'], pg.ttest(bics_hp_2, bics_pros_2, paired=True)['p-val']]

    draw_significant_bar_plots(axs[0], None, ['Pros+Mo', 'Pros+DL+M', 'Pros+DL', 'Pros+Hyperbolic', 'Prospective'],
                               fits_exp_2, std_fits_exp_2, pvals2, ylabel,
                               "AIC metric")

    AIC = False

    bics_pros_2, ll_pros_2 = get_model_BIC(experiment, 'prospective', AIC)
    bics_dl_2, _ = get_model_BIC(experiment, 'prospective_dl', AIC)
    bics_m_2, _ = get_model_BIC(experiment, 'prospective_momentum', AIC)
    bics_dlm_2, _ = get_model_BIC(experiment, 'prospective_dl_momentum', AIC)
    bics_hp_2, _ = get_model_BIC(experiment, 'prospective_hyperbolic', AIC)

    fits_exp_2 = [np.mean(bics_dlm_2), np.mean(bics_m_2), np.mean(bics_dl_2), np.mean(bics_hp_2), np.mean(bics_pros_2)]
    std_fits_exp_2 = [np.std(bics_dlm_2), np.std(bics_m_2), np.std(bics_dl_2), np.std(bics_hp_2), np.std(bics_pros_2)] / np.sqrt(
        len(bics_dlm_2))


    if AIC:
        ylabel = "AIC metric"
    else:
        ylabel = "BIC metric"

    pvals2 = [pg.ttest(bics_dlm_2, bics_m_2, paired=True)['p-val'], pg.ttest(bics_m_2, bics_dl_2, paired=True)['p-val'],
              pg.ttest(bics_dl_2, bics_pros_2, paired=True)['p-val'],
              pg.ttest(bics_dlm_2, bics_pros_2, paired=True)['p-val']]

    draw_significant_bar_plots(axs[1], None,['Pros+DL+M', 'Pros+Mo', 'Pros+DL', 'Pros+Hyperbolic', 'Prospective'],
                               fits_exp_2, std_fits_exp_2, pvals2, ylabel, 'BIC metric')

    plt.savefig("figures/manuscript/model_fits_prospective.png")
    plt.show()


def plot_model_fits_experiment(experiment=1, axs=None):
    if axs is None:
        if experiment == 4:
            fig = plt.figure(figsize=(15, 3))
        else:
            fig = plt.figure(figsize=(18, 3))
        gs = GridSpec(1, 4, width_ratios=[0.1, 1, 1, 1])
        axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3])]
        show = True
    else:
        show = False

    plot_aic_bic(axs[0], experiment, AIC=True)
    plot_aic_bic(axs[1], experiment, AIC=False)
    plot_cv_score(axs[2], experiment)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    if experiment != 4:

        axs[0].annotate("Experiment " + str(experiment), xy=(-0.45, 0.5), rotation=90, xycoords='axes fraction',
                    fontsize=15, fontweight='bold', ha='center', va='center')

    #fig.subplots_adjust(left=0.16, hspace=0.8)

    if show:
        plt.tight_layout()
        plt.savefig("figures/model_fits_experiment_" + str(experiment) + ".png")
        plt.show()


def plot_model_fits_instructions(axs=None):
    if axs is None:
        fig = plt.figure(figsize=(11, 4))
        gs = GridSpec(1, 3, width_ratios=[0.1, 1, 1])
        axs = [plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])]
        show = True
    else:
        show = False

    plot_aic_bic(axs[0], experiment=1, AIC=False)
    plot_aic_bic(axs[1], experiment="instr_1", AIC=False)

    title_box_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5)

    axs[0].annotate("No instructions", xy=(0.5, 1.2), xycoords='axes fraction',
                    fontsize=15, ha='center', va='center', fontweight='bold')

    axs[1].annotate("Instructions", xy=(0.5, 1.2), xycoords='axes fraction',
                    fontsize=15, ha='center', va='center', fontweight='bold')

    if show:
        plt.tight_layout()
        plt.savefig("figures/model_fits_instructions.png")
        plt.show()





if __name__ == "__main__":
    #plot_model_fits_experiment(experiment=1)
    #plot_model_fits_instructions()
    plot_model_fits_prospective()

