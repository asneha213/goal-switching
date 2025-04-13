import numpy as np
import scipy.stats as stats
import pingouin as pg


def get_stats_for_reporting_average_of_correlations(correlations):
    """
    :param corrs: list of correlation values
    :return: average correlation, 95% confidence interval, and p-value
    """

    # Step 2: Calculate the average of the original correlations
    avg_correlation = np.nanmean(correlations)

    # Step 3: Calculate confidence bounds for the average correlation

    # Step 3: Calculate the standard error of the mean (SEM)
    n = len(correlations)
    sem = np.nanstd(correlations) / np.sqrt(n)

    # Step 4: Calculate the t-statistic and p-value
    t_statistic = avg_correlation / sem
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n - 1))

    # Step 5: Calculate the confidence interval
    confidence_level = 0.95  # You can change this to your desired confidence level
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_critical * sem
    confidence_interval = (avg_correlation - margin_of_error, avg_correlation + margin_of_error)

    # Step 5: Print the results with a more precise p-value
    print(f"Average Correlation: {avg_correlation:.2f}")
    print(
        f"{confidence_level * 100}% Confidence Interval for Average Correlation: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")

    print(p_value)
    return avg_correlation, confidence_interval, p_value


def get_anova_conditions(experiment, measures):
    import pandas as pd

    num_subs = len(measures)
    if (experiment == 1) or (experiment == "instr_1"):
        data_dict = {'subjects': list(range(num_subs)) * 3,
                     'condition': [0] * num_subs + [1] * num_subs + [2] * num_subs,
                     'measures': list(measures[:, 0]) + list(measures[:, 1]) + list(measures[:, 2])}
    else:
        data_dict = {'subjects': list(range(num_subs)) * 2, 'condition': [0] * num_subs + [1] * num_subs,
                     'measures': list(measures[:, 0]) + list(measures[:, 1])}
    df = pd.DataFrame(data_dict)

    rm_anova_result = pg.rm_anova(data=df, dv='measures', \
                                  within='condition', subject='subjects')

    print(rm_anova_result)



