
from models import *
from behavior import SubjectMeasure

import numpy as np
import optuna
import pickle
from fit_behavior import get_optuna_params


class ModelOptimizer:
    def __init__(self, experiment, model_name, episodes=None):
        self.experiment = experiment
        self.model_name = model_name
        self.episodes = episodes

    def get_measure_mean(self, params):
        seeds = np.random.randint(0, 1000, 10)
        model_results = []
        for seed in seeds:
            model = get_model(self.model_name, params)
            if self.episodes:
                model_res = model.run_model(experiment=self.experiment, episodes=self.episodes, seed=seed)
            else:
                model_res = model.run_model(experiment=self.experiment, seed=seed)
            measures = SubjectMeasure(subject_id=seed, experiment=self.experiment, model_res=model_res)
            sub_measure = measures.get_task_performance()
            model_results.append(sub_measure)
        return model_results

    def get_performance(self, params):
        model_measures = self.get_measure_mean(params)
        err = -1 * np.mean(model_measures)
        print(err)
        return err

    def get_optuna_objective(self, trial):
        params = get_optuna_params(trial, self.model_name)
        model_measures = self.get_measure_mean(params)
        err = -1 * np.mean(model_measures)
        return err

    def fit_optuna(self):
        best_params = []
        best_vals = []
        for i in range(1):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=50)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals)
        return best_params, best_vals

    def get_model_optimal_params(self):
        file_name = "results_latent/sims/" + self.model_name + "_" + str(self.experiment) + "_optimal_params.pkl"
        res = self.fit_optuna()
        with open(file_name, 'wb') as f:
           pickle.dump(res, f)
        return res

    def load_model_optimal_params(self):
        file_name = "results/sims/" + self.model_name + "_" + str(self.experiment) + "_optimal_params.pkl"
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        return params

    def simulate_params(self, params=None, measure_name='num_goals', seed=100):

        np.random.seed(seed)
        if params is None:
            file_name = "results/sims/" + self.model_name + "_" + str(self.experiment) + "_optimal_params.pkl"
            with open(file_name, "rb") as f:
                params = pickle.load(f)
                params = list(params[0].values())
        model_measures = []
        if self.experiment == 1:
            num_sims = 44
        elif self.experiment == 2:
            num_sims = 50

        elif self.experiment == 3:
            num_sims = 30
        elif self.experiment == 4:
            num_sims = 30

        elif self.experiment == "normative":
            num_sims = 10

        for seed in range(num_sims):
            model = get_model(self.model_name, params)
            model_res = model.run_model(experiment=self.experiment, seed=seed)
            measures = SubjectMeasure(subject_id=seed, experiment=self.experiment, model_res=model_res)
            if measure_name in ['switches_actions', 'switches_probes', 'num_goals' ]:
                sub_measure = measures.get_sum_measure_condition(measure_name)
            elif measure_name == "retro_value_count":
                sub_measure = measures.get_individual_measure(measure_name)
            else:
                sub_measure = measures.get_mean_measure_condition(measure_name)
            model_measures.append(sub_measure)
        return np.array(model_measures)


if __name__ == "__main__":
    experiment = 4

    model_name = "momentum"
    model_name = "prospective"
    #model_name = "td_persistence"
    #model_name = "momentum"
    #model_name = "prospective"
    #model_name = "retrospective"
    #model_name = "prospective_rates"
    #model_name = "momentum_rates"
    #model_name = "prospective_hyperbolic"

    res = ModelOptimizer(experiment, model_name).get_model_optimal_params()
    print(res)

    #res = ModelOptimizer(experiment, model_name).load_model_optimal_params()
    #print(res)
    #model_name = "rescorla"
    # model_name = "prospective_rates"
    #model_name = "momentum_rates"

    #
    # measure_name = "retro_value"
    # params = [0.4, 0.2, 1, 10, 10, 0.9]
    # res = ModelOptimizer(experiment, model_name).simulate_params(\
    #     params=params, measure_name=measure_name)
    # print(np.sum(np.mean(res, axis=0)))
    # print(np.mean(res, axis=0))
    # #print(res)
    # #
    #
    # model_name = "momentum"
    #
    # params = [0.4, 0.2, -1, 3, 10, 0.9]
    # res = ModelOptimizer(experiment, model_name).simulate_params( \
    #     params=params, measure_name=measure_name)
    # print(np.sum(np.mean(res, axis=0)))
    # print(np.mean(res, axis=0))
    # #print(res)
    # #
    #
    # model_name = "td_persistence"
    #
    # params = [0.4, 0.2, -1, 7, 3, 0.9]
    # res = ModelOptimizer(experiment, model_name).simulate_params( \
    #     params=params, measure_name=measure_name)
    # print(np.sum(np.mean(res, axis=0)))
    # print(np.mean(res, axis=0))
