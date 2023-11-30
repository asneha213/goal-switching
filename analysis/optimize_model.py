from analysis.models import *
from analysis.behavior import SubjectMeasure

import numpy as np
import optuna
import pickle
from fit_behavior import get_optuna_params


class ModelOptimizer:
    def __init__(self, experiment, model_name, episodes=None):
        self.experiment = experiment
        self.model_name = model_name
        self.episodes = episodes

    def get_measure_mean(self, params, measure_name='performance'):
        seeds = np.random.randint(0, 1000, 20)
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


    def match_reference_measure(self, params, ref_value, measure_name='num_goals'):
        model_results = self.get_measure_mean(params, measure_name=measure_name)
        err = np.mean(np.sum(np.abs(np.array(model_results) - ref_value), axis=1))
        print(err, model_results[0])
        return err


    def get_performance(self, params):
        model_measures = self.get_measure_mean(params)
        err = -1 * np.mean(model_measures)
        # err = self.match_reference_measure(params, ref_value=[0.3, 0.3], measure_name='condition_action')
        print(err)
        return err

    def get_optuna_objective(self, trial):

        params = get_optuna_params(trial, self.model_name)
        model_measures = self.get_measure_mean(params)
        err = -1 * np.mean(model_measures)
        #err = self.match_reference_measure(params, ref_value=[0.3, 0.3], measure_name='condition_action')
        return err

    def fit_optuna(self):
        best_params = []
        best_vals = []
        for i in range(5):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=50)
            #print(study.best_params, study.best_value)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals, best_params)
        return best_params, best_vals


    def get_model_optimal_params(self):
        file_name = "results/sims/" + self.model_name + "_" + str(self.experiment) + "_optimal_params.pkl"
        res = self.fit_optuna()
        with open(file_name, 'wb') as f:
            pickle.dump(res, f)
        return res

    def simulate_params(self, params=None, measure_name='num_goals', seed=100):

        np.random.seed(seed)
        if params is None:
            file_name = "results/sims/" + self.model_name + "_" + str(self.experiment) + "_optimal_params.pkl"
            with open(file_name, "rb") as f:
                params = pickle.load(f)
                params = list(params[0].values())
        model_measures = []
        if self.experiment == 1:
            num_sims = 69
            #num_sims = 29
        elif self.experiment == 2:
            num_sims = 50

        elif self.experiment == "normative":
            num_sims = 10

        #num_sims = 2
        for seed in range(num_sims):
            model = get_model(self.model_name, params)
            model_res = model.run_model(experiment=self.experiment, seed=seed)
            measures = SubjectMeasure(subject_id=seed, experiment=self.experiment, model_res=model_res)
            #sub_measure = measures.get_subject_measure(measure_name=measure_name)
            if measure_name in ['switches_actions', 'switches_probes', 'num_goals' ]:
                sub_measure = measures.get_sum_measure_condition(measure_name)
            elif measure_name == "retro_value_count":
                sub_measure = measures.get_individual_measure(measure_name)
            else:
                sub_measure = measures.get_mean_measure_condition(measure_name)
            model_measures.append(sub_measure)
        return np.array(model_measures)



if __name__ == "__main__":
    experiment = "normative"
    model_name = 'td_hierarchy'
    res = ModelOptimizer(experiment, model_name).get_model_optimal_params()
    #res = ModelOptimizer(experiment, model_name).simulate_params()
    print(res)