from .prospective import Prospective
from .momentum import Momentum
from .td_persistence import TDPersistence
from .hybrid import Hybrid
from .retrospective import Retrospective
from .prospective_hyperbolic import ProspectiveHyperbolic
from .prospective_dl import ProspectiveDL
from .prospective_momentum import ProspectiveMomentum
from .prospective_dl_momentum import ProspectiveDLMomentum
from .rescorla import Rescorla

import sys
sys.path.append('../')

import numpy as np

from behavior import *


def get_model(model_name, params):
    if model_name == "prospective":
        return Prospective('model')(params)
    elif model_name == "hybrid":
        return Hybrid('model')(params)
    elif model_name == "momentum":
        return Momentum('model')(params)
    elif model_name == "td_persistence":
        return TDPersistence('model')(params)
    elif model_name == "retrospective":
        return Retrospective(params)
    elif model_name == "prospective_dl":
        return ProspectiveDL('model')(params)
    elif model_name == "prospective_momentum":
        return ProspectiveMomentum('model')(params)
    elif model_name == "prospective_dl_momentum":
        return ProspectiveDLMomentum('model')(params)
    elif model_name == "prospective_hyperbolic":
        return ProspectiveHyperbolic('model')(params)
    elif model_name == "rescorla":
        return Rescorla(params)

    else:
        raise Exception("Model not found")


class RunModel:
    def __init__(self, experiment, model_name, params):
        self.experiment = experiment
        self.model_name = model_name
        self.params = params

        self.model = get_model(model_name, params)
        self.model_res = self.get_model_res()

    def get_model_res(self, seed=100):

        model = get_model(self.model_name, self.params)
        model_res = model.run_model(experiment=self.experiment, seed=seed)
        return model_res


class ModelSim:
    def __init__(self, experiment, model_name, params, num_sims=5):
        self.experiment = experiment
        self.model_name = model_name
        self.params = params
        self.num_sims = num_sims

    def get_model_measures(self, measure_list, modes, seed=100):
        measures_array = []
        for i in range(len(measure_list)):
            measure_name = measure_list[i]
            measures = []
            for k in range(self.num_sims):
                model = get_model(self.model_name, self.params)
                model_res = model.run_model(experiment=self.experiment, seed=seed)
                measure = SubjectMeasure(subject_id=seed, experiment=self.experiment, model_res=model_res)
                measures.append(measure.get_measure(measure_name, modes[i]))
            measures_array.append(measures)
        return np.array(measures_array)

    def get_model_measure(self, measure_name):
        measures = []
        for i in range(self.num_sims):
            model = get_model(self.model_name, self.params)
            model_res = model.run_model(experiment=self.experiment)
            measure_model = SubjectMeasure(subject_id=-1, experiment=self.experiment, model_res=model_res)
            measure = measure_model.get_sum_measure_condition(measure_name)
            measures.append(measure)
        return np.array(measures)


def get_simulations_subject(experiment, model_name, params):
    runmodel = RunModel(experiment, model_name, params)
    model_res = runmodel.get_model_res()
    model_measures = SubjectMeasure(subject_id=-1, experiment=experiment, model_res=model_res)
    measures_sims = model_measures.get_measures_dict()
    return measures_sims


def get_model_simulation_of_measure(experiment, model_name, measure_name, optimal=False, sims_per_subject=1):
    subject_measure_simulations = []
    subject_names = get_experiment_subjects(experiment)
    for sim_num in range(sims_per_subject):
        subject_measures_single_sim = []
        for subject_num in range(len(subject_names)):
            subject_id = subject_names[subject_num]
            if not optimal:
                with open('results_latent/' + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl", "rb") as f:
                    model_fits = pickle.load(f)
                params = model_fits['params']
            else:
                file_name = "results_latent/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
                with open(file_name, "rb") as f:
                    params = pickle.load(f)
                    params = params[0]

            runmodel = RunModel(experiment, model_name, params)
            model_res = runmodel.get_model_res()

            subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)

            if measure_name == "retro_value":
                measure = subject_measures.get_mean_measure_condition(measure_name)
            elif measure_name == "retro_value_count":
                measure = subject_measures.get_individual_measure(measure_name)

            subject_measures_single_sim.append(measure)
        subject_measure_simulations.append(subject_measures_single_sim)

    if sims_per_subject == 1:
        return np.array(subject_measure_simulations)
    else:
        return subject_measure_simulations
