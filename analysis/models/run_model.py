from .prospective import Prospective
from .momentum import Momentum
from .td_persistence import TDPersistence
from .hybrid import Hybrid
from .retrospective import Retrospective
from .prospective_dl import ProspectiveDL
from .prospective_momentum import ProspectiveMomentum
from .prospective_dl_momentum import ProspectiveDLMomentum
from .rescorla import Rescorla
from .hybrid_mp import HybridMP

from .prospective_rates import ProspectiveRates
from .momentum_rates import MomentumRates


import sys
sys.path.append('../')

import numpy as np

from behavior import SubjectMeasure


def get_model(model_name, params):
    if model_name == "prospective":
        return Prospective(params)
    elif model_name == "hybrid":
        return Hybrid(params)
    elif model_name == "momentum":
        return Momentum(params)
    elif model_name == "td_persistence":
        return TDPersistence(params)
    elif model_name == "retrospective":
        return Retrospective(params)
    elif model_name == "prospective_dl":
        return ProspectiveDL(params)
    elif model_name == "prospective_momentum":
        return ProspectiveMomentum(params)
    elif model_name == "prospective_dl_momentum":
        return ProspectiveDLMomentum(params)
    elif model_name == "rescorla":
        return Rescorla(params)
    elif model_name == "hybrid_mp":
        return HybridMP(params)

    elif model_name == "prospective_rates":
        return ProspectiveRates(params)
    elif model_name == "momentum_rates":
        return MomentumRates(params)

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
