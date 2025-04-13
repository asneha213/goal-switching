import sys
sys.path.append('..')

from analysis.behavior.measures import *

from .run_model import get_model

import pickle
import numpy as np


def get_model_measure_experiment(experiment, measure_name, mode="condition"):
    """
    Get aggregate behavioral measure across all participants in an experiment
    """

    sub_measures = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        subject_measures = ModelMeasure(subject_id=subject_id, experiment=experiment)

        if mode == "condition":
            measure = subject_measures.get_sum_measure_condition(measure_name)
        elif mode == "mean_condition":
            measure = subject_measures.get_mean_measure_condition(measure_name)
        elif mode == "measure":
            measure = subject_measures.get_individual_measure(measure_name)
        sub_measures.append(measure)
    return np.array(sub_measures)

class ModelMeasure(SubjectMeasure):
    def __init__(self, subject_id, experiment, model_res=None):
        super().__init__(subject_id, experiment, model_res)


    def get_model_subject_actions(self, trial):
        model_name = "prospective"
        experiment = self.experiment
        file_name = "results_latent/sims/" + model_name + "_" + str(experiment) + "_optimal_params.pkl"
        with open(file_name, "rb") as f:
            params = pickle.load(f)
            params= params[0]
        self.params = params
        self.model_name = model_name
        self.model = get_model(model_name, params)
        self.model.M = {
            'p': trial['P'],
            'c': trial['C'],
            'b': trial['B']
        }

        self.model.slots = trial['slots_pre']
        modelflip, cardflip = self.model.run_subject_action_trial(trial, get_action=True)
        return modelflip, cardflip


    def get_goal_valuation(self, measure):
        prospective = np.zeros(self.num_conditions)
        retrospective = np.zeros(self.num_conditions)
        other = np.zeros(self.num_conditions)
        counts = np.zeros(self.num_conditions)
        counts_all = np.zeros(self.num_conditions)
        pros_all = np.zeros(self.num_conditions)
        retro_all = np.zeros(self.num_conditions)
        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                condition = trial['condition']

                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                if self.experiment != 4:
                    targets = np.array([7, 7, 7])
                else:
                    targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])

                progress = {'p': slots['p'] / targets[0],
                            'c': slots['c'] / targets[1],
                            'b': slots['b'] / targets[2]}
                retrospective_choice = ['p', 'c', 'b'][np.argmax(list(progress.values()))]
                next_best_token = ['p', 'c', 'b'][
                    find_second_highest_index(list(progress.values()))]
                modelflip, cardflip = self.get_model_subject_actions(trial)
                prospective_choice = modelflip.lower()[0]

                counts_all[condition] += 1
                cardselect = trial['cardselect'][0].lower()
                if cardselect != prospective_choice:
                    pros_all[condition] += 1
                    if progress[cardselect] >= progress[prospective_choice]:
                        retro_all[condition] += 1


                if measure == "prospective_retrospective_diverge":
                    if prospective_choice == retrospective_choice:
                        continue
                    cardselect = trial['cardselect'][0].lower()
                    counts[condition] += 1

                    if cardselect == prospective_choice:
                        prospective[condition] += 1
                    elif cardselect == retrospective_choice:
                        retrospective[condition] += 1
                    else:
                        other[condition] += 1
                else:
                    if prospective_choice != retrospective_choice:
                        continue
                    cardselect = trial['cardselect'][0].lower()
                    counts[condition] += 1

                    if cardselect == prospective_choice:
                        prospective[condition] += 1
                    elif cardselect == next_best_token:
                        retrospective[condition] += 1
                    else:
                        other[condition] += 1



        return prospective / counts, retrospective / counts, other / counts, pros_all / counts_all, counts / counts_all


    def get_stay_switch_condition_counts(self):

        # switches away from max prob token
        # previous round rewarded or not
        prob_switches = np.zeros((self.num_conditions, 2))

        # switches away from max slot token
        slot_switches = np.zeros((self.num_conditions, 2))

        prob_counts = np.zeros((self.num_conditions, 2))
        slot_counts = np.zeros((self.num_conditions, 2))

        prev_cardselect = ''
        prev_reward = -1

        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]

                modelflip, cardflip = self.get_model_subject_actions(trial)

                prospective_choice = modelflip.lower()[0]

                condition = trial['condition']
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == prospective_choice:
                    continue

                if slots[max_slot_token] == slots[prospective_choice]:
                    continue

                cardselect = trial['cardselect'][0].lower()

                if prev_reward != -1:
                    if prev_cardselect == prospective_choice:
                        prob_counts[condition][prev_reward] += 1
                        if cardselect != prospective_choice:
                            prob_switches[condition][prev_reward] += 1
                    elif prev_cardselect == max_slot_token:
                        slot_counts[condition][prev_reward] += 1
                        if cardselect != max_slot_token:
                            slot_switches[condition][prev_reward] += 1

                prev_cardselect = cardselect
                if trial['current_token'] == 'e':
                    prev_reward = 0
                else:
                    prev_reward = 1

        return slot_switches/ slot_counts, prob_switches/ prob_counts


    def get_stay_switch_condition_progress_counts(self):

        # switches away from max prob token
        # previous round rewarded or not
        prob_switches = np.zeros((2,2))

        # switches away from max slot token
        slot_switches = np.zeros((2, 2))

        prob_counts = np.zeros((2, 2))
        slot_counts = np.zeros((2, 2))

        prev_cardselect = ''
        prev_reward = -1

        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                modelflip, cardflip = self.get_model_subject_actions(trial)

                prospective_choice = modelflip.lower()[0]

                probs = [trial['P'], trial['C'], trial['B']]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                if self.experiment != 4:
                    targets = np.array([7, 7, 7])
                else:
                    targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])

                progress = {'p': slots['p'] / targets[0],
                            'c': slots['c'] / targets[1],
                            'b': slots['b'] / targets[2]}
                max_slot_token = ['p', 'c', 'b'][
                    np.argmax(list(progress.values()))]

                if progress[max_slot_token] - progress[prospective_choice] <= 0.5:
                    progress = 0
                else:
                    progress = 1

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == prospective_choice:
                    continue

                cardselect = trial['cardselect'][0].lower()

                if prev_reward != -1:
                    if prev_cardselect == prospective_choice:
                        prob_counts[progress][prev_reward] += 1
                        if cardselect != prospective_choice:
                            prob_switches[progress][prev_reward] += 1
                    elif prev_cardselect == max_slot_token:
                        slot_counts[progress][prev_reward] += 1
                        if cardselect != max_slot_token:
                            slot_switches[progress][prev_reward] += 1

                prev_cardselect = cardselect
                if trial['current_token'] == 'e':
                    prev_reward = 0
                else:
                    prev_reward = 1

        return safe_divide(slot_switches, slot_counts), safe_divide(prob_switches, prob_counts)
