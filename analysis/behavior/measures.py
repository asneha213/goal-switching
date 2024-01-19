from .behavior_utils import *

import sys
sys.path.append("..")

from .datautils import *


def get_measure_experiment(experiment, measure_name, mode="condition"):
    sub_measures = []
    subject_names = get_experiment_subjects(experiment)
    for subject_num in range(len(subject_names)):
        subject_id = subject_names[subject_num]
        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment)

        if mode == "condition":
            measure = subject_measures.get_sum_measure_condition(measure_name)
        elif mode == "mean_condition":
            measure = subject_measures.get_mean_measure_condition(measure_name)
        elif mode == "measure":
            measure = subject_measures.get_individual_measure(measure_name)
        sub_measures.append(measure)
    return np.array(sub_measures)


class SubjectMeasure:
    def __init__(self, subject_id, experiment, model_res=None):
        self.subject_id = subject_id
        self.experiment = experiment

        self.details = get_experiment_trial_details(experiment)

        self.num_episodes = self.details["num_episodes"]
        self.num_trials = self.details["num_trials"]
        self.num_conditions = self.details["num_conditions"]

        if model_res:
            self.data = model_res
        else:

            self.data = get_subject_data_from_id(experiment, subject_id)
        self.blocks = self.data['GoalSwitching']
        self.block_measures = self.get_block_wise_measures()

    def get_block_wise_measures(self):
        measures = {}
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            measures[block_num] = {}
            measures[block_num]['condition'] = block['trial_0']['condition']

            if 'reward_trials' not in block:
                measures[block_num]['num_goals'] = 0
            else:
                measures[block_num]['num_goals'] = len(block['reward_trials'])
            measures[block_num]['retro_value'] = self.get_retro_valuation_factor(block_num, mode="slot")
            measures[block_num]['optimal_action'] = self.get_retro_valuation_factor(block_num, mode="prob")

            measures[block_num]['switches_probes'] = self.get_goal_switches_across_probes(block_num)
            measures[block_num]['switches_blocks'] = self.get_goal_switches_across_blocks(block_num)
            measures[block_num]['switches_actions'] = self.get_action_switches(block_num)
            measures[block_num]['condition_action'] = self.get_condition_max_action(block_num)
            measures[block_num]['switches_completion'] = self.get_switches_after_completion(block_num)

        return measures

    def get_measure_condition(self, measure_name):
        measures = self.block_measures
        measure_condition = np.zeros((self.num_conditions, int(len(self.blocks.keys())/ self.num_conditions)))
        counts = np.zeros(self.num_conditions)
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]
            condition = block['trial_0']['condition']
            measure_condition[condition][int(counts[condition])] = (measures[block_num][measure_name])
            counts[condition] += 1
        return measure_condition

    def get_sum_measure_condition(self, measure_name):

        measure_array = self.get_measure_condition(measure_name)
        sum_measure = np.sum(measure_array, axis=1)

        return sum_measure

    def get_mean_measure_condition(self, measure_name):

        measure_array = self.get_measure_condition(measure_name)
        sum_measure = np.mean(measure_array, axis=1)

        return sum_measure

    def get_measure_blocks(self, measure_name):
        measures = self.get_block_wise_measures()
        measure = np.zeros(self.num_episodes)
        for block_num in list(self.blocks.keys()):
            measure[block_num] = measures[block_num][measure_name]
        return measure

    def get_measures_dict(self):
        measures = {}
        measures['num_goals'] = self.get_sum_measure_condition("num_goals")
        measures['switches_actions'] = self.get_sum_measure_condition("switches_actions")
        measures['switches_probes'] = self.get_sum_measure_condition("switches_probes")
        measures['retro_value'] = self.get_mean_measure_condition("retro_value")
        measures['condition_action'] = self.get_mean_measure_condition("condition_action")
        measures['performance'] = np.sum(measures['num_goals'])
        return measures

    def get_multiple_measures(self, measure_list):
        measures = []
        for measure_name in measure_list:
            measures.append(self.get_sum_measure_condition(measure_name))
        return np.array(measures)

    def get_measure(self, measure_name, mode="condition"):
        if mode == "condition":
            measure = self.get_sum_measure_condition(measure_name)
        elif mode == "mean_condition":
            measure = self.get_mean_measure_condition(measure_name)
        elif mode == "measure":
            measure = self.get_individual_measure(measure_name)
        return measure

    def get_task_performance(self):
        total_rewards = 0
        for block_num in list(self.blocks.keys()):
            block = self.blocks[str(block_num)]

            if 'reward_trials' not in block:
                continue
            num_rewards = len(block['reward_trials'])
            total_rewards += num_rewards

        return total_rewards

    def get_num_goals_condition(self):
        num_goals_sub = np.zeros(self.num_conditions)
        num_goals_blocks = np.zeros(self.num_episodes)
        for block_num in range(len(self.blocks)):
            block = self.blocks[str(block_num)]
            if 'reward_trials' not in block:
                continue
            reward_trials = block['reward_trials']
            condition = block['trial_0']['condition']
            num_goals = len(reward_trials)
            num_goals_sub[condition] += num_goals
            num_goals_blocks[block_num] += num_goals
        if self.mode == "condition":
            return np.array(num_goals_sub)
        else:
            return np.array(num_goals_blocks)

    def get_retro_valuation_factor(self, block_num, mode="slot"):
        retro_values = 0
        counts = 0

        for trial_num in range(self.num_trials):
            trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]

            probs = [trial['P'], trial['C'], trial['B']]
            max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
            slots = trial['slots_pre']
            slots_arr = np.array([slots['p'], slots['c'], slots['b']])
            max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

            if np.max(slots_arr) == 0:
                continue

            if max_slot_token == max_prob_token:
                continue

            cardselect = trial['cardselect'][0].lower()
            counts += 1

            if cardselect == max_slot_token:
                retro_values += 1

        if counts == 0:
            return 0
        else:
            return retro_values / counts

    def get_retro_valuation_per_count(self):
        retro_values = np.zeros((self.num_conditions, 7))
        counts = np.zeros((self.num_conditions, 7))

        for block_num in self.blocks.keys():
            for trial_num in range(self.num_trials):
                trial = self.blocks[block_num]['trial_' + str(trial_num)]
                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == max_prob_token:
                    continue

                cardselect = trial['cardselect'][0].lower()

                counts[condition][slots[max_slot_token] - slots[max_prob_token]] += 1

                if cardselect == max_slot_token:
                    retro_values[condition][slots[max_slot_token] - slots[max_prob_token]] += 1

        return retro_values/ counts

    def get_proportion_divergence_pros_retro(self):
        counts = 0

        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]

                probs = [trial['P'], trial['C'], trial['B']]
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue
                if max_slot_token == max_prob_token:
                    continue
                counts += 1

        return counts / (self.num_episodes * self.num_trials)

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

                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == max_prob_token:
                    continue

                if slots[max_prob_token] == slots[max_slot_token]:
                   continue

                cardselect = trial['cardselect'][0].lower()

                if prev_cardselect == max_prob_token:
                    prob_counts[condition][prev_reward] += 1
                    if cardselect == max_slot_token:
                        prob_switches[condition][prev_reward] += 1
                elif prev_cardselect == max_slot_token:
                    slot_counts[condition][prev_reward] += 1
                    if cardselect == max_prob_token:
                        slot_switches[condition][prev_reward] += 1

                prev_cardselect = cardselect
                if trial['current_token'] == 'e':
                    prev_reward = 0
                else:
                    prev_reward = 1

        return slot_switches/ slot_counts, prob_switches/ prob_counts

    def get_condition_max_action(self, block_num):
        counts = 0
        actions = 0

        block = self.blocks[str(block_num)]

        for trial_num in range(self.num_trials):
            trial = block['trial_' + str(trial_num)]
            select = trial['cardselect'][0].lower()
            probs = [trial['P'], trial['C'], trial['B']]
            max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
            counts += 1
            if select == max_prob_token:
                actions += 1
        return actions / counts

    def get_switches_after_completion(self, block_num):
        switches = 0
        counts = 0
        if "reward_trials" not in self.blocks[str(block_num)]:
            return -1
        reward_trials = self.blocks[str(block_num)]['reward_trials']

        for trial_ind in range(len(reward_trials)):
            trial_num = reward_trials[trial_ind]
            block = self.blocks[str(block_num)]
            target = block['targets'][trial_ind]

            max_token = ['p', 'c', 'b'][np.argmax([block['trial_' + str(trial_num)]['P'], block['trial_' + str(trial_num)]['C'], block['trial_' + str(trial_num)]['B']])]

            if target != max_token:
                continue

            if trial_num < 29:
                trial_a = self.blocks[str(block_num)]['trial_' + str(trial_num + 1)]
            else:
                if str(int(block_num) + 1) not in self.blocks:
                    continue
                trial_a = self.blocks[str(int(block_num) + 1)]['trial_' + str(0)]
            cardselect = trial_a['cardselect'][0].lower()
            counts += 1
            if cardselect == target:
                switches += 1

        if counts == 0:
            return np.nan
        else:
            return switches/ counts

    def get_goal_valuation(self):
        max_prob = np.zeros(self.num_conditions)
        max_slot = np.zeros(self.num_conditions)
        other = np.zeros(self.num_conditions)
        counts = np.zeros(self.num_conditions)
        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue

                if slots[max_prob_token] == slots[max_slot_token]:
                    continue

                cardselect = trial['cardselect'][0].lower()
                counts[condition] += 1

                if cardselect == max_prob_token:
                    max_prob[condition] += 1
                elif cardselect == max_slot_token:
                    max_slot[condition] += 1
                else:
                    other[condition] += 1

        return max_prob / counts, max_slot / counts, other / counts

    def get_goal_actions_array(self):
        choices_p = []
        choices_c = []
        choices_b = []
        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                if trial['condition'] != 0:
                    continue
                cardselect = trial['cardselect'][0].lower()
                if cardselect == 'p':
                    choices_p.append(1)
                    choices_c.append(0)
                    choices_b.append(0)
                elif cardselect == 'c':
                    choices_p.append(0)
                    choices_c.append(1)
                    choices_b.append(0)
                else:
                    choices_p.append(0)
                    choices_c.append(0)
                    choices_b.append(1)

        goal_p = get_goals_from_choices(choices_p)
        goal_c = get_goals_from_choices(choices_c)
        goal_b = get_goals_from_choices(choices_b)

        return goal_p, goal_c, goal_b

    def get_goal_probes_array(self):
        choices_p = []
        choices_c = []
        choices_b = []
        choices_u = []
        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                if trial['condition'] != 0:
                    continue
                if (trial_num + 1) % 3 == 0:
                    if "probe_keypress" not in trial:
                        choices_p.append(0)
                        choices_c.append(0)
                        choices_b.append(0)
                        choices_u.append(0)
                        continue
                    probekey = trial["probe_keypress"]
                    if probekey == 37:
                        choices_p.append(1)
                        choices_c.append(0)
                        choices_b.append(0)
                        choices_u.append(0)
                    elif probekey == 38:
                        choices_p.append(0)
                        choices_c.append(1)
                        choices_b.append(0)
                        choices_u.append(0)
                    elif probekey == 39:
                        choices_p.append(0)
                        choices_c.append(0)
                        choices_b.append(1)
                        choices_u.append(0)
                    else:
                        choices_p.append(0)
                        choices_c.append(0)
                        choices_b.append(0)
                        choices_u.append(1)

        return choices_p[:59], choices_c[:59], choices_b[:59]

    def get_goal_switches_across_blocks(self, block_num):

        block = self.blocks[str(block_num)]
        if 'targets' not in block:
            return 0
        targets = block['targets']
        switches = get_goal_switches(targets)
        switches_sub= switches

        return switches_sub

    def get_goal_switches_across_probes(self, block_num):
        switches_sub = 0
        last_probe = ''
        count = 0

        for trial_num in range(self.num_trials):
            trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
            if "probe_keypress" not in trial:
                continue
            probekey = trial["probe_keypress"]
            if probekey != last_probe:
                switches_sub += 1
            last_probe = probekey
            count += 1

        return switches_sub

    def get_action_switches(self, block_num):
        switches_sub = 0
        last_action = ''
        count = 0

        for trial_num in range(self.num_trials):
            # if trial_num % 3 != 0:
            #     continue
            trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
            cardselect = trial["cardselect"]
            card_type = cardselect[0].lower()

            if card_type != last_action:
                switches_sub += 1
            last_action = card_type
            count += 1

        return switches_sub

    def get_individual_measure(self, measure_name):

        if measure_name == "performance":
            measure = self.get_task_performance()
        elif measure_name == "retro_value_count":
            measure = self.get_retro_valuation_per_count()
        elif measure_name == "goal_shifts":
            measure = self.get_goal_actions_array()
        elif measure_name == "goal_probe_shifts":
            measure = self.get_goal_probes_array()
        elif measure_name == "goal_valuation":
            measure = self.get_goal_valuation()
        return measure



