from .behavior_utils import *

import sys

sys.path.append('../')

from .datautils import *

import pickle

def get_measure_experiment(experiment, measure_name, mode="condition"):
    """
    Get aggregate behavioral measure across all participants in an experiment
    """

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
            measures[block_num][
                'retro_value_first'] = self.get_retro_valuation_factor(block_num,
                                                                 mode="slot", half='first')
            measures[block_num][
                'retro_value_second'] = self.get_retro_valuation_factor(block_num,
                                                                 mode="slot", half='second')
            measures[block_num]['optimal_action'] = self.get_retro_valuation_factor(block_num, mode="prob")
            measures[block_num]['obvious_choice'] = self.get_retro_valuation_factor(block_num, mode="slot", diverge=False)

            measures[block_num]['condition_action'] = self.get_condition_max_action(block_num)
            measures[block_num]['condition_action_first'] = self.get_condition_max_action(block_num, half='first')
            measures[block_num]['condition_action_second'] = self.get_condition_max_action(block_num, half='second')

            measures[block_num]['switches_completion'] = self.get_switches_after_completion(block_num)
            measures[block_num]['switches_probes'] = self.get_goal_switches_across_probes(block_num)
            measures[block_num]['switches_blocks'] = self.get_goal_switches_across_blocks(block_num)
            measures[block_num]['switches_actions'] = self.get_action_switches(block_num)


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
        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
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


    def get_obvious_choice_proportion(self, block_num, half=None):
        retro_values = 0
        counts = 0

        for trial_num in range(self.num_trials):
            trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]

            if half == "first":
                if trial_num >= self.num_trials / 2:
                    continue
            elif half == "second":
                if trial_num < self.num_trials / 2:
                    continue

            probs = [trial['P'], trial['C'], trial['B']]
            max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
            slots = trial['slots_pre']
            slots_arr = np.array([slots['p'], slots['c'], slots['b']])
            if self.experiment != 4:
                targets = np.array([7, 7, 7])
            else:
                targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])

            progress_arr = slots_arr / targets

            max_slot_token = ['p', 'c', 'b'][np.argmax(progress_arr)]

            if np.max(slots_arr) == 0:
                continue

            if max_slot_token != max_prob_token:
                continue

            cardselect = trial['cardselect'][0].lower()
            counts += 1

            if cardselect == max_slot_token:
                retro_values += 1

        if counts == 0:
            return 0
        else:
            return retro_values / counts


    def get_obvious_choice_per_count(self, half=None):
        retro_values = np.zeros((self.num_conditions, 7))
        counts = np.zeros((self.num_conditions, 7))

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
            for trial_num in range(self.num_trials):

                if half == "first":
                    if trial_num >= self.num_trials / 2:
                        continue
                elif half == "second":
                    if trial_num < self.num_trials / 2:
                        continue

                trial = self.blocks[block_num]['trial_' + str(trial_num)]
                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token != max_prob_token:
                    continue

                cardselect = trial['cardselect'][0].lower()

                next_best_token = ['p', 'c', 'b'][find_second_highest_index(slots_arr)]

                counts[condition][slots[max_slot_token] - slots[next_best_token]] += 1

                if cardselect == max_slot_token:
                    retro_values[condition][slots[max_slot_token] - slots[next_best_token]] += 1

        return safe_divide(retro_values, counts)

    def get_retro_valuation_factor(self, block_num, mode="slot", half=None, diverge=True):
        retro_values = 0
        counts = 0

        for trial_num in range(self.num_trials):
            trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]

            if half == "first":
                if trial_num >= self.num_trials / 2:
                    continue
            elif half == "second":
                if trial_num < self.num_trials / 2:
                    continue

            probs = [trial['P'], trial['C'], trial['B']]

            slots = trial['slots_pre']
            slots_arr = np.array([slots['p'], slots['c'], slots['b']])
            if self.experiment != 4:
                targets = np.array([7, 7, 7])
                max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
            else:
                targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])
                max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]

            progress_arr = slots_arr / targets

            max_slot_token = ['p', 'c', 'b'][np.argmax(progress_arr)]

            if np.max(slots_arr) == 0:
                continue

            if diverge:
                if max_slot_token == max_prob_token:
                    continue
            else:
                if max_slot_token != max_prob_token:
                    continue

            cardselect = trial['cardselect'][0].lower()
            counts += 1

            if cardselect == max_slot_token:
                retro_values += 1

        if counts == 0:
            return 0
        else:
            return retro_values / counts

    def get_retro_valuation_per_count(self, half=None, diverge=True):
        retro_values = np.zeros((self.num_conditions, 7))
        counts = np.zeros((self.num_conditions, 7))

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
            for trial_num in range(self.num_trials):
                if half == "first":
                    if trial_num >= self.num_trials / 2:
                        continue
                elif half == "second":
                    if trial_num < self.num_trials / 2:
                        continue
                trial = self.blocks[block_num]['trial_' + str(trial_num)]
                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]

                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])
                if self.experiment != 4:
                    trial['P_T'] = 7
                    trial['C_T'] = 7
                    trial['B_T'] = 7

                progress = {'p': slots['p'] / trial['P_T'],
                            'c': slots['c'] / trial['C_T'],
                            'b': slots['b'] / trial['B_T']}
                targets = np.array(
                    [trial['P_T'], trial['C_T'], trial['B_T']])
                max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                max_slot_token = ['p', 'c', 'b'][
                    np.argmax(list(progress.values()))]
                next_best_token = ['p', 'c', 'b'][find_second_highest_index(list(progress.values()))]

                if np.max(slots_arr) == 0:
                    continue

                if diverge:
                    if max_slot_token == max_prob_token:
                        continue
                else:
                    if max_slot_token != max_prob_token:
                        continue

                cardselect = trial['cardselect'][0].lower()

                if diverge:
                    progress_diff = int(np.floor((progress[max_slot_token] - progress[max_prob_token])/0.14))
                else:
                    progress_diff = int(np.floor((progress[max_slot_token] - progress[next_best_token])/0.14))

                counts[condition][progress_diff] += 1

                if cardselect == max_slot_token:
                    retro_values[condition][progress_diff] += 1

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
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                if self.experiment != 4:
                    max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                    max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]

                else:
                    progress = {'p': slots['p'] / trial['P_T'],
                                'c': slots['c'] / trial['C_T'],
                                'b': slots['b'] / trial['B_T']}
                    targets = np.array(
                        [trial['P_T'], trial['C_T'], trial['B_T']])
                    max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                    max_slot_token = ['p', 'c', 'b'][
                        np.argmax(list(progress.values()))]

                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == max_prob_token:
                    continue


                cardselect = trial['cardselect'][0].lower()

                if prev_cardselect == max_prob_token:
                    prob_counts[condition][prev_reward] += 1
                    if cardselect != max_prob_token:
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

                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]
                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                if self.experiment != 4:
                    max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
                    max_slot_token = ['p', 'c', 'b'][np.argmax(slots_arr)]
                    if slots[max_slot_token] - slots[max_prob_token] <= 3:
                        progress = 0
                    else:
                        progress = 1

                else:
                    progress = {'p': slots['p'] / trial['P_T'],
                                'c': slots['c'] / trial['C_T'],
                                'b': slots['b'] / trial['B_T']}
                    targets = np.array(
                        [trial['P_T'], trial['C_T'], trial['B_T']])
                    max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                    max_slot_token = ['p', 'c', 'b'][
                        np.argmax(list(progress.values()))]

                    if progress[max_slot_token] - progress[max_prob_token] <= 0.5:
                        progress = 0
                    else:
                        progress = 1



                if np.max(slots_arr) == 0:
                    continue

                if max_slot_token == max_prob_token:
                    continue

                cardselect = trial['cardselect'][0].lower()

                if prev_cardselect == max_prob_token:
                    prob_counts[progress][prev_reward] += 1
                    if cardselect != max_prob_token:
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


    def get_condition_max_action(self, block_num, half=None):
        counts = 0
        actions = 0

        block = self.blocks[str(block_num)]

        for trial_num in range(self.num_trials):
            if half == "first":
                if trial_num >= self.num_trials / 2:
                    continue
            elif half == "second":
                if trial_num < self.num_trials / 2:
                    continue
            trial = block['trial_' + str(trial_num)]
            select = trial['cardselect'][0].lower()
            probs = np.array([trial['P'], trial['C'], trial['B']])
            if self.experiment != 4:
                targets = np.array([7, 7, 7])
            else:
                targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])
            max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
            counts += 1
            if select == max_prob_token:
                actions += 1
        return actions / counts


    def get_condition_max_action_blockwise(self):

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))

        counts = np.zeros(len(block_keys))
        actions = np.zeros(len(block_keys))

        for block_num in block_keys:
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block['trial_' + str(trial_num)]
                select = trial['cardselect'][0].lower()
                probs = np.array([trial['P'], trial['C'], trial['B']])
                if self.experiment != 4:
                    targets = np.array([7, 7, 7])
                else:
                    targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])
                max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                counts[int(block_num)] += 1
                if select == max_prob_token:
                    actions[int(block_num)] += 1
        return actions / counts


    def get_optimal_goal_condition(self):
        goalselects = np.zeros((self.num_conditions, 3, 3))
        counts = np.zeros((self.num_conditions, 3))

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block["trial_" + str(trial_num)]
                condition = trial["condition"]
                probs = np.array([trial['P'], trial['C'], trial['B']])
                if self.experiment != 4:
                    targets = np.array([7, 7, 7])
                else:
                    targets = np.array(
                        [trial['P_T'], trial['C_T'], trial['B_T']])
                max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                goalselect = trial['cardselect'][0].lower()

                goal_condition = ['p', 'c', 'b'].index(max_prob_token)

                counts[condition][goal_condition] += 1
                goalindex = ['p', 'c', 'b'].index(goalselect)

                goalselects[condition][goal_condition][goalindex] += 1

        #goalselects = goalselects / counts[:, None]

        return goalselects, counts

    def get_goal_action_congruence(self):
        congruence = 0
        counts = 0
        prev_probe = -1

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block["trial_" + str(trial_num)]
                cardflip = trial["cardflip"][0].lower()
                if prev_probe != -1:
                    counts += 1
                    if cardflip == prev_probe:
                        congruence += 1
                    prev_probe = -1

                if "probe_keypress" in trial:
                    prev_probe = GOAL_PROBE[trial["probe_keypress"]]

        return congruence / counts


    def get_goal_switches_around_probes(self):
        all_switches = 0
        probe_switches = 0
        prev_action = -1

        block_keys = list(self.blocks.keys())
        block_keys = sorted(block_keys, key=lambda x: int(x))
        for block_num in block_keys:
            block = self.blocks[str(block_num)]
            for trial_num in range(self.num_trials):
                trial = block["trial_" + str(trial_num)]
                cardflip = trial["cardflip"][0].lower()
                if prev_action != -1:
                    if cardflip != prev_action:
                        all_switches += 1
                        if "probe_keypress" in trial:
                            probe_switches += 1
                prev_action = cardflip

        return probe_switches / all_switches

    def get_goal_switches_across_blocks(self, block_num):

        block = self.blocks[str(block_num)]
        if 'targets' not in block:
            return 0
        targets = block['targets']
        switches = get_goal_switches(targets)
        switches_sub = switches

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

    def get_goal_valuation(self, measure):
        max_prob = np.zeros(self.num_conditions)
        max_slot = np.zeros(self.num_conditions)
        other = np.zeros(self.num_conditions)
        counts = np.zeros(self.num_conditions)
        counts_all = np.zeros(self.num_conditions)
        for block_num in range(len(self.blocks)):
            for trial_num in range(self.num_trials):
                trial = self.blocks[str(block_num)]['trial_' + str(trial_num)]
                condition = trial['condition']
                probs = [trial['P'], trial['C'], trial['B']]

                slots = trial['slots_pre']
                slots_arr = np.array([slots['p'], slots['c'], slots['b']])

                if self.experiment != 4:
                    targets = np.array([7, 7, 7])
                else:
                    targets = np.array([trial['P_T'], trial['C_T'], trial['B_T']])
                max_prob_token = ['p', 'c', 'b'][np.argmin(targets / probs)]
                progress = {'p': slots['p'] / targets[0],
                            'c': slots['c'] / targets[1],
                            'b': slots['b'] / targets[2]}
                max_slot_token = ['p', 'c', 'b'][np.argmax(list(progress.values()))]
                next_best_token = ['p', 'c', 'b'][
                    find_second_highest_index(list(progress.values()))]

                # if np.max(slots_arr) == 0:
                #     continue

                counts_all[condition] += 1

                if measure == "maxprogress_dominant_diverge":
                    if max_prob_token == max_slot_token:
                        continue
                    cardselect = trial['cardselect'][0].lower()
                    counts[condition] += 1

                    if cardselect == max_prob_token:
                        max_prob[condition] += 1
                    elif cardselect == max_slot_token:
                        max_slot[condition] += 1
                    else:
                        other[condition] += 1
                else:
                    if max_prob_token != max_slot_token:
                        continue
                    cardselect = trial['cardselect'][0].lower()
                    counts[condition] += 1

                    if cardselect == max_prob_token:
                        max_prob[condition] += 1
                    elif cardselect == next_best_token:
                        max_slot[condition] += 1
                    else:
                        other[condition] += 1


        return max_prob / counts, max_slot / counts, other / counts, counts/ counts_all


    def get_model_derived_retrospective_choice(self):
        pass


    def get_individual_measure(self, measure_name):

        if measure_name == "performance":
            measure = self.get_task_performance()
        elif measure_name == "retro_value_count":
            measure = self.get_retro_valuation_per_count()
        elif measure_name == "retro_value_count_first":
            measure = self.get_retro_valuation_per_count(half='first')
        elif measure_name == "retro_value_count_second":
            measure = self.get_retro_valuation_per_count(half='second')
        elif measure_name == "obvious_choice_count":
            measure = self.get_retro_valuation_per_count(diverge=False)
        elif measure_name == "goal_valuation":
            measure = self.get_goal_valuation()
        elif measure_name == "retrospective_choice_model":
            measure = self.get_model_derived_retrospective_choice()
        elif measure_name == "goal_action_congruence":
            measure = self.get_goal_action_congruence()
        elif measure_name == "switches_around_probes":
            measure = self.get_goal_switches_around_probes()
        elif measure_name == "condition_action_blockwise":
            measure = self.get_condition_max_action_blockwise()


        return measure



