from scipy.special import softmax
import numpy as np
import json

import sys
sys.path.append("..")

from behavior import *
from generate import *


class Model:
    def __init__(self, params, num_slots=7, target=7):
        self.experiment = None
        self.reward = 10
        self.num_slots = num_slots

        self.targets = {
            'p': target,
            'c': target,
            'b': target
        }

        self.params = params
        self.alpha = params[0]
        self.alpha_c = params[1]
        self.beta_0 = params[2]
        self.beta_g = params[3]
        self.beta_a = params[4]

        self.block = None
        self.prev_goal = ''

        self.reset_card_probs()

    def reset_slots(self):
        self.slots = {'p': 0, 'c': 0, 'b': 0}

    def reset_card_probs(self):
        self.C = {'p': 0.33, 'c': 0.33, 'b': 0.33}

    def set_targets(self, targets):
        self.targets['p'] = targets[0]
        self.targets['c'] = targets[1]
        self.targets['b'] = targets[2]

    def update_card_probs(self, card, flip):
        card_type = card[0].lower()

        self.C[card_type] += self.alpha_c * (1 - self.C[card_type])

        for token in ['p', 'c', 'b']:
            if token != card_type:
                self.C[token] += self.alpha_c * (0 - self.C[token])

    def update_params(self, token, flip):
        pass

    def get_slot_tokens(self, flip):
        tokens = []
        for con in ['p', 'c', 'b']:
            if con == flip:
                continue
            if self.slots[con] > 0:
                tokens.append(con)
        return tokens

    def check_slot_status(self, slots):
        if slots['p'] == self.targets['p']:
            return 1, 'p'
        elif slots['c'] == self.targets['c']:
            return 1, 'c'
        elif slots['b'] == self.targets['b']:
            return 1, 'b'

        return 0, -1

    def calculate_qvals_goals(self):
        pass

    def choose_current_goal(self):
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]))

        if (self.prev_goal == '') or (self.prev_goal == 'u'):
            goal = np.random.choice(['p', 'c', 'b'], p=q_probs)
            return goal, -1, q_probs

        q_cons = np.array([q_vals[token] for token in ['p', 'c', 'b'] if token != self.prev_goal])

        pstay = 1. / (1 + np.exp(self.beta_0 - self.C[self.prev_goal]  - self.beta_g * (q_vals[self.prev_goal] - np.max(q_cons))))

        switch_action_probs = [pstay, 1 - pstay]
        switch_action = np.random.choice([0, 1], p=switch_action_probs)

        if switch_action == 0:
            goal = self.prev_goal
        else:
            action_probs = softmax(self.beta_g * q_cons)
            goal = np.random.choice([token for token in ['p', 'c', 'b'] if token != self.prev_goal],
                                            p=action_probs)

        return goal, switch_action, switch_action_probs

    def choose_card_action(self, cards):
        q_vals = self.calculate_qvals_goals()

        if (self.prev_goal == '') or (self.prev_goal == 'u'):
            action_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]))

            goal = np.random.choice(['p', 'c', 'b'], p=action_probs)
        else:
            goal = self.prev_goal

        tokens = [cards[0][0].lower(), cards[1][0].lower(), cards[2][0].lower()]

        q_cons = np.array([q_vals[token] for token in ['p', 'c', 'b'] if token != goal])

        p_stay = 1. / (1 + np.exp(self.beta_0 - self.C[goal] - self.beta_a * (q_vals[goal] - np.max(q_cons))))

        switch_action_probs = [p_stay, 1 - p_stay]
        switch_action = np.random.choice([0, 1], p=switch_action_probs)

        if switch_action == 0:
            action = tokens.index(goal)
        else:

            cons = [token for token in ['p', 'c', 'b'] if token != goal]
            q_cons = np.array([q_vals[token] for token in cons])
            action_probs = softmax(self.beta_a * q_cons)
            action_token = np.random.choice(cons, p=action_probs)
            action = tokens.index(action_token)

        return action, switch_action, switch_action_probs

    def run_subject_action_trial(self, trial, probe=False):

        cards = [trial['37'], trial['39'], trial['38']]
        self.slots = trial['slots_pre']
        self.block = trial['condition']

        action,  switch_action, action_switch_probs = self.choose_card_action(cards)
        cardflip = trial['cardflip']
        flip = trial['current_token']

        if cardflip.lower()[0] == self.prev_goal:
            switch_action_prob = action_switch_probs[0]
        else:
            switch_action_prob = action_switch_probs[1]


        if "probe_keypress" in trial.keys():
            goal, goal_switch_action, goal_switch_probs = self.choose_current_goal()

            goal_sub = get_goal_from_probe(trial["probe_keypress"])

            if self.prev_goal == '':
                if goal_sub == "u":
                    goal_switch_prob = -1
                else:
                    goal_switch_prob = goal_switch_probs[['p', 'c', 'b'].index(goal_sub)]
            else:
                if goal_sub == self.prev_goal:
                    goal_switch_prob = goal_switch_probs[0]
                else:
                    goal_switch_prob = goal_switch_probs[1]

            self.prev_goal = goal_sub
        else:
            goal_switch_prob = -1

        self.update_card_probs(cardflip, flip)

        if goal_switch_prob == -1:
            return switch_action_prob
        else:
            return switch_action_prob * goal_switch_prob

    def run_experiment(self, trials, seed=100):
        # np.random.seed(seed)
        reward_trials = []
        goals = []
        count = -1
        trials_res = []

        for trial_num in range(len(trials)):

            trial_data = trials[trial_num]
            trial = trial_data.copy()

            self.trial_num = trial_num

            cards = [trial['37'], trial['39'], trial['38']]
            trial['slots_pre'] = self.slots.copy()

            self.slots_pre = self.slots.copy()
            self.block = trial['condition']

            # choose goal and action
            action, switch_action, switch_action_probs = self.choose_card_action(cards)

            trial['action_probs'] = switch_action_probs
            flip = trial[cards[action] + "_token"]
            trial['current_token'] = flip
            trial["cardselect"] = cards[action]
            trial["cardflip"] = trial[cards[action] + "_card"]

            self.flip = flip
            self.select = cards[action].lower()[0]
            self.update_params(self.select, flip)

            if flip != 'e':
                # choose slot action
                self.slots[flip] += 1
                trial['slots'] = self.slots.copy()

            self.update_card_probs(trial[cards[action] + "_card"], flip)
            status, token = self.check_slot_status(self.slots)
            trial['slot_status'] = status

            count += 1

            if status == 1:
                self.slots[token] = 0
                reward_trials.append(count)
                goals.append(token)

            if (trial_num + 1) % 3 == 0:
                goal, goal_switch_action, goal_switch_probs = self.choose_current_goal()

                self.prev_goal = goal
                trial['probe_keypress'] = get_probe_from_goal(self.prev_goal)

            trials_res.append(trial)

        return trials_res, goals, reward_trials

    def run_task_block(self, block_trials, seed=100):
        block_trials_res = {"GoalSwitching": {}}
        blocks = block_trials_res["GoalSwitching"]
        self.reset_slots()
        self.reset_card_probs()
        for block_num in range(len(block_trials)):

            if self.experiment != "normative":
                self.reset_card_probs()
                trials = block_trials[block_num]['trials']
            else:
                trials = block_trials[block_num]

            blocks[str(block_num)] = {}


            targets = [trials[0]['P_T'], trials[0]['C_T'], trials[0]['B_T']]
            self.set_targets(targets)

            self.P = np.max([trials[0]['P'], trials[0]['C'], trials[0]['B']])

            trials_res, targets, reward_trials = self.run_experiment(trials, seed=seed)

            for trial_num in range(len(trials_res)):
                blocks[str(block_num)]["trial_" + str(trial_num)] = trials_res[trial_num]

            blocks[str(block_num)]["targets"] = targets
            blocks[str(block_num)]["reward_trials"] = reward_trials
        return block_trials_res

    def run_model(self, experiment, episodes=None, seed=100):
        if experiment == 1:
            self.experiment = 1
            with open('generate/json/trials_exp_1.json', 'r') as file:
                episodes = json.load(file)
        elif experiment == 2:
            self.experiment = 2
            with open('generate/json/trials_exp_2.json', 'r') as file:
                episodes = json.load(file)
        elif experiment == "normative":
            self.experiment = "normative"

        model_res = self.run_task_block(episodes, seed=seed)

        return model_res


