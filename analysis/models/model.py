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
            'p' : target,
            'c': target,
            'b': target
        }

        self.params = params
        self.prev_action = ''
        self.beta = 1
        self.beta_0 = None
        self.P = None

    def reset_slots(self):
        self.slots = {'p': 0, 'c': 0, 'b': 0}

    def reset_card_probs(self):
        pass

    def set_targets(self, targets):
        self.targets['p'] = targets[0]
        self.targets['c'] = targets[1]
        self.targets['b'] = targets[2]

    def set_card_probs(self, probs):
        pass

    def update_card_probs(self, card, flip):
        pass

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
            self.prev_action = ""
            return 1, 'p'
        elif slots['c'] == self.targets['c']:
            self.prev_action = ""
            return 1, 'c'
        elif slots['b'] == self.targets['b']:
            self.prev_action = ""
            return 1, 'b'

        return 0, -1

    def calculate_qvals_goals(self):
        pass

    def choose_card_action_0(self, cards):
        token0 = cards[0][0].lower()
        token1 = cards[1][0].lower()
        token2 = cards[2][0].lower()

        q_vals = self.calculate_qvals_goals()

        vals = np.array([q_vals[token0], q_vals[token1], q_vals[token2]])

        action_probs = softmax(self.beta * vals)

        action = np.random.choice([0, 1, 2], p=action_probs)

        return action, -1, action_probs, q_vals

    def run_subject_action_trial_0(self, trial, probe=False):

        cards = [trial['37'], trial['39'], trial['38']]
        self.slots = trial['slots_pre']
        action, switch_action, action_probs, q_vals = self.choose_card_action(cards)
        cardflip = trial['cardflip']
        flip = trial['current_token']
        sub_action = cards.index(cardflip)
        self.update_card_probs(cardflip, flip)
        card_action_prob = action_probs[sub_action]
        return card_action_prob

    def choose_card_action(self, cards):

        tokens = [cards[0][0].lower(), cards[1][0].lower(), cards[2][0].lower()]
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(self.beta * np.array([q_vals[token] for token in tokens]))

        if self.prev_action == '':
            action = np.random.choice([0, 1, 2], p=q_probs)
            return action, -1, [0.5, 0.5], q_probs

        q_counter = np.max([q_vals[token] for token in ['p', 'c', 'b'] if token!=self.prev_action])

        action_probs = softmax(
            self.beta * np.array([q_vals[token] for token in ['p', 'c', 'b'] if token != self.prev_action]))

        pstay = 1 / (1 + np.exp(self.beta_0 - self.beta * (q_vals[self.prev_action] - q_counter)))
        pswitch = 1 - pstay

        switch_action_probs = [pstay, pswitch]
        switch_action = np.random.choice([0, 1], p=switch_action_probs)

        if switch_action == 0:
            action = tokens.index(self.prev_action)
        else:
            action_token = np.random.choice([token for token in ['p', 'c', 'b'] if token!=self.prev_action], p=action_probs)
            action = tokens.index(action_token)

        return action, switch_action, switch_action_probs, action_probs


    def run_subject_action_trial(self, trial, probe=False):

        cards = [trial['37'], trial['39'], trial['38']]
        self.slots = trial['slots_pre']
        action, switch_action, switch_action_probs, action_probs = self.choose_card_action(cards)
        cardflip = trial['cardflip']
        flip = trial['current_token']
        if switch_action != -1:
            if cardflip.lower()[0] == self.prev_action:
                switch_action_prob = switch_action_probs[0]
                action_prob = 1.0

            else:
                switch_action_prob = switch_action_probs[1]
                action_prob = (action_probs[['p', 'c', 'b'].index(cardflip.lower()[0]) - 1])
        else:
            sub_action = cards.index(cardflip)
            switch_action_prob = action_probs[sub_action]

        self.prev_action = cardflip.lower()[0]

        if "probe_keypress" in trial.keys():
            action, switch_action, switch_action_probs, action_probs = self.choose_card_action(cards)
            goal_sub = get_goal_from_probe(trial["probe_keypress"])
            if goal_sub == self.prev_action:
                switch_action_prob *= switch_action_probs[0]
            else:
                switch_action_prob *= switch_action_probs[1]

        self.update_card_probs(cardflip, flip)
        return switch_action_prob

    def run_experiment(self, trials, seed=100):
        #np.random.seed(seed)
        reward_trials = []
        goals = []
        count = -1
        trials_res = []

        for trial_num in range(len(trials)):

            trial_data = trials[trial_num]
            trial = trial_data.copy()

            cards = [trial['37'], trial['39'], trial['38']]
            trial['slots_pre'] = self.slots.copy()

            self.slots_pre = self.slots.copy()

            # choose goal and action
            action, switch_action, switch_action_probs, action_probs = self.choose_card_action(cards)

            #print(action_probs)

            self.prev_action = cards[action].lower()[0]

            #print(self.prev_action, action)

            if (trial_num + 1) % 3 == 0:
                trial['probe_keypress'] = self.prev_action

            trial['action_probs'] = action_probs
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
            trials_res.append(trial)

            if status == 1:
                self.slots[token] = 0
                reward_trials.append(count)
                goals.append(token)

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
