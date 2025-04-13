from scipy.special import softmax
import numpy as np
import json

import sys
sys.path.append("..")

from behavior import *
from generate import *


class ModelJustGoal:
    def __init__(self, params, num_slots=7, target=7):
        """
        Set up model parameters
        """
        self.experiment = None
        self.reward = 10
        self.num_slots = num_slots

        self.targets = {
            'p': target,
            'c': target,
            'b': target
        }

        self.params = params
        self.block = None
        self.prev_goal = ''
        self.q_vals = None

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
        """
        Method to update card probabilities after a card is flipped
        """
        self.alpha_c = self.params['alpha_c']
        card_type = card[0].lower()
        
        # Increase goal perseveration upon choosing the goal
        self.C[card_type] += self.alpha_c * (1 - self.C[card_type])
        #
        for token in ['p', 'c', 'b']:
            if token != card_type:
                self.C[token] += self.alpha_c * (0 - self.C[token])
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
            return 1, 'p'
        elif slots['c'] == self.targets['c']:
            return 1, 'c'
        elif slots['b'] == self.targets['b']:
            return 1, 'b'

        return 0, -1

    def calculate_qvals_goals(self):
        return {'p': 0.33, 'c': 0.33, 'b': 0.33}



    def choose_card_action(self, cards):
        """
        Choose the card to flip in a given round given a goal
        """
        self.beta_g = self.params['beta_g']
        self.beta_c = self.params['beta_c']

        q_vals = self.calculate_qvals_goals()

        tokens = [cards[0][0].lower(), cards[1][0].lower(), cards[2][0].lower()]
        q_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]) + \
            self.beta_c * np.array([self.C['p'], self.C['c'], self.C['b']]))
        #q_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]))
        q_probs_dict = {tokens[i]: q_probs[i] for i in range(len(tokens))}

        action = np.random.choice(tokens, p=q_probs)
        return action, q_probs_dict


    def run_subject_action_trial(self, trial, probe=False, get_action=False):

        cards = [trial['37'], trial['39'], trial['38']]
        self.slots = trial['slots_pre']
        self.block = trial['condition']

        # Get conditional probabilities of choosing the subject action given different goals
        cardflip = trial['cardflip']
        action_sub = cardflip.lower()[0]

        action_m, action_selection_probs = self.choose_card_action(cards)

        action_selection_prob = action_selection_probs[action_sub]

        flip = trial['current_token']
        self.update_card_probs(cardflip, flip)

        return action_selection_prob, -1


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

            # choose current action flip
            action_token, action_selection_probs = self.choose_card_action(cards)
            tokens = [cards[0][0].lower(), cards[1][0].lower(), cards[2][0].lower()]
            action_ind = tokens.index(action_token)

            trial['action_probs'] = action_selection_probs
            flip = trial[cards[action_ind] + "_token"]
            trial['current_token'] = flip

            trial["cardselect"] = cards[action_ind]
            trial["cardflip"] = cards[action_ind]

            self.flip = flip
            self.select = cards[action_ind].lower()[0]

            if flip != 'e':
                # choose slot action
                self.slots[flip] += 1
                trial['slots'] = self.slots.copy()

            self.update_card_probs(trial[cards[action_ind] + "_card"], flip)
            status, token = self.check_slot_status(self.slots)
            trial['slot_status'] = status

            count += 1

            if status == 1:
                self.slots[token] = 0
                reward_trials.append(count)
                goals.append(token)


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
        self.experiment = experiment

        exp_file = 'generate/json/trials_exp_' + str(experiment) + '.json'
        with open(exp_file, 'r') as file:
            episodes = json.load(file)

        model_res = self.run_task_block(episodes, seed=seed)

        return model_res


