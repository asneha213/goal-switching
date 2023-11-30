from .model import Model
from scipy.special import softmax
import numpy as np

import sys
sys.path.append("..")

from behavior import *


class Prospective(Model):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.gamma = params[1]
        self.beta_g = params[2]
        self.beta_0g = params[3]

        if len(params) > 4:
            self.beta_a = params[4]
        self.beta_0a = params[3]

        self.rho = 0
        self.prev_goal = ''
        self.reset_card_probs()

    def reset_card_probs(self):

        self.M = {
            'p': 0.33,
            'c': 0.33,
            'b': 0.33
        }

    def update_card_probs(self, card, flip):
        card_type = card[0].lower()

        if flip != 'e':
            self.M[card_type] +=  self.alpha * (1 - self.M[card_type])
            self.rho += self.alpha * (1 - self.rho)
        else:
            self.M[card_type] +=  self.alpha * (0 - self.M[card_type])
            self.rho += self.alpha * (0 - self.rho)


    def calculate_goal_value_recursion(self, token, p, count, k):
        if count == self.targets[token]:
            return self.reward
        if k > 10:
            return 0
        q = ( p * self.gamma * self.calculate_goal_value_recursion(token, p, count + 1, k+1) )\
           / (1 - (1 - p) * self.gamma)

        return q

    def calculate_qvals_goals(self):
        qvals = {}

        probs = [self.M['p'], self.M['c'], self.M['b']]

        #calculate goal values
        qvals['p'] = self.calculate_goal_value_recursion('p', probs[0], count=self.slots['p'], k=0)
        qvals['c'] = self.calculate_goal_value_recursion('c', probs[1], count=self.slots['c'], k=0)
        qvals['b'] = self.calculate_goal_value_recursion('b', probs[2], count=self.slots['b'], k=0)

        #print(qvals)
        return qvals

    def choose_current_goal(self):
        q_vals = self.calculate_qvals_goals()
        q_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]))

        if (self.prev_goal == '') or (self.prev_goal == 'u'):
            goal = np.random.choice(['p', 'c', 'b'], p=q_probs)
            return goal, -1, q_probs

        q_cons = np.array([q_vals[token] for token in ['p', 'c', 'b'] if token != self.prev_goal])

        pstay = 1. / (1 + np.exp(self.beta_0g - self.beta_g * (q_vals[self.prev_goal] - np.max(q_cons))))

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

        #p_stay = 1. / (1 + np.exp(self.beta_0a - self.beta_a * (self.M[goal] - self.rho)))
        p_stay = 1. / (1 + np.exp(self.beta_0a - self.beta_a * (q_vals[goal] - np.max(q_cons))))


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

        action, switch_action, action_switch_probs = self.choose_card_action(cards)
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
            action, switch_action, switch_action_probs = self.choose_card_action(cards)

            self.prev_action = cards[action].lower()[0]

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

            goal, goal_switch_action, goal_switch_probs = self.choose_current_goal()

            self.prev_goal = goal

            if (trial_num + 1) % 3 == 0:
                trial['probe_keypress'] = get_probe_from_goal(self.prev_goal)

            trials_res.append(trial)

        return trials_res, goals, reward_trials




