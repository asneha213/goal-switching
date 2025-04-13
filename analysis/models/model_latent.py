from scipy.special import softmax
import numpy as np
import json

import sys
sys.path.append("..")

from behavior import *
from generate import *


class Model:
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
        ## store conditional probabilities of choosing a goal given the previous goal at different timepoints
        self.prev_goal_cond_probs = {
                0: {'p': 0.33, 'c': 0.33, 'b': 0.33},
                1: {'p': 0.33, 'c': 0.33, 'b': 0.33},
                2: {'p': 0.33, 'c': 0.33, 'b': 0.33},
        }
        self.time_point = -1
        self.prev_goal_selection_probs = {'p': 0.33, 'c': 0.33, 'b': 0.33}
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
        return {'p': 0.33, 'c': 0.33, 'b': 0.33}

    def choose_goal(self, prev_goal):
        """
        Chosen the goal in a given round
        """
        self.beta_g = self.params['beta_g']
        self.beta_0 = self.params['beta_0']

        q_vals = self.calculate_qvals_goals()

        q_probs = softmax(self.beta_g * np.array([q_vals['p'], q_vals['c'], q_vals['b']]))

        goal_select_probs = {'p': 0, 'c': 0, 'b': 0}

        # If it's the start of the game choose randomly

        if (prev_goal == '') or (prev_goal == 'u'):
            goal = np.random.choice(['p', 'c', 'b'], p=q_probs)
            return goal, {'p': q_probs[0], 'c': q_probs[1], 'b': q_probs[2]}


        # get the other two options alternate to the previously chosen goal
        cons = [token for token in ['p', 'c', 'b'] if token != prev_goal]


        # Get the value of the best alternative goal
        q_cons = np.array([q_vals[token] for token in cons])


        # Get the stay probability given the advantage of the current goal over the best alternative
        pstay = 1. / (1 + np.exp(self.beta_0 - self.C[prev_goal] - self.beta_g * (q_vals[prev_goal] - np.max(q_cons))))


        # probability of staying with the previously chosen goal
        goal_select_probs[prev_goal] = pstay

        q_cons_probs = softmax(self.beta_g * q_cons)

        # probability of choosing one of the other two goals: probability of choosing the goal * probability of not staying
        for i, token in enumerate(cons):
            goal_select_probs[token] = q_cons_probs[i] * (1 - pstay)

        goal = np.random.choice(['p', 'c', 'b'], p=[goal_select_probs['p'], goal_select_probs['c'], goal_select_probs['b']])

        return goal, goal_select_probs

    def choose_card_action(self, cards, goal):
        """
        Choose the card to flip in a given round given a goal
        """
        self.beta_a = self.params['beta_a']
        self.beta_0 = self.params['beta_0']

        if goal == "":
            return np.random.choice(['p', 'c', 'b']), {'p': 0.33, 'c': 0.33, 'b': 0.33}

        q_vals = self.calculate_qvals_goals()

        action_select_probs = {'p': 0, 'c': 0, 'b': 0}

        # get the values of other two goal alternate to the current goal
        cons = [token for token in ['p', 'c', 'b'] if token != goal]
        q_cons = np.array([q_vals[token] for token in ['p', 'c', 'b'] if token != goal])


        # probability of staying with the current goal
        p_stay = 1. / (1 + np.exp(self.beta_0 - self.C[goal] - self.beta_a * (q_vals[goal] - np.max(q_cons))))


        action_select_probs[goal] = p_stay

        # probability of choosing one of the other two goals: probability of choosing the goal * probability of not staying
        for i, token in enumerate(cons):
            action_select_probs[token] = softmax(self.beta_a * q_cons)[i] * (1 - p_stay)

        action_ind = np.random.choice([0, 1, 2], p=[action_select_probs['p'], action_select_probs['c'], action_select_probs['b']])
        action_token = ['p', 'c', 'b'][action_ind]

        return action_token, action_select_probs



    def get_goal_selection_probs_at_timepoint(self):
        """
        Get the probability of choosing a goal given the previous goal
        """

        if self.time_point == 0:
            goal_probs_t = {'p': 0, 'c': 0, 'b': 0}
            goal_probs_t[self.prev_goal] = 1
            self.prev_goal_cond_probs[0] = goal_probs_t
            return goal_probs_t

        ## If the timepoint is the goal probe round, return deterministic probabilities of choosing the goal (given by the participant)

        if self.time_point == 1:
            goal_t_1, goal_probs_t_1_cond = self.choose_goal(self.prev_goal)
            self.prev_goal_cond_probs[1] = goal_probs_t_1_cond

            ## If the timepoint is 1, return sum(Pr(goal_at_timepoint_1 | goal_at_timepoint_0)) for all goals
            return goal_probs_t_1_cond


        if self.time_point == 2:
            goal_probs_t_2_cond = {'p': 0, 'c': 0, 'b': 0}

            for goal_2 in ['p', 'c', 'b']:
                for goal_1 in ['p', 'c', 'b']:
                    goal_t_2, goal_probs_t_2 = self.choose_goal(goal_1)
                    goal_probs_t_2_cond[goal_2] += goal_probs_t_2[goal_2] * self.prev_goal_cond_probs[1][goal_1]


            self.prev_goal_cond_probs[2] = goal_probs_t_2_cond
            return goal_probs_t_2_cond

        ## If the timepoint is 2, return sum(Pr(goal_at_timepoint_2 | goal_at_timepoint_1)) for all goals

        if self.time_point == 3:
            goal_probs_t_3_cond = {'p': 0, 'c': 0, 'b': 0}

            for goal_3 in ['p', 'c', 'b']:
                for goal_2 in ['p', 'c', 'b']:
                    goal_t_3, goal_probs_t_3 = self.choose_goal(goal_2)
                    goal_probs_t_3_cond[goal_3] += goal_probs_t_3[goal_3] * self.prev_goal_cond_probs[2][goal_2]

            return goal_probs_t_3_cond





    def run_subject_action_trial(self, trial, probe=False, get_action=False):
        """
        Get goal and action selection probabilities for a given trial
        """

        cards = [trial['37'], trial['39'], trial['38']]

        ## Update the slot configuration and condition as seen by the participants
        self.slots = trial['slots_pre']
        self.block = trial['condition']

        cardflip = trial['cardflip']
        action_sub = cardflip.lower()[0]


        # self.time_point tracks the time since the last goal probe
        # If time_point >=3, reset to 0
        if self.time_point >= 4:
            self.time_point = 0

        # If it is not the starting round, update the prev_goal_selection_probs to get the conditional probabilities in the current round
        if self.time_point != -1:
            self.prev_goal_selection_probs = self.get_goal_selection_probs_at_timepoint()

        # Get conditional probabilities of choosing the action that stays with the current goal integrated over all goals
        # Pr(action|goal) = sum(Pr(action_t|goal_{t-1}) * Pr(goal_{t-1})) for all goals

        action_cond_probs = {'p': 0, 'c': 0, 'b': 0}

        for action in ['p', 'c', 'b']:
            for goal in ['p', 'c', 'b']:
                action_token, action_selection_probs = self.choose_card_action(cards, goal)
                action_cond_probs[action] += action_selection_probs[action] * self.prev_goal_selection_probs[goal]

        action_selection_prob = action_cond_probs[action_sub]

        # If it is the starting round, set the prev_goal to the action chosen in the first round
        if self.time_point == -1:
            self.prev_goal = action_sub

        # Get the action prescribed by the model (for the model_measures script, to get prescribed prospective action)
        if get_action:
            q_vals = self.calculate_qvals_goals()
            goal_max = max(q_vals, key=q_vals.get)
            return goal_max, cardflip

        flip = trial['current_token']
        self.update_card_probs(cardflip, flip)

        if "probe_keypress" in trial.keys():
            # choose goal probe after outcome of the round
            self.time_point += 1
            self.slots = trial['slots']
            goal_sub = get_goal_from_probe(trial["probe_keypress"])
            # Get conditional probabilities of choosing the current goal given the last provided goal probe and the timepoint since the last goal probe
            goal_selection_probs = self.get_goal_selection_probs_at_timepoint()
            # If goal probe is not reported, chose the last chosen goal probe as the current reported goal
            if goal_sub == 'u':
                goal_sub = self.prev_goal
            goal_selection_prob = goal_selection_probs[goal_sub]
            self.prev_goal = goal_sub
            self.time_point = 0
            self.prev_goal_cond_probs = {
                    0: {'p': 0.33, 'c': 0.33, 'b': 0.33},
                    1: {'p': 0.33, 'c': 0.33, 'b': 0.33},
                    2: {'p': 0.33, 'c': 0.33, 'b': 0.33},
            }
            
        else:
            self.time_point += 1
            goal_selection_prob = -1
        

        if goal_selection_prob == -1:
            return action_selection_prob, -1
        else:
            return action_selection_prob, goal_selection_prob



    def run_experiment(self, trials, seed=100):
        """
        Run simulations of the model choices in an experimental block with the model parameters
        """
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
            action_token, action_selection_probs = self.choose_card_action(cards, self.prev_goal)
            tokens = [cards[0][0].lower(), cards[1][0].lower(), cards[2][0].lower()]
            action_ind = tokens.index(action_token)

            trial['action_probs'] = action_selection_probs
            flip = trial[cards[action_ind] + "_token"]
            trial['current_token'] = flip

            trial["cardselect"] = cards[action_ind]
            trial["cardflip"] = cards[action_ind]

            self.flip = flip
            self.select = cards[action_ind].lower()[0]

            # If token received, update the slots
            if flip != 'e':
                self.slots[flip] += 1
            
            # Update the card probabilities
            self.update_card_probs(trial[cards[action_ind] + "_card"], flip)

            # Check slot status and return whether the target was achieved for any goal
            status, token = self.check_slot_status(self.slots)
            trial['slot_status'] = status

            count += 1

            if status == 1:
                # If target achieved, reset the slot to 0
                self.slots[token] = 0
                reward_trials.append(count)
                goals.append(token)

            ## Choose the new goal for the next round with the latest slot configuration
            trial['slots'] = self.slots.copy()
            goal, goal_selection_probs = self.choose_goal(self.prev_goal)
            trial['goal_probs'] = goal_selection_probs
            self.prev_goal = goal

            # Report the goal probe every third round
            if (trial_num + 1) % 3 == 0:
                trial['probe_keypress'] = get_probe_from_goal(goal)

            trials_res.append(trial)

        return trials_res, goals, reward_trials

    def run_task_block(self, block_trials, seed=100):
        """
        Simulate the whole experiment with the blocks as seen by participants
        """
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


