from .model_just_goal import *
from scipy.special import softmax
import numpy as np



class Rescorla(ModelJustGoal):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params['alpha']
        self.alpha_c = params['alpha_c']
        self.beta_g = params['beta_g']
        self.beta_c = params['beta_c']

        self.prev_goal = ''

        self.reset_card_probs()

    def reset_card_probs(self):
        super().reset_card_probs()
        self.M = {
            'p': 0.33,
            'c': 0.33,
            'b': 0.33
        }

    def update_card_probs(self, card, flip):
        super().update_card_probs(card, flip)
        card_type = card[0].lower()

        if flip != 'e':
            self.M[card_type] +=  self.alpha * (1 - self.M[card_type])
        else:
            self.M[card_type] +=  self.alpha * (0 - self.M[card_type])


    def choose_card_action(self, cards):
        action_vals = [self.beta_g * self.M[card[0].lower()] + self.beta_c * self.C[card[0].lower()] for card in cards]
        action_probs = softmax(action_vals)
        action = np.random.choice(range(3), p=action_probs)
        action_token = cards[action][0].lower()
        action_probs = {cards[i][0].lower(): action_probs[i] for i in range(3)}
        return action, action_probs



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
            action, action_probs = self.choose_card_action(cards)

            flip = trial[cards[action] + "_token"]
            trial['current_token'] = flip
            trial["cardselect"] = cards[action]
            trial["cardflip"] = cards[action]

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
            goal = cards[action][0].lower()
            self.prev_goal = goal

            if status == 1:
                self.slots[token] = 0
                reward_trials.append(count)
                goals.append(token)


            if (trial_num + 1) % 3 == 0:
                goal = cards[action][0].lower()
                trial['probe_keypress'] = get_probe_from_goal(self.prev_goal)

            trials_res.append(trial)

        return trials_res, goals, reward_trials


