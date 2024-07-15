import numpy as np
from card_shows import *
import pandas as pd


class GenerateExperimentSubject:
    "Generate experiment json for subjects"
    def __init__(self, experiment):
        self.experiment = experiment
        self.get_experiment_details()
        self.get_conditions()

    def get_experiment_details(self):
        details = {}
        if (self.experiment == 1) or (self.experiment == "instr_1"):
            self.n_episodes = 18
            self.n_trials = 30
            self.n_conditions = 3
            self.mode = 'probs'

        elif self.experiment == 2:
            self.n_episodes = 12
            self.n_trials = 30
            self.n_conditions = 2
            self.mode = 'probs'

        elif self.experiment == 3:
            self.n_episodes = 12
            self.n_trials = 30
            self.n_conditions = 2
            self.mode = 'rates'

        elif self.experiment == 4:
            self.n_episodes = 12
            self.n_trials = 30
            self.n_conditions = 2
            self.mode = 'probs'

        elif self.experiment == "normative":
            self.n_episodes = 18
            self.n_trials = 30
            self.n_conditions = 1
            self.mode = 'rates'

    def get_conditions(self):

        "Experiment probabilities or mean rates per condition"

        if (self.experiment == 1) or (self.experiment == "instr_1"):
            # Conditions are 80-20-20, 70-30-30, 60-40-40
            self.conditions = [
                [[0.2, 0.2, 0.8], 0],
                [[0.7, 0.3, 0.3], 1],
                [[0.4, 0.6, 0.4], 2],
                [[0.3, 0.3, 0.7], 1],
                [[0.8, 0.2, 0.2], 0],
                [[0.4, 0.6, 0.4], 2],
                [[0.3, 0.3, 0.7], 1],
                [[0.2, 0.8, 0.2], 0],
                [[0.6, 0.4, 0.4], 2],
            ]

            self.conditions = self.conditions * 2

        elif self.experiment == 2:
            # Conditions are 75-25-25, 55-45-45
            b0 = 0.75; b1 = 0.25
            g0 = 0.55; g1 = 0.45

            self.conditions = [
                [[b1, b1, b0], 0],
                [[g0, g1, g1], 1],
                [[g1, g1, g0], 1],
                [[b1, b0, b1], 0],
                [[g1, g1, g0], 1],
                [[b0, b1, b1], 0],
                [[b1, b0, b1], 0],
                [[g1, g1, g0], 1],
                [[b1, b0, b1], 0],
                [[g0, g1, g1], 1],
                [[g1, g0, g1], 1],
                [[b0, b1, b1], 0],
            ]


        elif self.experiment == 3:
        # Token targets, mean rates, condition_number
            p_t = 24
            c_t = 48
            b_t = 72

            self.conditions = [
                # # 6, 12, 12
                [[p_t, c_t, b_t], [4, 4, 6], 0, 'p'],
                [[p_t, c_t, b_t], [2, 8, 6], 0, 'c'],
                [[p_t, c_t, b_t], [2, 4, 12], 0, 'b'],
                [[p_t, c_t, b_t], [4, 4, 6], 0, 'p'],
                [[p_t, c_t, b_t], [2, 8, 6], 0, 'c'],
                [[p_t, c_t, b_t], [2, 4, 12], 0, 'b'],


                # 10, 12, 12
                [[p_t, c_t, b_t], [2.5, 4, 6], 1, 'p'],
                [[p_t, c_t, b_t], [2, 5, 6], 1, 'c'],
                [[p_t, c_t, b_t], [2, 4, 7], 1, 'b'],
                [[p_t, c_t, b_t], [2.5, 4, 6], 1, 'p'],
                [[p_t, c_t, b_t], [2, 5, 6], 1, 'c'],
                [[p_t, c_t, b_t], [2, 4, 7], 1, 'b'],
            ]


        elif self.experiment == 4:
        # Token targets, mean rates, condition_number
            p_t = 4
            c_t = 6
            b_t = 8

            self.conditions = [
                [[p_t, c_t, b_t], [0.5, 0.4, 0.5], 0],
                [[p_t, c_t, b_t], [0.25, 0.75, 0.5], 0],
                [[p_t, c_t, b_t], [0.25, 0.4, 0.9], 0],
                [[p_t, c_t, b_t], [0.5, 0.4, 0.5], 0],
                [[p_t, c_t, b_t], [0.25, 0.75, 0.5], 0],
                [[p_t, c_t, b_t], [0.25, 0.4, 0.9], 0],

                [[p_t, c_t, b_t], [0.5, 0.5, 0.65], 1],
                [[p_t, c_t, b_t], [0.33, 0.75, 0.65], 1],
                [[p_t, c_t, b_t], [0.33, 0.5, 0.9], 1],
                [[p_t, c_t, b_t], [0.5, 0.5, 0.65], 1],
                [[p_t, c_t, b_t], [0.33, 0.75, 0.65], 1],
                [[p_t, c_t, b_t], [0.33, 0.5, 0.9], 1],

            ]

        return self.conditions


    def get_episodes(self):
        episodes = []
        conditions = self.conditions.copy()
        np.random.shuffle(conditions)
        for i in range(self.n_episodes):
            condition_array = conditions[i]
            targets = condition_array[0]
            probs = condition_array[1]
            condition = condition_array[2]
            if self.experiment == 3:
                best_token = condition_array[3]
            trials = []
            card_shows = get_card_shows(self.n_trials, shows=3)
            for i in range(self.n_trials):
                trial = {}
                trial['trial'] = i
                trial['P'] = probs[0]
                trial['C'] = probs[1]
                trial['B'] = probs[2]

                cards = card_shows[i]

                if self.mode == "probs":
                    p_stoc = 1
                    execute = np.random.choice([0, 1], p=[1 - p_stoc, p_stoc])
                    trial['execute'] = execute

                    trial[cards[0] + '_card'], trial[cards[0] + '_token'] = \
                        flip_card_stochastic(probs, cards, card=cards[0], execute=execute)

                    trial[cards[1] + '_card'], trial[cards[1] + '_token'] = \
                        flip_card_stochastic(probs, cards, card=cards[1], execute=execute)

                    trial[cards[2] + '_card'], trial[cards[2] + '_token'] = \
                        flip_card_stochastic(probs, cards, card=cards[2], execute=execute)

                else:
                    trial[cards[0] + '_token'] = int(np.random.poisson(lam=probs[0]))
                    trial[cards[1] + '_token'] = int(np.random.poisson(lam=probs[1]))
                    trial[cards[2] + '_token'] = int(np.random.poisson(lam=probs[2]))

                np.random.shuffle(cards)

                trial['37'] = cards[0]
                trial['39'] = cards[1]
                trial['38'] = cards[2]

                trial[cards[0]] = '37'
                trial[cards[1]] = '39'
                trial[cards[2]] = '38'

                trial['P_T'] = targets[0]
                trial['C_T'] = targets[1]
                trial['B_T'] = targets[2]
                trial['condition'] = condition

                if self.experiment == 3:
                    trial['best_token'] = best_token

                trials.append(trial)
            episodes.append(trials)

        return episodes

    def generate_experiment_json(self, file_name='json/trials_block.json'):
        episodes = self.get_episodes()
        df = pd.DataFrame()
        count = 0
        for trials in episodes:
            for trial_num in range(len(trials)):
                trial_dict = trials[trial_num]
                trial_dict['block'] = count
                df = pd.concat([df, pd.DataFrame(trial_dict, index=[trial_num])])
            count += 1

        df.groupby(['block']).apply(lambda x: x[df.columns[0:]].to_dict('records')).reset_index().rename(
            columns={0: 'trials'}).to_json(file_name, orient='records', indent=4)

        return df


if __name__ == "__main__":
    experiment = GenerateExperimentSubject(4)
    experiment.generate_experiment_json(file_name='json/trials_practice_exp_4.json')










