from .model_latent import Model
import numpy as np


def ProspectiveHyperbolic(base_class_name):
    if base_class_name == "model":
        base_class = Model

    class ProspectiveHyperbolicClass(base_class):
        def __init__(self, params):
            super().__init__(params)

            self.params = params
            self.reset_card_probs()

        def reset_card_probs(self):
            super().reset_card_probs()

            self.M = {
                'p': 0.33,
                'c': 0.33,
                'b': 0.33
            }

        def update_card_probs(self, card, flip):
            self.alpha = self.params['alpha']
            super().update_card_probs(card, flip)

            card_type = card[0].lower()

            if flip != 'e':
                self.M[card_type] +=  self.alpha * (1 - self.M[card_type])
            else:
                self.M[card_type] +=  self.alpha * (0 - self.M[card_type])


        def calculate_goal_value_recursion(self, token, p, count, k):
            self.gamma = self.params['gamma']
            if count == self.targets[token]:
                return self.reward
            if k > 20:
                return 0

            q = ( p * self.gamma * self.calculate_goal_value_recursion(token, p, count + 1, k+1) ) \
                / (1 - (1 - p) * self.gamma)

            return q

        def calculate_qvals_goals(self):
            qvals = {}

            probs = [self.M['p'], self.M['c'], self.M['b']]

            #calculate goal values
            qvals['p'] = self.calculate_goal_value_recursion('p', probs[0], count=self.slots['p'], k=0)
            qvals['c'] = self.calculate_goal_value_recursion('c', probs[1], count=self.slots['c'], k=0)
            qvals['b'] = self.calculate_goal_value_recursion('b', probs[2], count=self.slots['b'], k=0)

            #print(self.slots, qvals)

            return qvals

        def finish_goal(self, goal, p, count, k):
            if k > 10:
                return 15
            if count == self.targets[goal]:
                return k
            else:
                if np.random.random() < p:
                    count += 1
                return self.finish_goal(goal, p, count, k + 1)

        def calculate_goal_value_recursion(self, token, p, count, k=0):
            gamma = self.params['gamma']
            delay = self.finish_goal(token, p, count, k)
            discount = 1 / (1 + gamma * delay)
            return discount * self.reward

    return ProspectiveHyperbolicClass




