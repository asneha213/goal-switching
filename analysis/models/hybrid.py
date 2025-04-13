from .model_latent import Model
import sys
sys.path.append("../src/generate/")



def Hybrid(base_class_name):
    if base_class_name == "model":
        base_class = Model

    class HybridClass(base_class):

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
            super().update_card_probs(card, flip)
            self.alpha = self.params['alpha']

            card_type = card[0].lower()

            if flip != 'e':
                self.M[card_type] += self.alpha * (1 - self.M[card_type])
            else:
                self.M[card_type] += self.alpha * (0 - self.M[card_type])

        def calculate_goal_value_recursion(self, token, p, count, k):
            self.gamma = self.params['gamma']
            if count == self.targets[token]:
                return self.reward
            if k > 10:
                return 0
            q = ( p * self.gamma * self.calculate_goal_value_recursion(token, p, count + 1, k+1) )\
                / (1 - (1 - p) * self.gamma)
            return q

        def calculate_goal_value_count(self, token, count, k):
            if count == self.targets[token]:
                return self.reward
            if k > 10:
                return 0
            q = self.gamma * self.calculate_goal_value_count(token, count + 1, k+1)
            return q

        def calculate_goal_value_hybrid(self, token, p):
            self.wa = self.params['wa']
            q_pros = self.calculate_goal_value_recursion(token, p, self.slots[token], k=0)
            q_retro = self.calculate_goal_value_count(token, self.slots[token], k=0)
            q = self.wa * q_pros + (1 - self.wa) * q_retro
            return q

        def calculate_qvals_goals(self, update=True):
            qvals = {}

            # calculate goal values
            qvals['p'] = self.calculate_goal_value_hybrid('p', self.M['p'])
            qvals['c'] = self.calculate_goal_value_hybrid('c', self.M['c'])
            qvals['b'] = self.calculate_goal_value_hybrid('b', self.M['b'])
            return qvals

    return HybridClass
