from .model import Model


class Retrospective(Model):
    def __init__(self, params):
        super().__init__(params)

        self.gamma = params[0]
        self.alpha_c = params[1]
        self.beta_0 = params[2]
        self.beta_g = params[3]
        self.beta_a = params[4]

        self.prev_goal = ''

        self.reset_card_probs()

    def reset_card_probs(self):
        super().reset_card_probs()
        self.M = {
            'p': 0.33,
            'c': 0.33,
            'b': 0.33
        }

    def calculate_goal_value_count(self, token, count, k):
        if count == self.targets[token]:
            return self.reward
        if k > 10:
            return 0
        q = self.gamma * self.calculate_goal_value_count(token, count + 1, k + 1)
        return q

    def calculate_qvals_goals(self):
        qvals = {}

        # calculate goal values
        qvals['p'] = self.calculate_goal_value_count('p', count=self.slots['p'], k=0)
        qvals['c'] = self.calculate_goal_value_count('c', count=self.slots['c'], k=0)
        qvals['b'] = self.calculate_goal_value_count('b', count=self.slots['b'], k=0)
        return qvals
