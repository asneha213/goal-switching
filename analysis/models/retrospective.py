from .prospective import Prospective


class Retrospective(Prospective):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.beta_g = params[1]
        self.beta_0g = params[2]
        self.beta_a = params[3]
        self.beta_0a = params[4]

        self.rho = 0
        self.prev_goal = ''
        self.exp = 1

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
            self.rho  += self.alpha * (1 - self.rho)
        else:
            self.M[card_type] +=  self.alpha * (0 - self.M[card_type])
            self.rho  += self.alpha * (0 - self.rho)

    def calculate_goal_value_counting(self, token, count, k):
        # if count == self.targets[token]:
        #     return self.reward

        q = (count / self.targets[token]) * self.reward

        return q

    def calculate_qvals_goals(self):
        qvals = {}

        # calculate goal values
        qvals['p'] = self.calculate_goal_value_counting('p', count=self.slots['p'], k=0)
        qvals['c'] = self.calculate_goal_value_counting('c', count=self.slots['c'], k=0)
        qvals['b'] = self.calculate_goal_value_counting('b', count=self.slots['b'], k=0)
        return qvals
