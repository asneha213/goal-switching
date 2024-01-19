from .model import Model


class Prospective(Model):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.alpha_c = params[1]
        self.beta_0 = params[2]
        self.beta_g = params[3]
        self.beta_a = params[4]
        self.gamma = params[5]

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

        return qvals





