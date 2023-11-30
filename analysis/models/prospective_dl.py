from .prospective import Prospective


class ProspectiveDL(Prospective):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.alpha_e = params[1]
        self.gamma = params[2]
        self.beta_g = params[3]
        self.beta_0g = params[4]
        self.beta_a = params[5]

        self.beta_0a = params[3]

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
            self.M[card_type] += self.alpha * (1 - self.M[card_type])
            self.rho += self.alpha * (1 - self.rho)
        else:
            self.M[card_type] += self.alpha_e * (0 - self.M[card_type])
            self.rho += self.alpha_e * (0 - self.rho)

    def calculate_goal_value_recursion(self, token, p, count, k):
        if count == self.targets[token]:
            return self.reward
        if k > 10:
            return 0
        q = (p * self.gamma * self.calculate_goal_value_recursion(token, p, count + 1, k + 1)) \
            / (1 - (1 - p) * self.gamma)
        return q

    def calculate_qvals_goals(self):
        qvals = {}

        probs = [self.M['p'], self.M['c'], self.M['b']]

        # calculate goal values
        qvals['p'] = self.calculate_goal_value_recursion('p', probs[0], count=self.slots['p'], k=0)
        qvals['c'] = self.calculate_goal_value_recursion('c', probs[1], count=self.slots['c'], k=0)
        qvals['b'] = self.calculate_goal_value_recursion('b', probs[2], count=self.slots['b'], k=0)
        return qvals
