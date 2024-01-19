from .prospective import Prospective


class ProspectiveMomentum(Prospective):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.alpha_c = params[1]
        self.beta_0 = params[2]
        self.beta_g = params[3]
        self.beta_a = params[4]
        self.gamma = params[5]
        self.alpha_e = params[6]
        self.alpha_m = params[7]

        self.prev_goal = ''

        self.reset_card_probs()

        self.mm = {
            'p': 0,
            'c': 0,
            'b': 0
        }

    def check_slot_status(self, slots):
        if slots['p'] == self.targets['p']:
            self.mm['p']  *= 0
            return 1, 'p'
        elif slots['c'] == self.targets['c']:
            self.mm['c'] *= 0
            return 1, 'c'
        elif slots['b'] == self.targets['b']:
            self.mm['b'] *= 0
            return 1, 'b'

        return 0, -1

    def update_card_probs(self, card, flip):
        super().update_card_probs(card, flip)
        card_type = card[0].lower()

        if flip != 'e':
            rpe = 1 - self.M[card_type]
            self.M[card_type] += self.alpha * (self.mm[card_type] + 1 - self.M[card_type])
        else:
            rpe = 0 - self.M[card_type]
            self.M[card_type] += self.alpha_e * (self.mm[card_type] + 0 - self.M[card_type])

        self.mm[card_type] += self.alpha_m * (rpe - self.mm[card_type])

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
