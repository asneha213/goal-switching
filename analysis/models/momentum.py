from .prospective import Prospective


class Momentum(Prospective):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.gamma = params[1]
        self.beta_g = params[2]
        self.beta_0g = params[3]
        self.beta_a = params[4]

        self.beta_0a = params[3]

        self.rho = 0
        self.prev_goal = ''
        self.exp = 1

        self.reset_card_probs()

    def reset_card_probs(self):

        self.w = {'p': 0.5, 'c': 0.5, 'b': 0.5}
        #self.w = {'p': 5, 'c': 5, 'b': 5}
        self.b = {'p': 0.1, 'c': 0.1, 'b': 0.1}

    def update_card_probs(self, card, flip):
        token = card[0].lower()
        if flip == 'e':
            progress_new = self.slots[token] / self.targets[token]
            progress = self.slots[token] / self.targets[token]
        else:
            progress_new = (self.slots[token] + 1) / self.targets[token]
            progress = self.slots[token] / self.targets[token]

        delta = self.gamma * (self.w[token] * (progress_new ** self.exp) + self.b[token]) - self.w[token] * (
                        progress ** self.exp) - self.b[token]

        self.w[token] += self.alpha * delta * (progress_new ** self.exp)
        self.b[token] += self.alpha * delta


    def calculate_goal_value(self, token, count, k, update=True):

        progress = count / self.targets[token]

        val = self.w[token]*(progress**self.exp) + self.b[token]

        return val


    def calculate_qvals_goals(self):
        q_vals = {'p': 0, 'c': 0, 'b': 0}

        for token in ['p', 'c', 'b']:
            q_vals[token] = self.calculate_goal_value(token, self.slots[token], self.targets[token])

        # print(self.slots)
        #print(q_vals)
        return q_vals


