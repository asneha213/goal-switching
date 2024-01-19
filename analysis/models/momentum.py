from .model import Model


class Momentum(Model):
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
        self.C = {
            'p': 0.33,
            'c': 0.33,
            'b': 0.33
        }
        self.w = {'p': 0.5, 'c': 0.5, 'b': 0.5}
        self.b = {'p': 0.1, 'c': 0.1, 'b': 0.1}

    def update_card_probs(self, card, flip):
        super().update_card_probs(card, flip)
        token = card[0].lower()
        if flip == 'e':
            progress_new = self.slots[token] / self.targets[token]
            progress = self.slots[token] / self.targets[token]
        else:
            progress_new = (self.slots[token] + 1) / self.targets[token]
            progress = self.slots[token] / self.targets[token]

        delta = self.gamma * (self.w[token] * progress_new + self.b[token]) - self.w[token] * (
                        progress) - self.b[token]

        self.w[token] += self.alpha * delta * progress_new
        self.b[token] += self.alpha * delta

    def calculate_goal_value(self, token, count):

        progress = count / self.targets[token]
        val = self.w[token] * progress + self.b[token]

        return val

    def calculate_qvals_goals(self):
        q_vals = {'p': 0, 'c': 0, 'b': 0}

        for token in ['p', 'c', 'b']:
            q_vals[token] = self.calculate_goal_value(token, self.slots[token])

        return q_vals


