from .model_latent import Model

def Momentum(base_class_name):
    if base_class_name == "model":
        base_class = Model

    class MomentumClass(base_class):
        def __init__(self, params):
            super().__init__(params)
            params = self.params
            self.reset_card_probs()

        def reset_card_probs(self):
            super().reset_card_probs()
            self.w = {'p': 0.5, 'c': 0.5, 'b': 0.5}
            self.b = {'p': 0.1, 'c': 0.1, 'b': 0.1}

        def update_card_probs(self, card, flip):
            self.alpha = self.params['alpha']
            self.gamma = self.params['gamma']
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

    return MomentumClass
