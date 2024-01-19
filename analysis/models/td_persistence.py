from .model import Model


class TDPersistence(Model):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
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

    def update_card_probs(self, card, flip):
        super().update_card_probs(card, flip)
        card_type = card[0].lower()

        if flip != 'e':
            self.M[card_type] +=  self.alpha * (1 - self.M[card_type])
        else:
            self.M[card_type] +=  self.alpha * (0 - self.M[card_type])

    def calculate_qvals_goals(self):
        return self.M


