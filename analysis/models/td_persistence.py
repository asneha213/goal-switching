from .prospective import Prospective


class TDPersistence(Prospective):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.beta_g = params[1]
        self.beta_0g = params[2]
        self.beta_a = params[3]

        self.beta_0a = params[2]

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

    def calculate_qvals_goals(self):
        return self.M
