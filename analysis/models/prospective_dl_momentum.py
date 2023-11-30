from .prospective_momentum import ProspectiveMomentum


class ProspectiveDLMomentum(ProspectiveMomentum):
    def __init__(self, params):
        super().__init__(params)

        self.alpha = params[0]
        self.alpha_e = params[1]
        self.alpha_m = params[2]
        self.gamma = params[3]
        self.beta_g = params[4]
        self.beta_0g = params[5]
        self.beta_a = params[6]

        self.beta_0a = params[5]

        self.rho = 0
        self.prev_goal = ''
        self.exp = 1

        self.reset_card_probs()

