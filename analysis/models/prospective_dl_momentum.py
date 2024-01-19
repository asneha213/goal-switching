from .prospective_momentum import ProspectiveMomentum


class ProspectiveDLMomentum(ProspectiveMomentum):
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

