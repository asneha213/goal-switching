from .model_latent import Model


def TDPersistence(base_class_name):
    if base_class_name == "model":
        base_class = Model

    class TDPersistenceClass(base_class):

        def __init__(self, params):
            super().__init__(params)
            self.params = params
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

            self.alpha = self.params['alpha']

            card_type = card[0].lower()

            if flip != 'e':
                self.M[card_type] +=  self.alpha * (1 - self.M[card_type])
            else:
                self.M[card_type] +=  self.alpha * (0 - self.M[card_type])

        def calculate_qvals_goals(self):
            return self.M

    return TDPersistenceClass 


