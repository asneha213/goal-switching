import numpy as np


CARDS_TWO = [
    ['P1', 'C1'],
    ['P2', 'C1'],
    ['P1', 'C2'],
    ['P2', 'C2'],
    ['P1', 'B1'],
    ['P2', 'B1'],
    ['P1', 'B2'],
    ['P2', 'B2'],
    ['C1', 'B1'],
    ['C2', 'B1'],
    ['C1', 'B2'],
    ['C2', 'B2'],
    ['P1', 'P2'],
    ['C1', 'C2'],
    ['B1', 'B2']
]

CARDS_THREE = [
    ['P1', 'C1', 'B1'],
    ['P2', 'C1', 'B1'],
    ['P1', 'C2', 'B1'],
    ['P2', 'C2', 'B1'],
    ['P1', 'C1', 'B2'],
    ['P2', 'C1', 'B2'],
    ['P1', 'C2', 'B2'],
    ['P2', 'C2', 'B2'],
]


def get_card_shows(num_trials=36, shows=2):
    cards_task = []
    if shows == 2:
        blocks = int(num_trials / 3)
    else:
        blocks = int(num_trials / 8) + 1
    for i in range(blocks):
        if shows == 3:
            cards = CARDS_THREE
        else:
            cards = []
            cardtypes = [['P', 'C'], ['C', 'B'], ['B', 'P']]
            for cardtype in cardtypes:
                cards = cards + [[cardtype[0] + np.random.choice(['1', '2']), \
                                 cardtype[1] + np.random.choice(['1', '2'])]]
        np.random.shuffle(cards)
        cards_task += cards
    return cards_task


def flip_card_stochastic(probs, cards, card, execute):
    card_id = ['p', 'c', 'b'].index(card.lower()[0])
    cards_lower = [i.lower()[0] for i in cards]
    if execute:
        flip = np.random.choice([0, 1], p=[1 - probs[card_id], probs[card_id]])
        card_flip = card
    else:
        card_ids = list(set([0, 1, 2]) - set([card_id]))
        card_id_n = np.random.choice(card_ids, p=[0.5, 0.5])
        card_flip_id = cards_lower.index(['p', 'c', 'b'][card_id_n])

        flip = np.random.choice([0, 1], p=[1 - probs[card_id_n], probs[card_id_n]])
        card_flip = cards[card_flip_id]
    if flip:
        token = card_flip.lower()[0]
    else:
        token = 'e'

    return card_flip, token


