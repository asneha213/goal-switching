from .card_shows import  *
from .random_walks import *


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


def get_trial_dict_card_draws_trial(cards, probs, trial_num, p_stoc):
    trial = {}
    trial['trial'] = trial_num
    trial['P'] = probs[0]
    trial['C'] = probs[1]
    trial['B'] = probs[2]

    execute = np.random.choice([0, 1], p=[1 - p_stoc, p_stoc])
    trial['execute'] = execute

    trial[cards[0] + '_card'], trial[cards[0] + '_token'] = \
        flip_card_stochastic(probs, cards, card=cards[0], execute=execute)

    trial[cards[1] + '_card'], trial[cards[1] + '_token'] = \
        flip_card_stochastic(probs, cards, card=cards[1], execute=execute)

    trial[cards[2] + '_card'], trial[cards[2] + '_token'] = \
        flip_card_stochastic(probs, cards, card=cards[2], execute=execute)

    np.random.shuffle(cards)

    trial['37'] = cards[0]
    trial['39'] = cards[1]
    trial['38'] = cards[2]

    trial[cards[0]] = '37'
    trial[cards[1]] = '39'
    trial[cards[2]] = '38'

    return trial


def generate_experiment_episode_trials(probs, slots, condition, p_stoc, num_trials):
    trials = []
    cards = get_card_shows(num_trials, shows=3)

    for i in range(num_trials):
        trial = get_trial_dict_card_draws_trial(cards[i], probs, trial_num=i, p_stoc=p_stoc)

        trial['P_T'] = slots[0]
        trial['C_T'] = slots[1]
        trial['B_T'] = slots[2]
        trial['condition'] = condition
        trials.append(trial)

    return trials


def generate_normative_episode_trials(num_episodes, num_trials, walk, seed):

    np.random.seed(seed)

    probs1, probs2, probs3 = get_random_walks_token_probs(num_trials=num_trials*num_episodes, walk=walk)

    episodes = []

    for episode_num in range(num_episodes):
        trials = []
        cards = get_card_shows(num_trials, shows=3)
        for trial_num in range(num_trials):

            prob_num = episode_num * num_trials + trial_num
            probs = [probs1[prob_num], probs2[prob_num], probs3[prob_num]]
            trial = get_trial_dict_card_draws_trial(cards[trial_num], probs, trial_num, p_stoc=1.0)
            trial['P_T'] = 7
            trial['C_T'] = 7
            trial['B_T'] = 7
            trial['condition'] = 0
            trials.append(trial)

        episodes.append(trials)

    return episodes, [probs1, probs2, probs3]


if __name__ == "__main__":
    trials = generate_experiment_episode_trials([0.2, 0.5, 0.8], slots=[4,7,9])
