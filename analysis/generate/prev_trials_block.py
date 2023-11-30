def generate_experiment_2_trials(num_trials=25, seed=120):

    prob_conditions = [
        [0.25, 0.5, 0.75],
        [0.15, 0.5, 0.85],
        [0.1, 0.5, 0.9],
        [0.35, 0.5, 0.65],
    ]
    slot_conditions = [
        [7, 1, 1],
        [6, 2, 1],
        [5, 3, 1],
        [5, 2, 2],
        [4, 3, 2],
        [4, 4, 1]
    ]

    experiment_trials = []
    for i in range(len(prob_conditions)):
        for j in range(len(slot_conditions)):
            probs = prob_conditions[i]
            slots = slot_conditions[j]
            trials = generate_trial_dict_episode_exp_2(probs, slots, i, j, num_trials=num_trials, seed=seed)
            experiment_trials.append(trials)

    np.random.shuffle(experiment_trials)
    return experiment_trials


def generate_experiment_trials_block_v1(num_blocks=15, num_trials=72, seed=120):

    conditions = [
        [0.25, 0.25, 0.85],
        [0.35, 0.35, 0.75],
        [0.45, 0.45, 0.65],
    ]

    conditions = conditions * int(num_blocks / 3)

    order = np.arange(num_blocks)
    np.random.seed(seed)
    np.random.shuffle(order)
    conditions_sh = [conditions[i] for i in order]

    rand_ints = np.random.randint(0, 3, size=num_blocks)

    experiment_trials = []
    count = 0
    for probs in conditions_sh:
        rand_int = rand_ints[count]
        if rand_int == 1:
            probs1 = np.roll(probs, 1)
        elif rand_int == 2:
            probs1 = np.roll(probs, 2)
        else:
            probs1 = probs
        trials = generate_trial_dict_episode_v1(probs1, num_trials=num_trials, seed=seed)
        experiment_trials.append(trials)
        count += 1

    return experiment_trials, order


def generate_experiment_1_trials(num_blocks=16, num_trials=60, seed=120):
    # optimal goal selection/ persistence

    conditions = [
        [0.25, 0.25, 0.85],
        [0.35, 0.35, 0.75],
        [0.45, 0.45, 0.65],
        [0.55, 0.55, 0.55]
    ]

    conditions = conditions * int(num_blocks / 4)

    order = np.arange(num_blocks)
    np.random.seed(seed)
    np.random.shuffle(order)
    conditions_sh = [conditions[i] for i in order]

    rand_ints = np.random.randint(0, 3, size=num_blocks)

    experiment_trials = []
    count = 0
    for probs in conditions_sh:
        rand_int = rand_ints[count]
        if rand_int == 1:
            probs1 = np.roll(probs, 1)
        elif rand_int == 2:
            probs1 = np.roll(probs, 2)
        else:
            probs1 = probs
        trials = generate_trial_dict_episode(probs1, num_trials=num_trials, pr_t=0.1, seed=seed)
        experiment_trials.append(trials)
        count += 1

    return experiment_trials, order


def generate_main_experiment_json(version="v0"):
    if version == "v0":
        block_trials, order = generate_experiment_trials_block(num_trials=60, seed=120)
    else:
        block_trials, order = generate_experiment_trials_block_v1(num_trials=72, seed=120)
    print(order)
    df = generate_experiment_json(block_trials, file_name='json/trials_block' + version + '.json')



def flip_card(p, pr_t=0.0, card='p'):
    pr = min(1 - p, pr_t)

    if card == 'p':
        flip = np.random.choice(['e', 'p', 'c', 'b'], p=[1 - p - pr, p, pr/ 2, pr/ 2])

    elif card == 'c':
        flip = np.random.choice(['e', 'p', 'c', 'b'], p=[1 - p - pr, pr/2 , p, pr/2])

    elif card == 'b':
        flip = np.random.choice(['e', 'p', 'c', 'b'], p=[1 - p - pr, pr/2, pr/2, p])

    return flip




def generate_trial_dict_episode(probs, num_trials=36, start= 0, shows=2, pr_t=0.1, seed=100):
    np.random.seed(seed)
    trials = []
    cards = get_card_shows(num_trials, shows)
    for i in range(num_trials):
        trial = {}
        trial['trial'] = start + i

        trial['P'] = probs[0]
        trial['C'] = probs[1]
        trial['B'] = probs[2]

        trial['P1'] = flip_card(probs[0], pr_t, card='p')
        trial['P2'] = flip_card(probs[0], pr_t, card='p')
        trial['C1'] = flip_card(probs[1], pr_t, card='c')
        trial['C2'] = flip_card(probs[1], pr_t, card='c')
        trial['B1'] = flip_card(probs[2], pr_t, card='b')
        trial['B2'] = flip_card(probs[2], pr_t, card='b')

        if shows == 2:
            cards_trial = [cards[i][0], cards[i][1] ]
        else:
            cards_trial = [cards[i][0], cards[i][1], cards[i][2]]

        np.random.shuffle(cards_trial)

        trial['37'] = cards_trial[0]
        trial['39'] = cards_trial[1]

        if shows == 2:
            trial['38'] = 'E1'
        else:
            trial['38'] = cards_trial[2]
        trials.append(trial)

    return trials




