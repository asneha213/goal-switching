from .generate_trials_episode import *

import pandas as pd


def get_experiment_details(experiment):

    details = {}

    if experiment == 1:
        details["num_episodes"] = 18
        details["num_trials"] = 30
        details["num_conditions"] = 3

    elif experiment == 2:
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2

    elif experiment == 3:
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2

    elif experiment == "normative":
        details["num_episodes"] = 18
        details["num_trials"] = 30
        details["num_conditions"] = 1

    return details


def get_conditions(experiment):

    if experiment == 1 :
        conditions = [
            [[0.2, 0.2, 0.8], 0],
            [[0.7, 0.3, 0.3], 1],
            [[0.4, 0.6, 0.4], 2],
            [[0.3, 0.3, 0.7], 1],
            [[0.8, 0.2, 0.2], 0],
            [[0.4, 0.6, 0.4], 2],
            [[0.3, 0.3, 0.7], 1],
            [[0.2, 0.8, 0.2], 0],
            [[0.6, 0.4, 0.4], 2],
        ]

        conditions = conditions * 2

    elif experiment == 2:
        b0 = 0.75
        b1 = 0.25

        g0 = 0.55
        g1 = 0.45

        conditions = [
            [[b1, b1, b0], 0],
            [[g0, g1, g1], 1],
            [[g1, g1, g0], 1],
            [[b1, b0, b1], 0],
            [[g1, g1, g0], 1],
            [[b0, b1, b1], 0],
            [[b1, b0, b1], 0],
            [[g1, g1, g0], 1],
            [[b1, b0, b1], 0],
            [[g0, g1, g1], 1],
            [[g1, g0, g1], 1],
            [[b0, b1, b1], 0],
        ]

    return conditions


def generate_experiment_trials(experiment):

    details = get_experiment_details(experiment)

    num_episodes = details["num_episodes"]
    num_trials = details["num_trials"]

    conditions = get_conditions(experiment, num_episodes)

    episodes = []

    if experiment == 1:
        slots = [7, 7, 7]
    elif experiment == 2:
        slots = [7, 7, 7]

    for i in range(num_episodes):

        probs = conditions[i][0]
        condition = conditions[i][1]
        trials = generate_experiment_episode_trials(probs, slots, condition, \
                                                     p_stoc=1.0, num_trials=num_trials, seed=i*20)
        episodes.append(trials)

    return episodes


def generate_experiment_json(block_trials, file_name='json/trials_block.json'):
    df = pd.DataFrame()
    count = 0
    for trials in block_trials:
        for trial_num in range(len(trials)):
            trial_dict = trials[trial_num]
            trial_dict['block'] = count
            df = pd.concat([df, pd.DataFrame(trial_dict, index=[trial_num])])
        count += 1

    df.groupby(['block']).apply(lambda x: x[df.columns[0:]].to_dict('records')).reset_index().rename(
        columns={0: 'trials'}).to_json(file_name, orient='records', indent=4)

    return df


def generate_normative_json(file_name='json/trials_normative.json', walk=True, seed=100):
    episodes = generate_normative_episode_trials(num_episodes=18, num_trials=30, walk=walk, seed=seed)
    df = generate_experiment_json(episodes, file_name=file_name)



def generate_practice_json():
    probs = [0.6, 0.4, 0.4]
    slots = [7, 7, 7]
    trials = generate_experiment_episode_trials(probs, slots, 'u', p_stoc=1.0, num_trials=20, seed=130)
    df = generate_experiment_json([trials], file_name='json/trials_practice_exp_1.json')


def generate_main_experiment_json(experiment):
    episodes = generate_experiment_trials(experiment)
    df = generate_experiment_json(episodes, file_name='json/trials_exp_' + str(experiment) + '.json')


if __name__ == "__main__":

    generate_normative_json()

