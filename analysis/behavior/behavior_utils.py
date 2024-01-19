from .datautils import *
import numpy as np


GOAL_PROBE = {
    37: 'p',
    38: 'c',
    39: 'b',
    40: 'u'
}


def get_trials_block(subject_data, block_num):
    blocks = subject_data['GoalSwitching']
    block = blocks[str(block_num)]
    trials = []

    for i in range(30):
        trial = block['trial_' + str(i)]
        trials.append(trial)

    return trials


def get_bonuses():
    bonuses = []
    for subject_id in range(70):
        subject_data = get_subject_data_from_id(subject_id)
        bonus = subject_data['bonus']
        bonuses.append(bonus)
    return bonuses


def get_goal_from_probe(probe_key):
    if probe_key == 37:
        return 'p'
    elif probe_key == 38:
        return 'c'
    elif probe_key == 39:
        return 'b'
    else:
        return 'u'


def get_probe_from_goal(goal):
    if goal == 'p':
        return 37
    elif goal == 'c':
        return 38
    elif goal == 'b':
        return 39
    else:
        return 40


def get_readable_behavior(subject_id):
    subject_data = get_subject_data_from_id(subject_id)
    block_data = {}
    for block_num in range(18):
        trials = get_trials_block(subject_data, block_num)
        data = []
        for trial in trials:
            cardselect = trial['cardselect'].lower()[0]
            token = trial['current_token']
            ts = trial['slots_pre']
            slots = [ts['p'], ts['c'], ts['b']]
            condition = trial['condition']
            max_token = ['p', 'c', 'b'][np.argmax([trial['P'], trial['C'], trial['B']])]
            if 'probe_keypress' in trial:
                probe = get_goal_from_probe(trial['probe_keypress'])
            else:
                probe = -1
            data.append([slots, cardselect, token, condition, max_token, probe])
        block_data[block_num] = data
    return block_data, subject_data


def get_subject_data_for_viz(experiment, subject_id, model_res=False, block_range=np.arange(12)):
    if model_res:
        subject_data = model_res
    else:
        subject_data = get_subject_data_from_id(experiment, subject_id)

    blocks = subject_data['GoalSwitching']

    actions = []
    probes = []
    conditions = []
    max_prob_tokens = []


    for block_num in block_range:

        for trial_num in range(30):
            trial = blocks[str(block_num)]['trial_' + str(trial_num)]
            condition = trial['condition']
            probs = [trial['P'], trial['C'], trial['B']]
            max_prob_token = ['p', 'c', 'b'][np.argmax(probs)]
            cardselect = trial['cardselect'][0].lower()
            token = trial['current_token']
            actions.append(cardselect + '_' + token)
            conditions.append(condition)
            max_prob_tokens.append(max_prob_token)
            if 'probe_keypress' in trial:
                probe = get_goal_from_probe(trial['probe_keypress'])
                probes.append(probe)
            else:
                probes.append('na')

    data_viz = {
        'actions': actions,
        'probes': probes,
        'conditions': conditions,
        'max_prob_tokens': max_prob_tokens,
    }

    return data_viz


def last_trial_slots(block):
    last_trial = block['trial_59']
    if 'slots' in last_trial:
        slots = last_trial['slots']
    else:
        slots = last_trial['slots_pre']
    token_nums = [slots['p'], slots['c'], slots['b']]
    max_num = np.max(token_nums)
    token_percent = max_num / 6
    return token_percent


def get_goal_switches(goals):
    prev_goal = ''
    switches = 0
    for i in range(len(goals)):
        if i > 0:
            if goals[i] != prev_goal:
                switches += 1
        prev_goal = goals[i]
    return switches


def get_trial_conditions_order(exp):
    if exp == 1:
        conditions = [
            [0.25, 0.25, 0.85],
            [0.35, 0.35, 0.75],
            [0.45, 0.45, 0.65],
            [0.55, 0.55, 0.55]
        ]
        order = [2, 6, 9, 5, 11, 12, 14, 3, 13, 15, 1, 4, 8, 10, 0, 7]

    elif exp == 2:
        conditions = [
            [0.25, 0.25, 0.85],
            [0.35, 0.35, 0.75],
            [0.45, 0.45, 0.65],
        ]
        order = [2, 5, 12, 6, 13, 3, 9, 11, 14, 1, 4, 8, 10, 0, 7]

    return conditions, order


def sort_array(to_sort, order):
    to_sort = np.array(to_sort)
    order = np.array(order)
    inds = order.argsort()
    sorted = to_sort[inds]
    return sorted


def collate_across_conditions(measure, exp):

    def get_mean_condns(res):
        if (exp == 1):
            return (res[0:4] + res[4:8] + res[8:12] + res[12:16]) / 4
        elif (exp == 2) or (exp == 3):
            return (res[0:3] + res[3:6] + res[6:9] + \
                    res[9:12] + res[12:15]) / 5

    conditions, order = get_trial_conditions_order(exp)
    measure_sort = sort_array(measure, order)
    measure_cond = get_mean_condns(measure_sort)
    return measure_cond


def get_moving_average(arr, window_size=3):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3

    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages


def get_goals_from_choices(choices):

    goals = [ (choices[i + 1] + choices[i + 2] ) / 2 for i in range(len(choices)) if (i % 3 == 2) and (i < len(choices) - 2)]
    return goals


if __name__ == "__main__":
    subject_id = 0
    data, sub_data = get_readable_behavior(subject_id)
