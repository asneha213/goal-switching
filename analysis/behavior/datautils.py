import os
import json


def get_subject_data_from_id(experiment, sub_path):
    data_dir = "../data/experiment_" + str(experiment) + "/"
    sub_path = data_dir + sub_path + ".json"
    fp = open(sub_path)
    return json.load(fp)


def get_experiment_trial_details(experiment):
    details = {}

    if experiment == 1:
        details["num_episodes"] = 18
        details["num_trials"] = 30
        details["num_conditions"] = 3
        details["num_samples"] = 540

    if experiment == "instr_1":
        details["num_episodes"] = 18
        details["num_trials"] = 30
        details["num_conditions"] = 3
        details["num_samples"] = 540

    elif experiment == 2:
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2
        details["num_samples"] = 360

    elif experiment == 3:
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2

    elif experiment == "normative":
        details["num_episodes"] = 18
        details["num_trials"] = 30
        details["num_conditions"] = 1

    return details


def check_subject_validity(experiment, subject_file):
    if subject_file.endswith('.json') == False:
        return None
    fp = open(subject_file)
    data = json.load(fp)
    if "GoalSwitching" not in data:
        return None
    details = get_experiment_trial_details(experiment)
    for i in range(details["num_episodes"]):
        if str(i) not in data['GoalSwitching']:
            return None
        for j in range(details["num_trials"]):
            if 'trial_'+ str(j) not in data['GoalSwitching'][str(i)]:
                return None
    return data


def get_experiment_subjects(experiment):
    data_path = "../data/experiment_" + str(experiment) + "/"
    subject_names = []
    for subject_file in os.listdir(data_path):
        valid = check_subject_validity(experiment, data_path + subject_file)
        if valid:
            subject_names.append(subject_file.strip('.json'))

    return subject_names


if __name__ == "__main__":
    get_experiment_subjects(experiment="instr_1")


