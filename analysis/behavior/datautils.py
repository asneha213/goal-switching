import os
import json
import pandas as pd
import numpy as np


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
        details["num_samples"] = 360

    elif experiment == 'bandit':
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2

    elif experiment == 4:
        details["num_episodes"] = 12
        details["num_trials"] = 30
        details["num_conditions"] = 2
        details["num_samples"] = 360

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


def get_experiment_subjects(experiment, data_path=None):
    if data_path is None:
        data_path = "../data/experiment_" + str(experiment) + "/"
    subject_names = []
    for subject_file in os.listdir(data_path):
        valid = check_subject_validity(experiment, data_path + subject_file)
        if valid:
            subject_names.append(subject_file.strip('.json'))

    return subject_names

def get_demographic_info(experiment):
    data_path = "../data/experiment_" + str(experiment) + "/"
    subject_names = get_experiment_subjects(experiment, data_path)
    sub_names = [i.split('_')[0] for i in subject_names]
    filenames = os.listdir(data_path)
    demo_file = [f for f in filenames if f.startswith('prolific')]

    ages = []
    sex = []


    for df in demo_file:
        demo_info = pd.read_csv(data_path + df)
        demo_info = demo_info[demo_info['Participant id'].isin(sub_names)]
        participants = demo_info['Participant id'].to_list()
        ages.extend(demo_info['Age'].to_list())
        sex.extend(demo_info['Sex'].to_list())

    participants = np.array(participants)
    subject_names = np.array(subject_names)
    ages = np.array(ages)

    subject_ages = {}

    for subject_id in subject_names:
        sub_id = subject_id.split('_')[0]
        age = ages[np.where(participants == sub_id)]
        subject_ages[subject_id] = age[0]


    ages = [int(age) for age in ages if age != 'CONSENT_REVOKED' and age!='DATA_EXPIRED']
    sex = [s for s in sex if s != 'CONSENT_REVOKED' and s!='DATA_EXPIRED']
    print("Sex count: ", sex.count('Female'), sex.count('Male'))
    print("Age: ", min(ages), max(ages), np.mean(ages), np.std(ages))

    return subject_ages



if __name__ == "__main__":
    #get_experiment_subjects(experiment="instr_1")
    get_demographic_info(experiment=2)


