import os
import json
import pandas as pd


def get_list_subjects(experiment, data_dir="../data/experiment2/"):
    subject_path_list = {}
    subject_names = []
    for subject_file in os.listdir(data_dir):
        data = get_subject_data(experiment, data_dir + subject_file)
        if not data:
            continue
        subject_name = subject_file.split('_')[0]
        subject_path_list[subject_name] = data_dir + subject_file
        subject_names.append(subject_name)
    return subject_path_list, subject_names


def get_subject_data(experiment, sub_path):
    if sub_path.endswith('.json') == False:
        return None
    fp = open(sub_path)
    data = json.load(fp)
    if "GoalSwitching" not in data:
        return None
    if "bonus" not in data:
        return None
    if experiment == 1:
        blocks = 18
    elif experiment == 2:
        blocks = 12
    for i in range(blocks):
        if str(i) not in data['GoalSwitching']:
            return None
    return data


SUBJECT_LIST_1, SUBJECT_NAMES_1 = get_list_subjects(experiment=1, data_dir="../data/experiment1/")
SUBJECT_LIST_2, SUBJECT_NAMES_2= get_list_subjects(experiment=2, data_dir="../data/experiment2/")

def get_subject_data_from_id(experiment, subject_id):
    if experiment == 1:
        SUBJECT_LIST = SUBJECT_LIST_1
    elif experiment == 2:
        SUBJECT_LIST = SUBJECT_LIST_2
    subject_path = SUBJECT_LIST[subject_id]
    data = get_subject_data(experiment, subject_path)
    return data


def get_experiment_logistics(experiment):
    if experiment == 1:
        num_subjects = 69
        SUBJECT_NAMES = SUBJECT_NAMES_1
        num_samples = 540
    elif experiment == 2:
        num_subjects = 50
        SUBJECT_NAMES = SUBJECT_NAMES_2
        num_samples = 360
    return num_subjects, SUBJECT_NAMES, num_samples


def get_demographic_dataframe(experiment):
    if experiment == 1:
        data_dir = "../data/experiment1/"
        file_0 = "prolific_export_6410f1a2630ee8762a725350.csv"
        file_1 = "prolific_export_644bf6650c0b6fb4cb71eb1a.csv"
    elif experiment == 2:
        data_dir = "../data/experiment2/"
        file_0 = "prolific_export_6477c2e914692d4724e21008.csv"

    df = pd.read_csv(data_dir + file_0)
    df0 = df[["Participant id", "Age"]]

    if experiment == 1:
        df = pd.read_csv(data_dir + file_1)
        df1 = df[["Participant id", "Age"]]
        df = pd.concat([df0, df1])
    else:
        df = df0
    df = df.dropna()

    return df


def get_age_from_id(experiment, subject_id):
    if experiment == 1:
        SUBJECT_LIST = SUBJECT_LIST_1
    elif experiment == 2:
        SUBJECT_LIST = SUBJECT_LIST_2
    df = get_demographic_dataframe(experiment)
    subjects = [i.split('/')[-1].split('_')[0] for i in SUBJECT_LIST]
    subject_prolific_id = subjects[subject_id]
    age = df[df["Participant id"] == subject_prolific_id]["Age"].values[0]
    return age

