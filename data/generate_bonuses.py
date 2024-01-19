import os
import json
import pandas as pd
import numpy as np


def get_bonuses(data_dir="experiment_1/"):
    df = pd.DataFrame(columns=["subjectID", "bonus"])
    bonuses = []
    for subject_file in os.listdir(data_dir):
        if subject_file.endswith(".json") is False:
            continue
        sub_path = data_dir + subject_file
        fp = open(sub_path)
        data = json.load(fp)
        if "bonus" not in data:
            continue
        bonus = data["bonus"]
        if bonus < 10:
            continue
        subjectID = data["subjectID"]
        sub_dict = {"subjectID": subjectID, "bonus": bonus}
        df = pd.concat([df, pd.DataFrame([sub_dict])], ignore_index=True)
        bonuses.append(bonus)

    bonuses = np.array(bonuses)

    bonus_max = 3.0
    bonus_min = 0.5


    bonus_slope =  (bonus_max - bonus_min) / (np.max(bonuses) - np.min(bonuses))

    df["scaled_bonus"] = bonus_min + (df["bonus"] - np.min(bonuses)) *  bonus_slope
    df['scaled_bonus'] = df['scaled_bonus'].apply(lambda x: round(x, 2))
    df = df.set_index("subjectID")
    df[["scaled_bonus"]].to_csv("bonus_" + data_dir.strip('/') + ".csv")


if __name__ == "__main__":
    get_bonuses()