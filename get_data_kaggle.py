import os

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
DATASET = "kbevzuk/fruits-vegetables-classification-modified"  # dataset name
PATH = "Data/"  # path to data
"""
If I want to download data from competition need to use
    api.competition_download_cli(competition_opt=COMPETITION, path=PATH) or
    api.competition_download_file() or
    api.competition_download_files()

"""

api.dataset_download_cli(dataset=DATASET, path=PATH, unzip=True)

# Generate annotations

file_name = "annotation.csv"

if file_name not in os.listdir(PATH):
    df = pd.DataFrame(
        columns=["class_id", "class_name", "absolute_path", "dataset_type"]
    )

    directories = ["test/", "train/", "validation/"]

    data = dict()
    for directory in directories:
        data["dataset_type"] = directory[:-1]
        directory_path = os.path.join(PATH, directory)
        for indx, class_name in enumerate(os.listdir(directory_path)):
            data["class_id"] = indx
            data["class_name"] = class_name
            class_directory_path = os.path.join(
                directory_path, class_name + "/"
            )
            for img in os.listdir(class_directory_path):
                img_path = os.path.join(class_directory_path, img)
                data["absolute_path"] = img_path
                entry = pd.DataFrame([data])
                df = pd.concat([df, entry], ignore_index=True)

    df.to_csv(PATH + file_name, index=False)
