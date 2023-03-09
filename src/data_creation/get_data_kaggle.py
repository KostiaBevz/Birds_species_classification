import os
import sys

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.append("./")
import config

# TODO: command line
# import click
import logger

"""
If I want to download data from competition need to use
    api.competition_download_cli(competition_opt=COMPETITION, path=PATH) or
    api.competition_download_file() or
    api.competition_download_files()

"""
if __name__ == "__main__":
    log = logger.log
    api = KaggleApi()
    api.authenticate()
    path = config.DATA_DIR[:-1]
    dataset = config.DATASET
    file_name = config.ANNOTATION_FILE_NAME
    log.info("Downloading data")
    api.dataset_download_cli(dataset=dataset, path=path, unzip=True)
    log.info("Done")
    # Generate annotations
    if file_name not in os.listdir(path):
        log.info("Creating annotations")
        df = pd.DataFrame(
            columns=["class_id", "class_name", "absolute_path", "dataset_type"]
        )

        directories = ["test/", "train/", "validation/"]

        data = dict()
        for directory in directories:
            data["dataset_type"] = directory[:-1]
            directory_path = os.path.join(path, directory)
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

        df.to_csv(path + file_name, index=False)
