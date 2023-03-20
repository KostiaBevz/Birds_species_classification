import os
import sys

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.append("./")
import click

import config
import logger

"""
If I want to download data from competition need to use
    api.competition_download_cli(competition_opt=COMPETITION, path=PATH) or
    api.competition_download_file() or
    api.competition_download_files()
"""
log = logger.log
try:
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    log.error(e)


@click.command()
@click.option(
    "--dataset", default=config.DATASET, type=str, help="Kaggle dataset url"
)
@click.option(
    "--path",
    default=config.DATA_DIR[:-1],
    type=click.Path(dir_okay=True),
    help="Path to the data",
)
@click.option(
    "--force/--no-force",
    default=False,
    type=bool,
    help="Force reinstall of dataset from kaggle",
)
@click.option(
    "-f", "file_name",
    default=config.ANNOTATION_FILE_NAME,
    type=str,
    help="Name of file where to store annotations",
)
def download_kaggle_dataset(
    dataset: str = config.DATASET,
    path: str = config.DATA_DIR[:-1],
    file_name: str = config.ANNOTATION_FILE_NAME,
    force: bool = False,
) -> None:
    """
    Function perform download data from kaggle, using kaggle API,
    then perform annotations creation if needed.

    Args:
        dataset (str, optional):
            Full URI of dataset in kaggle.com.
            Defaults to config.DATASET.
        path (_type_, optional):
            Path to the directory where to store data.
            Defaults to config.DATA_DIR[:-1].
        force (bool, optional):
            If True force reinstall of dataset from kaggle.
            Defaults to False.
    """
    log.info("Downloading data")
    api.dataset_download_cli(
        dataset=dataset, path=path, unzip=True, force=force
    )
    if file_name not in os.listdir(path):
        generate_annotations()
    log.info("Done")


@click.command()
@click.option(
    "--file_name",
    default=config.ANNOTATION_FILE_NAME,
    type=str,
    help="Name of file where to store annotations",
)
@click.option(
    "--path",
    default=config.DATA_DIR[:-1],
    type=click.Path(dir_okay=True),
    help="Path to the data",
)
def generate_annotations(
    file_name: str = config.ANNOTATION_FILE_NAME,
    path: str = config.DATA_DIR[:-1],
):
    """
    Function create annotation file if it doesn't exist.

    Args:
        file_name (str, optional):
            Name of file to store annotations.
            Defaults to config.ANNOTATION_FILE_NAME.
        path (str, optional):
            Path where all data stored.
            Defaults to config.DATA_DIR[:-1].
    """
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

        df.to_parquet(path + "/" + file_name, index=False)


if __name__ == "__main__":
    download_kaggle_dataset()
