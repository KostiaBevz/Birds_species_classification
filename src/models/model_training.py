import gc
import os
import sys
from typing import Optional

import pandas as pd
import torch
from torcheval.metrics import MulticlassF1Score
from torchvision import transforms

sys.path.append("./")
import config
import logger
from models.ResNet import ResNet
from utils import (
    calculate_stat_of_input_dataset,
    create_custom_sampler,
    create_dataset_and_dataloader,
)
import mlflow
import click

seed = 102  # ask q
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)


# TODO: add proper inference
# TODO: add visualisation
# TODO: add possibility to start training with given model_state
# TODO: tests coverage
# TODO: deploy model, add CI/CD to github


@click.command()
@click.option(
    "-lr", "learning_rate",
    default=0.1,
    type=float,
    help="Learning rate used for training",
)
@click.option(
    "-epochs", "number_of_epochs",
    default=2,
    type=int,
    help="Number of epochs for training",
)
@click.option(
    "--mlflow_exp_name",
    default=config.MLFLOW_EXPERIMENT_NAME,
    type=str,
    help="Mlflow experiment name",
)
def main(
    annotation_file_name: Optional[str] = config.ANNOTATION_FILE_NAME,
    data_dir: Optional[str] = config.DATA_DIR,
    dataset_dir: Optional[str] = config.DATASET_DIR,
    batch_size: Optional[int] = config.BATCH_SIZE,
    num_workers: Optional[int] = config.NUM_WORKERS,
    dataset_name: Optional[str] = config.DATASET_NAME,
    image_number_channels: Optional[tuple] = config.IMAGE_NUMBER_CHANNELS,
    resnet_config: Optional[list] = config.RES_NET_CONFIG,
    mlflow_exp_name: Optional[str] = config.MLFLOW_EXPERIMENT_NAME,
    learning_rate: Optional[float] = 0.1,
    number_of_epochs: Optional[int] = 2,
    is_scheduler: Optional[bool] = False,
):
    """
    Wrapper around train_model function for CLI

    Args:
        annotation_file_name (str, optional):
            Name of file where annotations stored.
            Defaults to config.ANNOTATION_FILE_NAME.
        data_dir (str, optional):
            Path to the directory with data.
            Defaults to config.DATA_DIR.
        dataset_dir (str, optional):
            Path to the directory with datasets stats.
            Defaults to config.DATASET_DIR.
        batch_size (int, optional):
            Batch size value for data loaders creation.
            Defaults to config.BATCH_SIZE.
        num_workers (int, optional):
            Number of workers for multiprocessing.
            Defaults to config.NUM_WORKERS.
        dataset_name (str, optional):
            Name of dataset file where statistical data stored.
            Defaults to config.DATASET_NAME.
        image_number_channels (tuple, optional):
            Number of channels for images in dataset, in majority cases 1 or 3.
            Defaults to config.IMAGE_NUMBER_CHANNELS.
        resnet_config (list, optional):
            List with values that represent number of skip connectin layers.
            Defaults to config.RES_NET_CONFIG.
        mlflow_exp_name (str, optional):
            Name of mlflow experiment to store model,
            training metrics and other artifacts.
            Defaults to config.MLFLOW_EXPERIMENT_NAME.
        learning_rate (float, optional):
            Learning rate used for training.
            Defaults to 0.1.
        number_of_epochs (int, optional):
            Number of epochs for training.
            Defaults to 10.
        is_scheduler (bool, optional):
            Do we need to use scheduler or not.
            Defaults to False.
    """
    train_model(
        annotation_file_name,
        data_dir,
        dataset_dir,
        batch_size,
        num_workers,
        dataset_name,
        image_number_channels,
        resnet_config,
        mlflow_exp_name,
        learning_rate,
        number_of_epochs,
        is_scheduler
    )


def train_model(
    annotation_file_name: Optional[str] = config.ANNOTATION_FILE_NAME,
    data_dir: Optional[str] = config.DATA_DIR,
    dataset_dir: Optional[str] = config.DATASET_DIR,
    batch_size: Optional[int] = config.BATCH_SIZE,
    num_workers: Optional[int] = config.NUM_WORKERS,
    dataset_name: Optional[str] = config.DATASET_NAME,
    image_number_channels: Optional[tuple] = config.IMAGE_NUMBER_CHANNELS,
    resnet_config: Optional[list] = config.RES_NET_CONFIG,
    mlflow_exp_name: Optional[str] = config.MLFLOW_EXPERIMENT_NAME,
    learning_rate: Optional[float] = 0.1,
    number_of_epochs: Optional[int] = 2,
    is_scheduler: Optional[bool] = False,
):
    """
    Main script for processing dataset and model training

    Args:
        annotation_file_name (str, optional):
            Name of file where annotations stored.
            Defaults to config.ANNOTATION_FILE_NAME.
        data_dir (str, optional):
            Path to the directory with data.
            Defaults to config.DATA_DIR.
        dataset_dir (str, optional):
            Path to the directory with datasets stats.
            Defaults to config.DATASET_DIR.
        batch_size (int, optional):
            Batch size value for data loaders creation.
            Defaults to config.BATCH_SIZE.
        num_workers (int, optional):
            Number of workers for multiprocessing.
            Defaults to config.NUM_WORKERS.
        dataset_name (str, optional):
            Name of dataset file where statistical data stored.
            Defaults to config.DATASET_NAME.
        image_number_channels (tuple, optional):
            Number of channels for images in dataset, in majority cases 1 or 3.
            Defaults to config.IMAGE_NUMBER_CHANNELS.
        resnet_config (list, optional):
            List with values that represent number of skip connectin layers.
            Defaults to config.RES_NET_CONFIG.
        mlflow_exp_name (str, optional):
            Name of mlflow experiment to store model,
            training metrics and other artifacts.
            Defaults to config.MLFLOW_EXPERIMENT_NAME.
        learning_rate (float, optional):
            Learning rate used for training.
            Defaults to 0.1.
        number_of_epochs (int, optional):
            Number of epochs for training.
            Defaults to 10.
        is_scheduler (bool, optional):
            Do we need to use scheduler or not.
            Defaults to False.
    """
    log = logger.log
    device = torch.device("mps")
    data = pd.read_parquet(config.DATA_DIR + config.ANNOTATION_FILE_NAME)
    NUM_CLASSES = data["class_id"].nunique()
    train_data_placeholder = "train"
    datasets, data_loaders = create_dataset_and_dataloader(
        file_name=annotation_file_name,
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if f"{dataset_name}.pt" not in os.listdir(dataset_dir):
        mean, var, std = calculate_stat_of_input_dataset(
            data_loaders[train_data_placeholder]
        )
        collection = {"mean": mean, "var": var, "std": std}
        torch.save(collection, dataset_dir + f"{dataset_name}.pt")
    else:
        log.info("Getting dataset stats from .pth file")
        stat_data_tensors = torch.load(dataset_dir + f"{dataset_name}.pt")
        mean = stat_data_tensors["mean"]
        std = stat_data_tensors["std"]
    sampler = create_custom_sampler(
        root_dir=data_dir,
        dataset=datasets.get(train_data_placeholder),
        dataloader=data_loaders.get(train_data_placeholder),
        train_data_placeholder=train_data_placeholder,
    )
    custom_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    _, data_loaders = create_dataset_and_dataloader(
        file_name=annotation_file_name,
        root_dir=data_dir,
        batch_size=batch_size,
        transformation=custom_transform,
        num_workers=num_workers,
        sampler=sampler,
    )
    gc.collect()
    model = ResNet(
        in_channels=image_number_channels,
        num_classes=NUM_CLASSES,
        layers=resnet_config,
    ).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, verbose=True, gamma=0.1
        )
    metric = MulticlassF1Score(num_classes=NUM_CLASSES, device=device)
    try:
        experiment_id = mlflow.create_experiment(mlflow_exp_name)
    except Exception as e:
        log.error(e)
        experiment_id = mlflow.get_experiment_by_name(
            mlflow_exp_name
        ).experiment_id
    log.info("Mflow setup")
    log.info(f"Experiment_id : {experiment_id}")
    model.train_model(
        model=model,
        train_loader=data_loaders["train"],
        valid_loader=data_loaders["validation"],
        loss_fn=loss,
        optimizer=optimizer,
        metric=metric,
        num_epochs=number_of_epochs,
        device=device,
        experiment_id=experiment_id,
        scheduler=scheduler if is_scheduler else None,
    )


if __name__ == "__main__":
    """
    Optimal BATCH_SIZE=8 for my cpu memory if choose more
    aggresive numbers machine start lagging much
    """
    train_model(number_of_epochs=2)
    # TODO: ask F
    # import click

    # @click.command()
    # @click.argument('arg1')
    # @click.argument('arg2')
    # def my_command(arg1, arg2, arg3):
    #     # Do something with the arguments
    #     print(f"arg1 = {arg1}, arg2 = {arg2}")

    # my_command(arg1=1, arg2=2)
