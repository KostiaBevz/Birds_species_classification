import gc
import os
import sys

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

seed = 102  # ask q
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)


#  TODO: play with tensorboard
#  TODO: tests coverage
#  TODO: mlflow
#  TODO: deploy model, add CI/CD to github
#  TODO: think about try/except constructions in get_data_kaggle.py


if __name__ == "__main__":
    """
    Optimal BATCH_SIZE=8 for my cpu memory if choose more
    aggresive numbers machine start lagging much
    """
    log = logger.log
    device = torch.device("mps")
    data = pd.read_csv(config.DATA_DIR + config.ANNOTATION_FILE_NAME)
    NUM_CLASSES = data["class_id"].nunique()
    train_data_placeholder = "train"

    datasets, data_loaders = create_dataset_and_dataloader(
        file_name=config.ANNOTATION_FILE_NAME,
        root_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    if f"{config.DATASET_NAME}.pt" not in os.listdir(config.DATASET_DIR):
        mean, var, std = calculate_stat_of_input_dataset(
            data_loaders[train_data_placeholder]
        )
        collection = {"mean": mean, "var": var, "std": std}
        torch.save(
            collection, config.DATASET_DIR + f"{config.DATASET_NAME}.pt"
        )
    else:
        log.info("Getting dataset stats from .pth file")
        stat_data_tensors = torch.load(
            config.DATASET_DIR + f"{config.DATASET_NAME}.pt"
        )
        mean = stat_data_tensors["mean"]
        std = stat_data_tensors["std"]

    sampler = create_custom_sampler(
        root_dir=config.DATA_DIR,
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
        file_name=config.ANNOTATION_FILE_NAME,
        root_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        transformation=custom_transform,
        num_workers=config.NUM_WORKERS,
        sampler=sampler,
    )
    gc.collect()

    model = ResNet(
        in_channels=config.IMAGE_NUMBER_CHANNELS,
        num_classes=NUM_CLASSES,
        layers=config.RES_NET_CONFIG,
    ).to(device)

    # Model constants
    learning_rate = 0.1
    number_of_epochs = 2  # 10-15
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer, verbose=True, gamma=0.1
    # )
    metric = MulticlassF1Score(num_classes=NUM_CLASSES, device=device)
    try:
        experiment_id = mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        log.error(e)
        experiment_id = mlflow.get_experiment_by_name(
            config.MLFLOW_EXPERIMENT_NAME
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
        experiment_id=experiment_id
        # scheduler=scheduler,
    )
