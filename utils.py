import os
from os import cpu_count
from typing import Any, Optional, Tuple

import torch
import torcheval.metrics
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Sampler as _SamplerType
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from datasets.fruits_veg_dataset import Fruits_and_vegetables_dataset

# TODO: add CLI, add kaggle auth, logging


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: torch.nn,
    optimizer: torch.optim.Optimizer,
    metric: torcheval.metrics.Metric,
    num_epochs: Optional[int] = 5,
    device: Optional[torch.device] = torch.device("mps"),
    scheduler: Optional[Any] = None,
) -> None:
    """
    Perform model training

    Args:
        model: torch.nn.Module
            Pytorch model for training
        train_loader: DataLoader
            Data loader for training set
        validation_loader: DataLoader
            Data loader for validation set
        loss_fn: torch.nn
            Predefined loss function
        num_epochs: Optional[int]
            Number of epochs to train
        device: Optional[torch.device]
            Device to perform calculation
        scheduler: Optional[Any]
            Scheduler to adjust learning rate per epoch

    Returns:
        None
    """
    running_loss = 0.0
    avg_loss = 0.0
    len_of_data = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), leave=False, total=len_of_data)
        running_loss = 0.0
        for batch_indx, (input, labels) in loop:
            input, labels = input.to(device), labels.to(device)

            optimizer.zero_grad()

            out = model(input)

            loss = loss_fn(out, labels)
            loss.backward()
        #  Equal to torch.nn.functional.softmax(out, dim=1).max(dim=1).indices
            out = torch.max(out, dim=1).indices
            metric.update(out, labels)

            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            running_loss += loss.item()

        avg_loss = running_loss / (batch_indx + 1)
        train_f1_score = metric.compute()
        metric.reset()
        if scheduler:
            scheduler.step()

        # Evaluation phase
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for vbatch_indx, (vinputs, vlabels) in enumerate(
                validation_loader
            ):
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                voutputs = torch.max(voutputs, dim=1).indices
                metric.update(voutputs, vlabels)
                running_vloss += vloss
            avg_vloss = running_vloss / (vbatch_indx + 1)
        valid_f1_score = metric.compute()
        print(
            "LOSS: train {} / valid {}, F1_score: train {} / valid {}".format(
                avg_loss, avg_vloss, train_f1_score, valid_f1_score
            )
        )
        metric.reset()


def calculate_stat_of_input_dataset(
    dataloader: DataLoader,
) -> Tuple[float, float, float]:
    """
    Calculates mean, variance and standart deviation of images dataset

    Args:
        dataloader: DataLoader
            Pytorch data loader with defined pytorch dataset

    Returns:
        Tuple[float, float, float]
            Values for mean, variance and standart deviation of dataset

    """
    print("Calculating mean, std of dataset")
    channel_sum, channel_sum_squared, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader):
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_sum_squared += torch.mean(data**2, dim=[0, 2, 3])

        num_batches += 1
    total_mean = channel_sum / num_batches
    total_var = channel_sum_squared / num_batches - total_mean**2
    total_std = total_var**0.5

    return total_mean, total_var, total_std


def create_dataset_and_dataloader(
    file_name: str,
    root_dir: str,
    batch_size: Optional[int] = 16,
    transformation: Optional[torchvision.transforms.Compose] = None,
    num_workers: Optional[int] = cpu_count(),
    sampler: Optional[_SamplerType] = None,
    shuffle: Optional[bool] = True,
) -> dict[str, DataLoader]:
    """
    Creating train, test and validation datasets
    and constructing pytorch dataloaders

    Args:
        file_name: str
            Name of csv file with data annotations
        root_dir: str
            Name of root directory where data lie
        batch_size: int
            Number of samples to take in one iteration
        transformation: torchvision.transform
            Transformations applied to the initial image
        sampler: torch.utils.data.WeightedRandomSampler
            Sampler for imbalanced datasets
        shuffle: boot
            To have the data reshuffled at every epoch

    Returns:
        Dict(str, DataLoader)
            Dictionary of data loaders for each dataset
            (i.e. train, test, validation)
    """
    datasets_types = ["train", "validation", "test"]
    datasets = {
        dataset_type: Fruits_and_vegetables_dataset(
            csv_file=file_name,
            root_dir=root_dir,
            dataset_type=dataset_type,
            transform=transformation if transformation else None,
        )
        for dataset_type in datasets_types
    }
    data_loaders = {
        dataset: DataLoader(
            datasets[dataset],
            batch_size=batch_size,
            shuffle=shuffle if dataset != "train" else False,
            num_workers=num_workers,
            sampler=sampler if (sampler and dataset == "train") else None,
        )
        for dataset in datasets_types
    }
    return datasets, data_loaders


def create_custom_sampler(
    root_dir: str,
    dataset: Fruits_and_vegetables_dataset,
    dataloader: DataLoader,
    train_data_placeholder: Optional[str] = "train",
) -> WeightedRandomSampler:
    """
    Create custom WeightedRandomSampler from pytorch
    to deal with imbalanced dataset

    Args:
        root_dir: str
            Root directory with data files
        dataset: Fruits_and_vegetables_dataset
            Custom dataset with train data created with function below
        dataloader: DataLoader
            DataLoader with train data
        train_data_placeholder: str
            Placeholder for os.walk proper search

    Returns:
        sampler: WeightedRandomSampler
            Custom sampler for imbalanced dataset
    """
    print("Creating data sampler")
    class_weights = []
    for root, sub_dir, files in os.walk(root_dir + train_data_placeholder):
        if files:
            class_weights.append(1 / len(files))
    sample_weights = [0] * len(dataset)
    if dataloader.batch_size > 1:
        indx = 0
        for (_, labels) in dataloader:
            for label in labels:
                class_w = class_weights[label]
                sample_weights[indx] = class_w
                indx += 1
    else:
        for idx, (_, label) in enumerate(dataloader):
            class_w = class_weights[label]
            sample_weights[idx] = class_w
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler
