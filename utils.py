import os
from typing import Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Sampler as _SamplerType
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

import logger
from datasets.fruits_veg_dataset import Fruits_and_vegetables_dataset

log = logger.log


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
    log.info("Calculating mean, std of dataset")
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
    num_workers: Optional[int] = os.cpu_count(),
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
    log.info("Creating data sampler")
    class_weights = []
    for root, sub_dir, files in os.walk(root_dir + train_data_placeholder):
        if files:
            class_weights.append(1 / len(files))
    sample_weights = [0] * len(dataset)
    if dataloader.batch_size > 1:
        indx = 0
        for _, labels in dataloader:
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
