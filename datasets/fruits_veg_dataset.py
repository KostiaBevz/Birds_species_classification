import sys
from typing import Optional, Tuple

sys.path.append("./")
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import config


class Fruits_and_vegetables_dataset(Dataset):
    """
    Fruits and Vegetables Image Recognition pytorch dataset
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = "./",
        column_name: Optional[str] = "dataset_type",
        dataset_type: Optional[str] = "train",
        transform: Optional[T.Compose] = None,
        img_size: Optional[Tuple] = config.IMAGE_SIZE
    ) -> None:
        """
        Args:
            csv_file: string
                 Path to the csv file with data
            root_dir: string
                Directory with all images
            column_name: string
                Column name in processed dataframe
            dataset_type: string
                dataset to use 'train'/'valid'/'test'
            transform: Optional[T.Compose]
                dataset transformations
            img_size: Optional[Tuple]
                transformation of images size
        """
        data = pd.read_csv(root_dir + csv_file)
        data = data[data[column_name] == dataset_type]
        self.data = data.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        """
        Returns:
            Len of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index: integer
                Method that get sample of data from our dataset

        Returns:
            Tuple[torch.Tensor, int]
                Where image take place of an image presented in dataset
                with corresponding label

        """
        if torch.is_tensor(index):
            index = index.tolist()

        image_location = self.data.iloc[index, 2]
        image = np.uint8(cv2.cvtColor(
            cv2.imread(image_location), cv2.COLOR_BGR2RGB))
        image = cv2.resize(image, self.img_size)
        label = self.data.iloc[index, 0]
        if self.transform:
            image = self.transform(image)
        else:
            """
            swap color axis because
            numpy image: H x W x C
            torch image: C x H x W
            """
            # image = image.transpose((2, 0, 1))/255.
            # image = torch.from_numpy(image)
            """
            Better to use torchvision.transforms.ToTensor() do the same
            stuff but image need to be in np.uint8 type
            """
            image = T.ToTensor()(image)
        return image, label
