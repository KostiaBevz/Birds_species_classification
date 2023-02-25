import gc

import pandas as pd
import torch
from torcheval.metrics import MulticlassF1Score
from torchvision import transforms

from Models.VGG_net import VGG, VGG_Net
from utils import (
    calculate_stat_of_input_dataset,
    create_dataset_and_dataloader,
    train_model,
    create_custom_sampler,
)

seed = 102  # ask q
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)


#  TODO: add Callbacks to training loop
#  TODO: play with tensorboard
#  TODO: tests coverage
#  TODO: mlflow
#  TODO: deploy model, add CI/CD to github

# TODO: ask Fred about multiprocessing related to __main__
if __name__ == "__main__":
    """
    Optimal BATCH_SIZE=8 for my cpu memory if choose more
    aggresive numbers machine start lagging much
    """
    BATCH_SIZE = 16
    ROOT_DIR = "/Users/kostiantyn/Desktop/Birds_species_classification/Data/"
    FILE_NAME = "annotation.csv"
    device = torch.device("mps")
    data = pd.read_csv(ROOT_DIR + "annotation.csv")
    NUM_CLASSES = data["class_id"].nunique()
    train_data_placeholder = "train"
    num_workers = 2

    datasets, data_loaders = create_dataset_and_dataloader(
        file_name=FILE_NAME,
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    sampler = create_custom_sampler(
        root_dir=ROOT_DIR,
        dataset=datasets.get(train_data_placeholder),
        dataloader=data_loaders.get(train_data_placeholder),
        train_data_placeholder=train_data_placeholder,
    )

    mean, var, std = calculate_stat_of_input_dataset(
        data_loaders[train_data_placeholder]
    )

    custom_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    img_size, _ = next(iter(data_loaders[train_data_placeholder]))

    _, data_loaders = create_dataset_and_dataloader(
        file_name=FILE_NAME,
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        transformation=custom_transform,
        num_workers=num_workers,
        sampler=sampler,
    )
    gc.collect()

    model = VGG_Net(
        in_channels=img_size.shape[1],
        num_classes=NUM_CLASSES,
        model_structure=VGG,
        image_size=img_size.shape[2],
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

    train_model(
        model=model,
        train_loader=data_loaders["train"],
        validation_loader=data_loaders["validation"],
        loss_fn=loss,
        optimizer=optimizer,
        metric=metric,
        num_epochs=number_of_epochs,
        device=device,
        # scheduler=scheduler,
    )
