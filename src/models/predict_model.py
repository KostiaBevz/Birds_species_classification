import sys

sys.path.append("./")
import pandas as pd
import torch

import config
from models.ResNet import ResNet
from utils import create_dataset_and_dataloader
from torchvision import transforms
from torcheval.metrics import MulticlassF1Score

device = torch.device("mps")
data = pd.read_csv(config.DATA_DIR + config.ANNOTATION_FILE_NAME)
NUM_CLASSES = data["class_id"].nunique()

# TODO: refactor

model = ResNet(
    in_channels=config.IMAGE_NUMBER_CHANNELS,
    num_classes=NUM_CLASSES,
    layers=config.RES_NET_CONFIG,
).to(device)
optimizer = torch.optim.Adam(model.parameters())
checkpoint = torch.load(
    config.TRAINED_MODELS_DIRECTORY + config.CHECKPOINT_MODEL_FILE_NAME
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
print(epoch, loss)

stat_data_tensors = torch.load(
    config.DATASET_DIR + f"{config.DATASET_NAME}.pt"
)
mean = stat_data_tensors["mean"]
std = stat_data_tensors["std"]
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
    num_workers=0,
)
metric = MulticlassF1Score(num_classes=NUM_CLASSES, device=device)
validation_loader = data_loaders['validation']
running_vloss = 0
loss = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    for vbatch_indx, (vinputs, vlabels) in enumerate(
        validation_loader
    ):
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        vloss = loss(voutputs, vlabels)

        voutputs = torch.max(voutputs, dim=1).indices
        metric.update(voutputs, vlabels)
        running_vloss += vloss
    avg_vloss = running_vloss / (vbatch_indx + 1)
valid_f1_score = metric.compute()
print(valid_f1_score)
