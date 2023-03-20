import sys

sys.path.append("./")
import pandas as pd
import torch

import config
from models.ResNet import ResNet
from utils import create_dataset_and_dataloader
from torchvision import transforms
from torcheval.metrics import MulticlassF1Score
import mlflow

device = torch.device("mps")
data = pd.read_csv(config.DATA_DIR + config.ANNOTATION_FILE_NAME)
NUM_CLASSES = data["class_id"].nunique()

# TODO: refactor -> move to the BaseModel as a method

experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
df = mlflow.search_runs(
    experiment_names=[config.MLFLOW_EXPERIMENT_NAME],
    order_by=["metrics.best_valid_loss DESC"],
)
df = mlflow.search_runs(experiment_names=["ResNet"]).sort_values(
    by=["metrics.validation f1 score"]
)
path = df.iloc[0]["artifact_uri"] + "/" + config.MLFLOW_ARTIFACT_PATH
model = ResNet(
    in_channels=config.IMAGE_NUMBER_CHANNELS,
    num_classes=NUM_CLASSES,
    layers=config.RES_NET_CONFIG,
).to(device)
model = mlflow.pytorch.load_state_dict(path)
