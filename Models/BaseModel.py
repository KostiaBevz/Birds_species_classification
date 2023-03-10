import sys
from abc import ABC
from typing import Dict, Optional

sys.path.append("./")

from copy import deepcopy

import torch
import torcheval.metrics.metric
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import logger
import mlflow

log = logger.log


class BaseModel(ABC):
    def __init__(self):
        pass

    def save_model():
        pass

    @staticmethod
    def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: torch.nn,
        optimizer: torch.optim.Optimizer,
        metric: torcheval.metrics.Metric,
        num_epochs: Optional[int] = 5,
        device: Optional[torch.device] = torch.device("mps"),
        scheduler: Optional[_LRScheduler] = None,
        max_patience: Optional[int] = 4,
        best_loss: Optional[float] = 10000.0,
        best_model: Optional[Dict] = {},
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        Perform model training

        Args:
            model: torch.nn.Module
                Pytorch model for training
            train_loader: DataLoader
                Data loader for training set
            valid_loader: DataLoader
                Data loader for validation set
            loss_fn: torch.nn
                Predefined loss function
            num_epochs: Optional[int]
                Number of epochs to train
            device: Optional[torch.device]
                Device to perform calculation
            scheduler: Optional[Any]
                Scheduler to adjust learning rate per epoch
            best_loss: Optional[float]
                Best loss of pretrained model,
                used only when initial training was performed before
            best_model: Optional[Dict]
                Best model state_dict,
                used only when initial training was performed before

        Returns:
            None
        """
        log.info("Training start")
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        mlflow.start_run(
            run_name=f"ResNet_{len(runs)}",
            experiment_id=experiment_id,
            tags={"version": f"v{len(runs)}"},
            description="ResNet model training experiments",
        )
        best_loss = best_loss
        patience = 0
        best_model = best_model
        len_of_data = len(train_loader)
        for epoch in range(num_epochs):
            if patience == max_patience:
                break
            model.train()
            loop = tqdm(
                enumerate(train_loader), leave=False, total=len_of_data
            )
            running_loss = 0.0
            for batch_indx, (input, labels) in loop:
                input, labels = input.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(input)
                loss = loss_fn(out, labels)
                loss.backward()
                """
                Equal to
                torch.nn.functional.softmax(out, dim=1).max(dim=1).indices
                """
                out = torch.max(out, dim=1).indices
                metric.update(out, labels)

                optimizer.step()

                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

                running_loss += loss.item()

            avg_loss = running_loss / (batch_indx + 1)
            train_f1_score = metric.compute()
            metric.reset()
            if scheduler:
                scheduler.step()
            # Save checkpoint
            if epoch % 5 == 0:
                log.info("Saving checkpoint")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    config.TRAINED_MODELS_DIRECTORY
                    + config.CHECKPOINT_MODEL_FILE_NAME,
                )

            # Evaluation phase
            model.eval()
            running_vloss = 0.0
            with torch.no_grad():
                for vbatch_indx, (vinputs, vlabels) in enumerate(valid_loader):
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)

                    voutputs = torch.max(voutputs, dim=1).indices
                    metric.update(voutputs, vlabels)
                    running_vloss += vloss
                avg_vloss = running_vloss / (vbatch_indx + 1)
            valid_f1_score = metric.compute()
            mlflow.log_metrics(
                {
                    "validation f1 score": valid_f1_score.item(),
                    "validation loss ": float(avg_vloss),
                    "train loss": float(avg_loss),
                    "train f1 score": train_f1_score.item(),
                },
                step=epoch,
            )
            print(
                f"LOSS: train {avg_loss} / valid {avg_vloss} \
                \nF1_score: train {train_f1_score} / valid {valid_f1_score}"
            )
            metric.reset()
            # Earlystopping Callback
            if avg_vloss < best_loss:
                best_loss = avg_vloss
                patience = 0
                best_model = deepcopy(model.state_dict())
            else:
                patience += 1

        log.info("Training end, saving model")
        torch.save(
            best_model,
            config.TRAINED_MODELS_DIRECTORY + config.BEST_MODEL_FILE_NAME,
        )
        mlflow.log_params(
            {"lr": optimizer.defaults["lr"], "num_epochs_trained": epoch + 1}
        )
        mlflow.log_metrics({"best_valid_loss": best_loss.item()})
        mlflow.pytorch.log_state_dict(best_model, config.MLFLOW_ARTIFACT_PATH)
        mlflow.end_run()
