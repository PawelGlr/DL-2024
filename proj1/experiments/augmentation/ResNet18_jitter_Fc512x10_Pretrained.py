import torchvision.models as models
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import v2 as transforms
import numpy as np

import sys
import json

sys.path.append(".")
from proj1.src.data import get_cv_data_loaders

if __name__ == '__main__':

    with open('proj1/experiments/augmentation/lr_opt.txt') as f: 
        config = json.loads(f.read())

    hyperparameters = {
        "lr": config['lr'],
        "pretrained": True,
        "model": "resnet18",
        "fc": "512x10",
        "regularization": "None",
        "frozen_layers": "None",
        "early_stopping": "None",
        "augmentations": "jitter",
        "other": "None"
    }

    run_name = "resnet18/test_aug_jitter"


    class ResNetClassifier(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(pretrained=True)
            self.resnet.fc = torch.nn.Linear(512, 10)

        def forward(self, x):
            return self.resnet(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            self.log("val_loss", loss)
            accuracy = (logits.argmax(1) == y).float().mean()
            self.log("val_accuracy", accuracy, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            self.log("hp/test_loss", loss)
            accuracy = (logits.argmax(1) == y).float().mean()
            self.log("hp/test_accuracy", accuracy)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=hyperparameters["lr"])

        def on_train_start(self) -> None:
            self.logger.log_hyperparams(
                hyperparameters,
                metrics={
                    "hp/test_loss": 0,
                    "hp/test_accuracy": 0,
                    "agg/mean_accuracy": 0,
                    "agg/std_accuracy": 0,
                },
            )
            return super().on_train_start()


    results = []
    for train_dataloader, valid_dataloader, test_dataloader in get_cv_data_loaders(train_transforms=[transforms.ColorJitter(brightness=.5, hue=.3)]):
        model = ResNetClassifier()
        logger = TensorBoardLogger(
            "lightning_logs",
            name=f"{run_name}/runs",
            default_hp_metric=False,
        )
        trainer = pl.Trainer(max_epochs=20, 
                             logger=logger, 
                             enable_checkpointing=False, 
                             callbacks=EarlyStopping(monitor="val_accuracy", mode="max", patience=2))
        trainer.fit(model, train_dataloader, valid_dataloader)
        test_scores = trainer.test(model, test_dataloader)
        results.append(test_scores[0])


    logger = TensorBoardLogger(
        "lightning_logs",
        name=f"{run_name}/agg",
        default_hp_metric=False,
    )
    test_accuracies = [score["hp/test_accuracy"] for score in results]
    aggregated_scores = {}
    aggregated_scores["agg/mean_accuracy"] = np.mean(test_accuracies)
    aggregated_scores["agg/std_accuracy"] = np.std(test_accuracies)
    logger.log_hyperparams(params=hyperparameters, metrics=aggregated_scores)
