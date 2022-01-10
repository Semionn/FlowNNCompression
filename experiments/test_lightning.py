from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from pytorch_lightning import LightningModule, Trainer
from torchvision.datasets import MNIST

from argparse import ArgumentParser, Namespace


class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss


def main(args: Namespace):
    lit_model = LitModel()
    dataset = MNIST(os.path.abspath(os.path.join(os.curdir, os.pardir)), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)
    # trainer = Trainer()
    # trainer.fit(lit_model, train_loader)
    trainer = Trainer(gpus=args.gpus, strategy=args.strategy)
    trainer.fit(lit_model, train_loader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--strategy", default="ddp", type=str, choices=["ddp", "dp", "ddp_spawn"])
    parser.add_argument("--gpus", default=1, type=int)
    args = parser.parse_args()
    main(args)
