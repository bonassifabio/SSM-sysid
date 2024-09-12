'''
Copyright (C) 2024 Fabio Bonassi, Carl Andersson, and co-authors

This file is part of ssm.

ssm is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ssm is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with gddpc.  If not, see <http://www.gnu.org/licenses/>.
'''

from typing import Callable, Tuple

import pytorch_lightning as pl
import torch

from Datasets.base import Statistics
from ssm.metrics import DecayingMSELoss


class LightningDynModel(pl.LightningModule):

    def __init__(self, model, stats_u: Statistics, stats_y: Statistics):
        super().__init__()
        self.model = model
        self.stats_u = stats_u
        self.stats_y = stats_y

        self.training_loss = DecayingMSELoss()
        self.val_metric = DecayingMSELoss()
        self.test_metric = DecayingMSELoss()
        self.optimizer = torch.optim.Adam
        self.optimizer_opts = None
        self.scheduler = None
        self.scheduler_opts = None

    def training_settings(self,
                          train_loss: torch.nn.Module = None,
                          val_metric: torch.nn.Module = None,
                          test_metric: dict[str, torch.nn.Module] = None,
                          optimizer: torch.optim.Optimizer = None,
                          optimizer_opts: dict = None,
                          scheduler: Callable = None,
                          scheduler_opts: dict = None) -> None:
        if train_loss is not None:
            self.train_loss = train_loss
        if val_metric is not None:
            self.val_metric = val_metric
        if test_metric is not None:
            self.test_metric = torch.nn.ModuleDict(test_metric) if isinstance(test_metric, dict) else {
                'test_metric': test_metric}
        if optimizer is not None:
            self.optimizer = optimizer
        if optimizer_opts is not None:
            self.optimizer_opts = optimizer_opts
        if scheduler is not None:
            self.scheduler = scheduler
        if scheduler_opts is not None:
            self.scheduler_opts = scheduler_opts

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> torch.Tensor:
        """Perform a training step"""
        U, Y = batch
        U = U.to(device=self.device)
        Y = Y.to(device=self.device)

        U = self.stats_u.normalize(U)

        y_hat = self.model(U)
        y_hat = self.stats_y.denormalize(y_hat)
        loss = self.train_loss(y_hat, Y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> torch.Tensor:
        """Perform a validation step"""
        U, Y = batch

        U = U.to(device=self.device)
        Y = Y.to(device=self.device)
        U = self.stats_u.normalize(U)

        y_hat = self.model(U)
        y_hat = self.stats_y.denormalize(y_hat)
        metric = self.val_metric(y_hat, Y)
        self.log('validation_metric', metric)
        return metric

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> dict:
        """Perform a test step"""
        U, Y = batch
        U = U.to(device=self.device)
        Y = Y.to(device=self.device)
        U = self.stats_u.normalize(U)
        y_hat = self.model(U)
        y_hat = self.stats_y.denormalize(y_hat)
        metrics_results = {}

        Y = Y.cpu()
        y_hat = y_hat.cpu()

        for name, metric in self.test_metric.items():
            metrics_results[name] = metric(y_hat, Y)

        return {'metrics': metrics_results, 'y': Y, 'y_hat': y_hat}

    def configure_optimizers(self):
        """Configure the optimizer"""
        opts = self.optimizer_opts if self.optimizer_opts is not None else {'lr': 1e-3}
        optimizer = torch.optim.Adam(self.parameters(), **opts) if self.optimizer is None else self.optimizer(
            self.parameters(), **opts)

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_opts)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "validation_metric"}]
        else:
            return optimizer

