from typing import Tuple

import pytorch_lightning as pl
import torch

from s5.metrics import DecayingMSELoss


class SubspaceEncodedRNN(pl.LightningModule):

    train_loss: torch.nn.Module
    val_metric: torch.nn.Module
    test_metric: torch.nn.ModuleDict
    optimizer: torch.optim.Optimizer
    optimizer_opts: dict

    def __init__(self, 
                 obsv: torch.nn.Module | pl.LightningModule, 
                 model: torch.nn.Module | pl.LightningModule) -> None:
        super().__init__()

        self.obsv = obsv
        self.model = model

        # Default options
        self.training_loss = DecayingMSELoss()
        self.val_metric = DecayingMSELoss()
        self.test_metric = DecayingMSELoss()
        self.optimizer = torch.optim.Adam
        self.optimizer_opts = None

        # self.save_hyperparameters()

    def training_settings(self, 
                          train_loss: torch.nn.Module = None, 
                          val_metric: torch.nn.Module = None, 
                          test_metric: dict[str, torch.nn.Module] = None,
                          optimizer: torch.optim.Optimizer = None,
                          optimizer_opts: dict = None) -> None:
        if train_loss is not None:
            self.train_loss = train_loss
        if val_metric is not None:
            self.val_metric = val_metric
        if test_metric is not None:
            self.test_metric = torch.nn.ModuleDict(test_metric) if isinstance(test_metric, dict) else {'test_metric': test_metric}
        if optimizer is not None:
            self.optimizer = optimizer
        if optimizer_opts is not None:
            self.optimizer_opts = optimizer_opts

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor]) -> torch.Tensor:

        x0_ = self.obsv(x0) if self.obsv is not None else None
        
        if x0_.ndim == 2:
            x0_ = x0_.unsqueeze(1)
        elif x0_.ndim == 3 and x0_.shape[1] > 1:
            x0_ = x0_[:, -1, :].unsqueeze(1)
            
        y = self.model(u, x0_)
        return y


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> torch.Tensor:
        """Perform a training step"""
        U, Y, X0 = batch
        U = U.to(device=self.device)
        Y = Y.to(device=self.device)
        X0 = X0.to(device=self.device)

        y_hat = self(U, X0)
        loss = self.train_loss(y_hat, Y)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> torch.Tensor:
        """Perform a validation step"""
        U, Y, X0 = batch
        U = U.to(device=self.device)
        Y = Y.to(device=self.device)
        X0 = X0.to(device=self.device)
        y_hat = self(U, X0)
        metric = self.val_metric(y_hat, Y)
        self.log('validation_metric', metric)
        return metric
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], bath_idx: int) -> dict:
        """Perform a test step"""
        U, Y, X0 = batch
        U = U.to(device=self.device)
        Y = Y.to(device=self.device)
        X0 = X0.to(device=self.device)
        y_hat = self(U, X0)
        metrics_results = {}

        Y = Y.cpu()
        y_hat  = y_hat.cpu()
 
        for name, metric in self.test_metric.items():
            metrics_results[name] = metric(y_hat, Y)

        return {'metrics': metrics_results, 'y': Y, 'y_hat': y_hat}
    
    def configure_optimizers(self):
        """Configure the optimizer"""
        opts = self.optimizer_opts if self.optimizer_opts is not None else { 'lr': 1e-3 }
        optimizer = torch.optim.Adam(self.parameters(), **opts) if self.optimizer is None else self.optimizer(self.parameters(), **opts)
        return optimizer


