
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
along with ssm.  If not, see <http://www.gnu.org/licenses/>.
'''

from typing import Callable, NamedTuple, Optional

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from Datasets.silverbox import create_silverbox_datasets
from ssm.callbacks import PlotValidationTrajectories
from ssm.LightningDynModel import LightningDynModel
from ssm.StackedSSM import StackedSSMModel

TrainingParams = NamedTuple('TrainingParams', [('batch_size', int),
                                               ('lr', float),
                                               ('early_stopping_patience', int),
                                               ('training_loss', nn.Module),
                                               ('validation_metric', nn.Module),
                                               ('test_metrics', dict[str, nn.Module] | nn.Module),
                                               ('scheduler', Optional[Callable]),
                                               ('scheduler_opts', Optional[dict])])

ModelParams = NamedTuple('ModelParams', [('base_model', type),
                                         ('base_model_kwargs', dict),
                                         ('N', list[int]),
                                         ('H', int),
                                         ('activation_fnc', Optional[nn.Module]),
                                         ('skip_connection', bool)])

def run_training_helper(model_kwargs: ModelParams,
                        training_params: TrainingParams,
                        name: str = None,
                        accelerator: str = 'cpu', show_progress=True) -> LightningDynModel:

    train_dataset, valid_dataset, test_dataset, stats_u, stats_y = create_silverbox_datasets()

    train_loader = DataLoader(train_dataset, batch_size=training_params.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=training_params.batch_size*5, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = StackedSSMModel(1, 1, state_sizes=model_kwargs.N, hidden_units=model_kwargs.H,
                            activation_fnc=model_kwargs.activation_fnc,
                            base_model=model_kwargs.base_model, base_model_kwargs=model_kwargs.base_model_kwargs)

    lightningModel = LightningDynModel(model=model, stats_u=stats_u, stats_y=stats_y)

    lightningModel.training_settings(train_loss=training_params.training_loss,
                                     val_metric=training_params.validation_metric,
                                     test_metric=training_params.test_metrics,
                                     optimizer_opts={'lr': training_params.lr},
                                     scheduler=training_params.scheduler,
                                     scheduler_opts=training_params.scheduler_opts
                                     )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best-model', monitor='validation_metric', save_top_k=1,
                                                       mode='min', save_last=True)
    timer = pl.callbacks.Timer(interval='epoch')
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='validation_metric',
                                                         patience=training_params.early_stopping_patience, mode='min')
    test = PlotValidationTrajectories(test_dataloader=test_loader, hidden_states=False, every_n_epochs=20)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [early_stopping_callback, timer, checkpoint_callback, test, lr_monitor]

    # # Trainer
    trainer = pl.Trainer(max_epochs=5000,
                         default_root_dir="training_logs",
                         callbacks=callbacks,
                         accelerator=accelerator,
                         enable_progress_bar=show_progress,
                         log_every_n_steps=1)

    trainer.fit(lightningModel, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return lightningModel

