
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn

from ssm.layer import LRU
from ssm.StackedSSM import StackedSSMModel
from ssm.util import match_metric_or_loss
from train_helper import ModelParams, TrainingParams, run_training_helper

lru_kwargs = {"rmin":0.1, "rmax": 0.95, "max_phase":6.283,
              "use_parallel_scan":True}

model_kwargs = ModelParams(base_model=LRU,
                           base_model_kwargs=lru_kwargs,
                           N=[6, 6, 6, 6],
                           H=3,
                           activation_fnc=nn.ELU(),
                           skip_connection=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_opts = {'factor': 0.5, 'patience': 50, 'verbose': True, 'threshold': 1e-6}


tr_loss = match_metric_or_loss('wmse', gamma=0.9975)
val_metric = match_metric_or_loss('mse')
test_metric = {'mse': match_metric_or_loss('mse'), 'fit': match_metric_or_loss('fit')}

training_params = TrainingParams(batch_size=30,
                                 lr=1e-3,
                                 early_stopping_patience=100,
                                 training_loss=tr_loss,
                                 validation_metric=val_metric,
                                 test_metrics=test_metric,
                                 scheduler=scheduler,
                                 scheduler_opts=scheduler_opts)

model = run_training_helper(model_kwargs, training_params, accelerator='cpu', show_progress=True)