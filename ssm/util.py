from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from s5.metrics import DecayingMSELoss, FITIndex

def sanitize_activation(activation) -> torch.nn.Module:
    """
        Assert activation is a torch.nn.Module and return it.

    Args:
        activation: Activation funciton as string or torch.nn.Module

    Returns:
        Activation function as callable
    """
    if isinstance(activation, str):
        return match_activation(activation)
    elif isinstance(activation, torch.nn.Module):
        return activation
    else:
        raise ValueError(f'Activation "{activation}" is not a string or torch.nn.Module')


def match_activation(activation: str) -> torch.nn.Module:
    """Retrieve the activation function given its string.

    Args:
        activation (str): The activation function. Can be 'relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid', 'glu', 'swish' or 'none'.

    Raises:
        NotImplementedError: If the activation function is not implemented.

    Returns:
        torch.nn.Module: The activation function.
    """



    match activation:
        case 'relu':
            return torch.nn.ReLU()
        case 'leaky_relu':
            return torch.nn.LeakyReLU()
        case 'gelu':
            return torch.nn.GELU()
        case 'tanh':
            return torch.nn.Tanh()
        case 'sigmoid':
            return torch.nn.Sigmoid()
        case 'glu':
            return torch.nn.GLU()
        case 'swish':
            return torch.nn.SiLU()
        case 'none':
            return torch.nn.Identity()
        case _:
            raise NotImplementedError(f'Activation "{activation}" not implemented')


def match_metric_or_loss(loss_name: str, **kwargs) -> torch.nn.Module:
    if loss_name.lower() == 'mse' or loss_name.lower() == 'wmse':
        washout = kwargs.get('washout', 0)
        return DecayingMSELoss(**kwargs)
    elif loss_name.lower() == 'fit':
        washout = kwargs.get('washout', 0)
        return FITIndex(washout)
    else:
        raise ValueError(f'Unknown loss function {loss_name}')


class TransposableBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 3:
            return super().forward(input)
        else: 
            x = input.transpose(1, 2)
            return super().forward(x).transpose(1, 2)
