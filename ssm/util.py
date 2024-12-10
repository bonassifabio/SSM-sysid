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

import torch

from ssm.metrics import DecayingMSELoss, FITIndex


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
