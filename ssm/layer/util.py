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

import numpy as np
import torch
import torch.nn.functional as F


def parallel_scan(u, Lambda, B, x0=None):
    """
        Applies the parallel scan algorithm
    Args:
        u: (..., time, in_features) Input tensor of shape
        Lambda: (state_size) Complex valued diagonal state transition matrix
        B: (state_size, in_features) Complex valued input matrix
        x0: (..., state_size) Complex valued initial states
    Returns:
        x: (..., time+1, state_size) Complex valued state sequence
    """

    u = F.pad(u, (0, 0, 1, 0))
    u = u.to(dtype=torch.complex64)
    x = torch.einsum("ji,...i->...j", B, u)
    if x0 is not None:
        x[..., 0, :] = x0

    seq_length = u.shape[-2]
    log2_length = int(np.ceil(np.log2(seq_length)))

    Lambda_levels = [Lambda]


    for d in range(log2_length):
        width = 2 ** d
        step = 2 * width
        offset1 = width - 1
        offset2 = step - 1

        x_l = x[..., offset1:-width:step, :].clone()
        x_r = x[..., offset2::step, :].clone()

        x_new = x_l * Lambda + x_r

        x[..., offset2::step, :] = x_new

        Lambda = Lambda*Lambda
        Lambda_levels.append(Lambda)

    Lambda_levels.pop()
    for d in range(log2_length - 1, -1, -1):
        width = 2 ** d
        step = 2 * width
        offset1 = 2 * width - 1
        offset2 = step + width - 1
        Lambda = Lambda_levels.pop()

        x_l = x[..., offset1:-width:step, :].clone()
        x_r = x[..., offset2::step, :].clone()

        x_new = x_l * Lambda + x_r

        x[..., offset2::step, :] = x_new

    return x