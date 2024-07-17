import torch
import numpy as np
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