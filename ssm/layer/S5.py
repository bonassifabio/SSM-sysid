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

import numpy as np
import torch

from ssm.layer.initializations import (HippoDiagonalizedInitializer,
                                       S5Initializer)
from ssm.layer.LinearSSM import BaseLinearSSM
from ssm.layer.util import parallel_scan


class S5(BaseLinearSSM):
    """A Simplified Structured State Space (S5) cell"""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 state_size: int,
                 use_D=False,
                 use_parallel_scan=False,
                 return_next_y: bool = True,
                 initialization: S5Initializer = None,
                 dt_range = (1, 0.1),
                 discretization_method = 'zoh') -> None:
        """
        Single S5 Cell

        Args:

            in_features (int): The number of input features
            out_features (int): The number of output features
            state_size (int): The number of internal states
            use_parallel_scan (bool, optional): Whether to use parallel scan. Defaults to False.
            return_next_y (bool, optional): Whether to return the next output (y(k+1)). Defaults to True.
            initialization (S5Initializer, optional): The initialization method. Defaults to None.
            dt_range (tuple[float, float], optional): The range of the time scale of the S5 processes. Defaults to (1, 0.1).


        Raises:
            NotImplementedError: If the initialization method is not implemented

        Returns:
            torch.Tensor: The output sequence (shape: (B, T, P))
        """

        super().__init__(in_features, out_features, state_size, use_D=use_D, return_next_y=return_next_y)


        self.in_features = in_features
        self.out_features = out_features
        self.state_size = state_size

        self.use_parallel_scan = use_parallel_scan

        # Define the S5 parameters: Lambda, V, B, C, D
        if initialization is None:
            initialization = HippoDiagonalizedInitializer()

        if not isinstance(initialization, S5Initializer):
            raise ValueError(
                f'The initialization method should be an instance of S5Initializer, but got {type(initialization)}')

        diag, V = initialization(N=2*state_size, in_features=in_features, out_features=out_features)

        self.create_param("Lambda_re", np.log(-diag.Λ.real).astype(np.complex64))
        self.create_param("Lambda_im", np.log(diag.Λ.imag).astype(np.complex64))
        self.create_param("B_p", diag.Bd.astype(np.complex64))

        dt_range = np.log(dt_range)

        dt = torch.rand(state_size) * (dt_range[0] - dt_range[1]) + dt_range[1]

        self.create_param("log_timescale", dt)

        self.discretization_method = discretization_method

    @property
    def dt(self):
        return torch.exp(self.log_timescale)

    @property
    def Lambda_c(self):
        return -torch.exp(self.Lambda_re) + 1j * torch.exp(self.Lambda_im)

    @property
    def Lambda(self) -> torch.Tensor:
        """Get the state matrix Lambda"""
        Lambda, _ = self.discretize(self.discretization_method)
        return Lambda

    @property
    def A(self) -> torch.Tensor:
        return torch.diag(self.Lambda)
    @property
    def B(self) -> torch.Tensor:
        """Get the input matrix B"""
        _, B = self.discretize(self.discretization_method)

        return B

    @property
    def F(self):
        return torch.diag(self.Lambda_c)

    @property
    def G(self):
        return self.B_p

    def discretize(self, method: str = 'zoh') -> \
            (tuple)[torch.Tensor, torch.Tensor]:
        """Discretize the S5Cell with the given discretization method

        Args:
            method (str, optional): The discretization method. Defaults to 'zoh'.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The discretized system matrices (Λd, Bd)
        """
        dt = self.dt
        Lambda_c = self.Lambda_c
        B_c = self.B_p

        if method == 'zoh':
            Lambda = torch.exp(Lambda_c * dt)
            IL = torch.reshape((Lambda - 1) / Lambda_c, (-1, 1))
            Bd = IL * B_c
        else:
            raise NotImplementedError(f'Method "{method}" not implemented')

        return Lambda, Bd

    def step_state(self, x_t: torch.Tensor, u_t: torch.Tensor,
                   Lambda: torch.Tensor = None, B: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Apply a single simulation step of the S5 Cell

        Args:
            x_t (torch.Tensor): The current state (shape: (..., N))
            u_t (torch.Tensor): The current input (shape: (..., M))
            Lambda (torch.Tensor): The discretized state matrix Lambda (shape: (N,))
            B (torch.Tensor): The discretized input matrix B (shape: (N, M))

        Returns:
            x_t+1 (torch.Tensor): The next state (shape: (..., N))
        """
        if Lambda is None:
            Lambda = self.Lambda
        if B is None:
            B = self.B

        # Compute the next state
        x_tp1 = x_t * Lambda + torch.einsum("ij, ...j->...i", B, u_t)

        return x_tp1

    def simulate_states(self, u, x0=None):
        """
        Args:
            u: input tensor of shape (..., time, in_features)

        Returns:
            y: output tensor of shape (..., time, state_size)
        """
        Lambda, B = self.discretize(self.discretization_method)

        if self.use_parallel_scan:
            x = parallel_scan(u, Lambda, B, x0=x0)
        else:
            x = self.scan(u, x0, Lambda=Lambda, B=B)
        return x


if __name__ == "__main__":
    s5 = S5(1, 1, 4, use_D=False)
