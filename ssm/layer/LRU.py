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

from ssm.layer.LinearSSM import BaseLinearSSM
from ssm.layer.util import parallel_scan


class LRU(BaseLinearSSM):
    def __init__(self, in_features, out_features, state_size,
                 rmin: float = 0.0, rmax: float = 1.0, max_phase=6.283,
                 use_parallel_scan=True, use_D=False, return_next_y=False):
        """
            Linear recurrent unit layer (LRU) as described in https://arxiv.org/abs/2303.06349
        args:
            in_features: number of input features
            out_features: number of output features
            state_size: number of parallel LRU processes
            rmin: minimum value of the radius of the eigenvalues of the state transition matrix
            rmax: maximum value of the radius of the eigenvalues of the state transition matrix
            max_phase: maximum phase of the eigenvalues of the state transition matrix
            parallel_scan: whether to use parallel scan or not
            return_next_y: whether to return y_t+1 instead of y_t
        """

        super().__init__(in_features, out_features, state_size, use_D=use_D, return_next_y=return_next_y)

        u1 = torch.rand(state_size)
        u2 = torch.rand(state_size)
        self.create_param('nu_log', torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        self.create_param("theta_log", torch.log(max_phase*u2))


        B_re = torch.randn([state_size, in_features])/np.sqrt(in_features)
        B_im = torch.randn([state_size, in_features])/np.sqrt(in_features)
        self.create_param("B_p", torch.complex(B_re, B_im))

        self.use_parallel_scan = use_parallel_scan


    @property
    def Lambda(self):
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)
        return Lambda
    @property
    def B(self):
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        gammas = torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))[..., None]
        return self.B_p * gammas

    @property
    def A(self):
        return torch.diag(self.Lambda)

    @property
    def F(self):
        raise NotImplementedError("LRU does not have a continuous representation")

    def simulate_states(self, u, x0=None):
        """
        Args:
            u: input tensor of shape (..., time, in_features)

        Returns:
            y: output tensor of shape (..., time, state_size)
        """

        Lambda = self.Lambda
        B_ = self.B

        if self.use_parallel_scan:
            x = parallel_scan(u, Lambda, B_, x0=x0)
        else:
            x = self.scan(u, x0, Lambda=Lambda, B=B_)

        return x


    def step_state(self, x_t, u_t, Lambda = None, B = None, **kwargs):
        """
        Step function for the LRU

        x_t+1 = Lambda*x_t + B@u_t

        Lambda and B might be precomputed to avoid unnecessary computations
        Args:
            x_t: (..., state_size) Complex valued initial states
            u_t: (..., in_features) Complex valued input
            Lambda: (state_size) Diagonal of Complex valued state transition matrix
            B: (state_size, in_features) Complex valued input matrix
        Output:
            x_t+1
        """

        if Lambda is None:
            Lambda = self.Lambda
        if B is None:
            B = self.B
        x = Lambda*x_t + torch.einsum("ji, ...i->...j", B, u_t)
        return x






if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lru = LRU(in_features=2, out_features=2, state_size=2, rmin=0.1, rmax=0.9)
    u = torch.zeros(100, 2)
    u[50:, 0] = 1
    u[:50, 1] = 1
    y = lru(u)
    lru.use_parallel_scan = False
    y2 = lru(u)
    plt.subplot(2, 1, 1)
    plt.plot(y.detach().numpy())
    plt.plot(y2.detach().numpy())
    plt.show()