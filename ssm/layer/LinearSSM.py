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
import torch.nn as nn
from control import StateSpace
from scipy.linalg import svdvals


class BaseLinearSSM(nn.Module):
    def __init__(self, in_features: int, out_features: int, state_size: int, use_D=True, return_next_y: bool = True):
        super().__init__()
        """
            A base class for linear discrete state space models composed by A, B, C and D. A, B and C are complex-valued 
            while D is a real-valued.
            A and B are left undefined while C and D are defined as linear layers as to common DL practice
            
            if return_next_y is True:
                x_t+1 = A*x_t + B*u_t
                y_t = real(C*x_t) + D*u_t
            else:
                x_t = A*x_t-1 + B*u_t
                y_t = real(C*x_t) + D*u_t
                
            
        Args:
            in_features: number of input features
            out_features: number of output features
            state_size: size of the state_space
            use_D: whether to use D or not
            return_next_y: whether to return y_t+1 instead of y_t                
        """

        self.state_size = state_size
        self.in_features = in_features
        self.out_features = out_features
        self.return_next_y = return_next_y

        C_re = torch.randn([out_features, state_size]) / np.sqrt(2*state_size)
        C_im = torch.randn([out_features, state_size]) / np.sqrt(2*state_size)
        self.create_param('C_p', torch.complex(C_re, C_im))

        # A parameter for output bias is usually not used in SI but an important factor for the activation function.
        self.create_param("output_bias", torch.zeros(out_features))

        if use_D:
            # Don't use bias in D as it is already included in the output_bias
            self.D = nn.Linear(in_features, out_features, bias=False)
        else:
            self.D = None


    @property
    def A(self):
        raise NotImplementedError

    @property
    def B(self):
        raise NotImplementedError

    @property
    def F(self):
        raise NotImplementedError

    @property
    def G(self):
        raise NotImplementedError

    @property
    def C(self):
        return self.C_p

    def create_param(self, name, init_value, learnable=True):
        if not isinstance(init_value, torch.Tensor):
            init_value = torch.tensor(init_value)
        if learnable:
            self.register_parameter(name, nn.Parameter(init_value))
        else:
            self.register_buffer(name, init_value)

    def forward(self, u, x0=None):
        return self.simulate(u, x0)

    def simulate(self, u, x0=None):
        """
            Simulated output using the simulate_states function

            x_t+1 = lam_d*x_t + B*u_t
            y_t = (Real(x_t), Imag(y_t))
        Args:
            u: (..., time, in_features)
            x0: (..., state_size) Complex valued initial states
        Returns:
            y (..., time, out_features)
        """
        x = self.simulate_states(u, x0)

        if self.return_next_y:
            x = x[..., 1:, :]
        else:
            x = x[..., :-1, :]


        out = torch.real(torch.einsum("ji,...i->...j", self.C, x))
        #out = torch.real(x @ self.C)

        if self.D is not None:
            out = out + self.D(u)

        return out + self.output_bias

    def simulate_states(self, u, x0=None):
        """
            Simulate states using the scan function (default)

            Override this function for more efficient implementations

        Args:
            u: (..., time, in_features)
            x0: (..., state_size) Complex valued initial states
        Returns:
            x (..., time+1, state_size)
        """

        return self.scan(u, x0)



    def scan(self, u, x0=None, **kwargs):
        """
            Applies the scan algorithm with the step function

            for 1:T
                x_t+1 = step(x_t, u_t)
        Args:
            u: (..., time, in_features) Inpute sequence
            x0: (..., state_size) Complex valued initial states (Optional)
            kwargs: Additional arguments for the step function
        Returns:
            x (..., time+1, state_size)
        """
        u = u.to(torch.complex64)

        T = u.shape[-2]

        x = torch.zeros(u.shape[:-2] + (T + 1, self.state_size), device=u.device, dtype=torch.complex64)
        if x0 is not None:
            x[..., 0, :] = x0
        x_t = x[..., 0, :]


        for t in range(T):
            x_t = self.step_state(x_t, u[..., t, :], **kwargs)
            x[..., t + 1, :] = x_t

        return x

    def step_state(self, x_t: torch.Tensor, u_t: torch.Tensor, A=None, B=None, **kwargs):
        """
            Naive step function for a linear sequence model

            x_t+1 = A@x_t + B@u_t
        Args:
            x_t: (..., state_size) Complex valued initial states
            u_t: (..., in_features) Complex valued input
            kwargs: Potential additional arguments
        Output:
            x_t+1
        """
        if A is None:
            A = self.A
        if B is None:
            B = self.B

        x_t = torch.einsum("ji,...i->...j", A, x_t)
        x_t = x_t + torch.einsum("ji,...i->...j", B, u_t)
        return x_t

    def step_output(self, x_t: torch.Tensor, u_t: torch.Tensor, C=None, **kwargs):
        """
            Naive step function for a linear sequence model
            x_t+1 = A@x_t + B@u_t
            y_t = C@x_t + D@u_t
            or
            y_t = C@x_t+1 + D@u_t
        Args:
            x_t: (..., state_size) Complex valued initial states
            kwargs: Potential additional arguments
        Output:
            y_t
        """
        if C is None:
            C = self.C

        x_tp1 = self.step_state(x_t, u_t, **kwargs)

        if self.return_next_y:
            y = torch.real(torch.einsum("ji,...i->...j", C, x_tp1))
            #y = torch.real(x_tp1 @ C)
        else:
            y = torch.real(torch.einsum("ji,...i->...j", C, x_t))
            #y = torch.real(x_t @ C)

        if self.D is not None:
            y = y + self.D(u_t)

        return x_tp1, y + self.output_bias

    def poles_zeros_gains(self):
        """Get the poles, zeros, gains (minumum-maximum singular values), scaling and rotation of the underlying continuous-time LTI system.
        """

        F = self.F
        G = self.G
        C = self.C
        D = torch.zeros((self.out_features, self.in_features))

        F = torch.block_diag(F, F.conj())
        G = torch.cat([G, G.conj()], dim=0)
        C = torch.cat([C, C.conj()], dim=1)/2

        print(F,G,C)

        F = F.detach().numpy()
        G = G.detach().numpy()
        C = C.detach().numpy()
        D = D.detach().numpy()



        ss = StateSpace(F, G, C, D)
        g0 = ss.dcgain().reshape((self.out_features, self.in_features))
        Σ = svdvals(g0)


        return ss.poles(), ss.zeros(), (Σ.min().item(), Σ.max().item())



