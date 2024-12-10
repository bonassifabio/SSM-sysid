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
from numpy.linalg import eigh
from torch import nn

from ssm.layer.LinearSSM import BaseLinearSSM


class S4(BaseLinearSSM):

    def __init__(self, in_features: int, out_features: int, state_size, order=4, dt_range=(1, 0.1),
                 use_D=False, return_next_y: bool = True):
        super().__init__(in_features=in_features, out_features=out_features, state_size=state_size,
                         use_D=use_D, return_next_y=return_next_y)
        """
            S4 layer as described in https://arxiv.org/abs/2111.00396 with the addition of splitting 
            the state spaces up into several parallel S4 processes of degree "order"
        Args:
            in_features: number of input features
            out_features: number of output features
            state_size: the size of the full state space i.e. number of parallel S4 processes times order
            order: order of the S4 process
            dt_range: range of the time scale of the S4 processes
            use_D: whether to use D or not
            return_next_y: whether to return y_t+1 instead of y_t  
        """

        if state_size % order != 0:
            raise ValueError("state_size must be divisible by order")

        # l_max is the maximum cover of the kernel in frequency space.
        # This need to be sufficiently large to avoid aliasing problems with slow kernels.
        # Note that this parameter does not correspond to the Nyquist frequency but is related to the kernel not being bandlimited in general
        # A heurstic value for l_max is 10/dt_min
        # Might not be optimal or sufficient for all cases
        self.l_max = int(1/dt_range[1]*10)


        self.state_size = state_size
        self.order = order
        self.n_parallel_s4 = self.state_size // self.order
        Lambda, P, B, _ = make_DPLR_HiPPO(self.order)
        dt = np.random.uniform(np.log(dt_range[0]), np.log(dt_range[1]), self.n_parallel_s4).astype(np.float32)

        # Whether to use the scan algorithm instead of the FFT based algorithm
        # For debugging purposes
        self._use_scan = False

        self.create_param('Lambda', Lambda[None, :].repeat(self.n_parallel_s4, 0))
        self.create_param('P', P[None, :].repeat(self.n_parallel_s4, 0))
        self.create_param('B_p', B[None, :, None].repeat(self.n_parallel_s4, 0).repeat(in_features, -1))
        self.create_param('log_timescale', dt)

    @property
    def dt(self):
        return torch.exp(self.log_timescale)

    @property
    def A(self):
        Ab, _ = self.discrete_DPLR()
        return Ab

    @property
    def B(self):
        _, Bb = self.discrete_DPLR()
        return Bb

    @property
    def F(self):
        I = torch.eye(self.order, dtype=torch.complex64)[None, ...]

        # Lambda (n_parallel_s4, order, order)
        Lambda = self.Lambda[..., None]

        # P/Q (n_parallel_s4, order, 1)
        P = self.P[..., :, None]
        Qc = P.conj()

        Fb = I * Lambda - P @ Qc.mT
        F = torch.block_diag(*Fb)
        return F

    @property
    def G(self):
        G = self.B_p.reshape(self.n_parallel_s4 * self.order, self.in_features)
        return G


    def discrete_DPLR(self):

        I = torch.eye(self.order, dtype=torch.complex64)[None, ...]

        #Lambda (n_parallel_s4, order, order)
        Lambda = self.Lambda[..., None]

        # P/Q (n_parallel_s4, order, 1)
        P = self.P[..., :, None]
        Qc = P.conj()

        # (n_parallel_s4, 1, 1)
        dt = self.dt[:, None, None]

        A = I*Lambda - P @ Qc.mT

        # Forward Euler
        A0 = (2.0 / dt) * I + A

        # Backward Euler
        D = I / ((2.0 / dt) - Lambda)

        # Note (1.0 / (1 + (Qc.mT @ D @ P))) is ...x1x1
        A1 = D - (D @ P @ (1.0 / (1 + (Qc.mT @ D @ P))) @ Qc.mT @ D)

        # A bar and B bar
        Ab = A1 @ A0

        Ab = torch.block_diag(*Ab)
        B = self.B_p.reshape(self.n_parallel_s4 * self.order, self.in_features)
        A1 = torch.block_diag(*A1)
        Bb = 2 * A1 @ B

        return Ab, Bb

    def simulate_states(self, u, x0=None):
        """
        Args:
            u: input tensor of shape (..., time, in_features)

        Returns:
            y: output tensor of shape (..., time, state_size)
        """
        if self._use_scan:
            A, B = self.discrete_DPLR()
            x = self.scan(u, x0, A=A, B=B)
        else:
            x = self.scan_fft(u, x0)

        return x

    def scan_fft(self, u, x0=None):
        """
            Simulated output from n_parallel_s4 number of independent S4 processes
        Args:
            u:  (..., T, input_size)
            x0: (..., n_parallel_s4, order) Complex valued initial states

        Returns:
            x: (..., T+1, state_size)
        """
        u = F.pad(u, (0, 0, 1, 0))
        u = u.to(dtype=torch.complex64)

        # Note that the current implementation is not the most efficient
        # as it scales quadratically with the order as
        # O(order^2 * n_parallel_s4 * T * log(T))
        # instead of O(in_features * out_features * T * log(T) )

        length = u.shape[-2]
        l_max = max(self.l_max, 2*length)
        kernel_fft = K_DPLR(self.Lambda, self.P, self.P, self.dt, l_max)

        ub = torch.einsum("kji,...i->...kj", self.B_p, u)
        if x0 is not None:
            ub[..., 0, :, :] = x0

        ub = torch.fft.fft(F.pad(ub, (0, 0, 0, 0, 0, l_max - length)), dim=-3)

        yd = torch.einsum("...ijk,...ik->...ij",   kernel_fft, ub)

        out = torch.fft.ifft(yd, dim=-3)[..., :length, :, :]

        out = out.reshape(*out.shape[:-2], -1)

        # l_max should be large enough so that the correction factor is not needed
        # C_corr = (torch.eye(self.state_size) - torch.matrix_power(self.A, l_max)


        return out




def K_DPLR(Lambda, P, Q, dt, L):
    # Note this is not the most efficient implementation if order is large
    # However, it is the most general implementation to include the case where
    # x0 is not None

    order = Lambda.shape[-1]

    # (L, 1, 1, 1)
    Omega_L = torch.exp((-2j * torch.pi) * (torch.arange(L) / L))[:, None, None, None]

    # eye(order), (n_parallel_s4, 1, order)
    aterm = (torch.eye(order, dtype=torch.complex64),
             (Q.real - Q.imag*1j)[..., None, :])

    # eye(order), (n_parallel_s4, order, 1)
    bterm = (torch.eye(order, dtype=torch.complex64),
             P[..., None])

    # (L, n_parallel_s4, 1, 1)
    g = (2.0 / dt[:, None, None]) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    # (L)
    c = 2.0 / (1.0 + Omega_L)

    # (1, n_parallel_s4, order, 1)
    Lambda = Lambda[None, ..., None]

    k00 = aterm[0] @ (bterm[0] * cauchy_dot(g, Lambda))
    k01 = aterm[0] @ (bterm[1] * cauchy_dot(g, Lambda))
    k10 = aterm[1] @ (bterm[0] * cauchy_dot(g, Lambda))
    k11 = aterm[1] @ (bterm[1] * cauchy_dot(g, Lambda))

    out = (c * (k00 - k01 @ (1.0 / (1.0 + k11)) @ k10)).type(torch.complex64)
    return out

def cauchy_dot(omega, lambd):
    return (1 / (omega - lambd))

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return ((Lambda_real + 1j * Lambda_imag).astype(np.complex64), P.astype(np.complex64),
            B.astype(np.complex64), V.astype(np.complex64))

def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


if __name__ == "__main__":
    s4 = S4(in_features=1, out_features=1, state_size=3, order=3, dt_range=(0.1, 0.3))
    if False:
        import matplotlib.pyplot as plt

        s4 = S4(2, out_features=2, state_size=3, order=3, dt_range=(0.1, 0.3))
        u = torch.zeros(100, 2)
        u[:50, 0] = 1
        u[50:, 0] = -1
        y = s4(u)
        s4._use_scan = True
        y2 = s4(u)
        plt.subplot(3, 1, 1)
        plt.plot(y.detach())
        plt.subplot(3, 1, 2)
        plt.plot(y2.detach())
        plt.subplot(3, 1, 3)
        plt.plot(y.detach())
        plt.plot(y2.detach())
        plt.show()