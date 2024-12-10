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
import torch.nn.functional as F


class SingleLayerGRU(torch.nn.Module):

    def __init__(self, hidden_states: int, in_features: int, out_features: int) -> None:
        super().__init__()

        self.hidden_states = hidden_states
        self.in_features = in_features
        self.out_features = out_features

        _kernel_fz = torch.zeros(in_features + hidden_states, 2*hidden_states)
        _bias_fz = torch.zeros(2*hidden_states)
        _kernel_r = torch.zeros(in_features + hidden_states, hidden_states)
        _bias_r = torch.zeros(hidden_states)

        torch.nn.init.xavier_normal_(_kernel_fz, gain=0.5)
        torch.nn.init.xavier_normal_(_kernel_r, gain=0.5)
        
        self.kernel_fz = torch.nn.Parameter(_kernel_fz)
        self.bias_fz = torch.nn.Parameter(_bias_fz)
        self.kernel_r = torch.nn.Parameter(_kernel_r)
        self.bias_r = torch.nn.Parameter(_bias_r)
        self.out = torch.nn.Linear(hidden_states, out_features)

        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['N'] = hidden_states
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
        
    def step(self,
             u: torch.Tensor,
             x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # u: (B, in_features)
        # x0: (B, hidden_states)
        u_ = torch.cat([u, x0], dim=-1)   # (B, in_features + hidden_states)
        fz = u_ @ self.kernel_fz + self.bias_fz  # (B, 2*hidden_states)
        fz = F.sigmoid(fz)
        f, z = torch.split(fz, self.hidden_states, dim=-1)

        ur_ = torch.cat([u, f * x0], dim=-1) # (B, in_features + hidden_states)
        r = ur_ @ self.kernel_r + self.bias_r
        r = F.tanh(r)

        xk = (1 - z) * x0 + z * r
        y = self.out(x0)

        return y, xk
    

    def simulate(self,
                u: torch.Tensor,
                x0: list[torch.Tensor] = None) -> torch.Tensor:
        
        if x0 is None:
            x0 = torch.rand(u.shape[0],  1, self.hidden_states)

        y = torch.empty(u.shape[0], u.shape[1], self.out_features, device=u.device)
        xk_ = x0.squeeze(1)
        
        for t, u_ in enumerate(u.transpose(0, 1)):
            y_, xk_ = self.step(u_, xk_)
            y[:, t, :] = y_
            
        return y
    

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor] = None) -> torch.Tensor:
        
        return self.simulate(u, x0)
    

class SingleLayerLSTM(torch.nn.Module):

    def __init__(self, hidden_states: int, in_features: int, out_features: int) -> None:
        super().__init__()

        self.hidden_states = hidden_states
        self.in_features = in_features
        self.out_features = out_features

        _kernel_fiz = torch.zeros(in_features + hidden_states, 3*hidden_states)
        _bias_fiz = torch.zeros(3*hidden_states)
        _kernel_r = torch.zeros(in_features + hidden_states, hidden_states)
        _bias_r = torch.zeros(hidden_states)

        torch.nn.init.xavier_normal_(_kernel_fiz, gain=0.5)
        torch.nn.init.zeros_(_bias_fiz)
        torch.nn.init.xavier_normal_(_kernel_r, gain=0.5)
        torch.nn.init.zeros_(_bias_r)
        
        self.kernel_fiz = torch.nn.Parameter(_kernel_fiz)
        self.bias_fiz = torch.nn.Parameter(_bias_fiz)
        self.kernel_r = torch.nn.Parameter(_kernel_r)
        self.bias_r = torch.nn.Parameter(_bias_r)
        self.out = torch.nn.Linear(hidden_states, out_features)

        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['N'] = hidden_states
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
        
    def step(self,
             u: torch.Tensor,
             h0: torch.Tensor,
             c0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # u: (B, in_features)
        # x0: (B, hidden_states)
        u_ = torch.cat([u, h0], dim=-1)   # (B, in_features + hidden_states)
        fiz = u_ @ self.kernel_fiz + self.bias_fiz  # (B, 3*hidden_states)
        fiz = F.sigmoid(fiz)
        f, i, z = torch.split(fiz, self.hidden_states, dim=-1)

        r = u_ @ self.kernel_r + self.bias_r  # (B, hidden_states)
        r = F.tanh(r)

        ck = f * c0 + i * r
        hk = z * F.tanh(ck)
        y = self.out(hk)

        return y, hk, ck
    

    def simulate(self,
                u: torch.Tensor,
                x0: torch.Tensor = None) -> torch.Tensor:
        
        if x0 is None:
            h0 = torch.rand(u.shape[0],  1, self.hidden_states)
            c0 = torch.rand(u.shape[0],  1, self.hidden_states)
        else:
            h0, c0 = x0.split(self.hidden_states, dim=-1)

        y = torch.empty(u.shape[0], u.shape[1], self.out_features, device=u.device)
        hk_ = h0.squeeze(1)
        ck_ = c0.squeeze(1)
        
        for t, u_ in enumerate(u.transpose(0, 1)):
            y_, hk_, ck_ = self.step(u_, hk_, ck_)
            y[:, t, :] = y_

        return y
    

    def forward(self, 
                u: torch.Tensor, 
                x0: list[torch.Tensor] = None) -> torch.Tensor:
                
        return self.simulate(u, x0)
    

class FFNNObserver(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, context_window: int) -> None:
        super().__init__()

        self.in_features = in_features * context_window
        self.out_features = out_features

        self.ffnn = torch.nn.Sequential(torch.nn.Linear(self.in_features, self.in_features // 2),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(self.in_features // 2, self.out_features),
                                        torch.nn.Sigmoid())
        
        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['fixed_context_window'] = context_window
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
    
    def forward(self, 
                u: torch.Tensor, 
                _: list[torch.Tensor] = None) -> torch.Tensor:
        
        return self.ffnn(u.flatten(1))
    

class CNNObserver(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden_channels: int, scan_window: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.scan_window = scan_window

        # Dimension of the inner convolution

        self.cnn = torch.nn.Sequential(torch.nn.Conv1d(in_channels=self.in_features, 
                                                       out_channels=self.hidden_channels, 
                                                       kernel_size=self.scan_window, 
                                                       padding=self.scan_window // 2),
                                        torch.nn.GELU(),
                                        torch.nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.out_features, kernel_size=scan_window*2, stride=scan_window//4))
        
        self.params_str = {}
        self.params_str['model'] = self.__class__.__name__ 
        self.params_str['hidden_channels'] = hidden_channels
        self.params_str['scan_window'] = scan_window
        self.params_str['in_features'] = in_features
        self.params_str['out_features'] = out_features
    
        
    def forward(self, 
            u: torch.Tensor, 
            _: list[torch.Tensor] = None) -> torch.Tensor:
            
        y = self.cnn(u.transpose(1, 2)).transpose(1, 2)
        y = y[:, -1, :].unsqueeze(1)
        y = F.tanh(y)
        return y
