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
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        if input_size == output_size:
            self.true_skip = True
            self.skip = nn.Sequential()
        else:
            self.true_skip = False
            self.skip = nn.Linear(input_size, output_size, bias=False)


    def forward(self, x):
        return self.skip(x)