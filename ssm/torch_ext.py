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