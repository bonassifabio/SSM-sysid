import torch
import torch.nn as nn

from s5.layer import LRU, S4, S5
from s5.util import sanitize_activation

class StackedSSMModel(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 state_sizes: list[int],
                 hidden_units: int | list[int],
                 activation_fnc: nn.Module = nn.GELU(),
                 skip_connection: bool = True,
                 base_model: type = LRU,
                 base_model_kwargs: dict = None,
                 static_prediction_model = nn.Linear,
                 static_prediction_model_kwargs: dict = None
                 ) -> None:
        """Construct a stacked state space model

        Args:
            in_features (int): Number of features of the input signal.
            out_features (int): Number of features of the output signal.
            state_sizes (list[int]): Number of states of each layer.
            hidden_units (int | list[int]): Number of hidden features of each layer. If `H` is an integer, then all layers have the same number of hidden features.
            activation_fnc (nn.Module): Activation function. Defaults to nn.GELU().
            skip_connection (bool): Whether to use skip connection. Defaults to True.
            base_model (nn.Module): Base model. Defaults to nn.LSTM.
            base_model_kwargs (dict, optional): Base model keyword arguments. Defaults to {}.
            static_prediction_model (nn.Module): Static prediction model. Defaults to nn.Linear.
            static_prediction_model_kwargs (dict, optional): Static prediction model keyword arguments. Defaults to {}.
        """
        super().__init__()

        if not isinstance(state_sizes, list):
            state_sizes = [state_sizes]
        if not isinstance(hidden_units, list):
            hidden_units = [hidden_units] * len(state_sizes)
        assert(len(state_sizes) == len(hidden_units))

        if base_model_kwargs is None:
            base_model_kwargs = {}

        if static_prediction_model_kwargs is None:
            static_prediction_model_kwargs = {}

        self.in_features = in_features
        self.out_features = out_features
        self.layers = torch.nn.ModuleList()

        self.activation_fnc = sanitize_activation(activation_fnc)
        self.skip_connection = skip_connection


        for i, state_size in enumerate(state_sizes):
            in_features_ = in_features if i == 0 else hidden_units[i - 1]
            out_features_ = hidden_units[i]

            self.layers.append(base_model(in_features=in_features_,
                                          out_features=out_features_,
                                          state_size=state_size,
                                          **base_model_kwargs))

        self.static_prediction_model = static_prediction_model(hidden_units[-1], out_features, **static_prediction_model_kwargs)

    def forward(self,
                u: torch.Tensor
                ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """ Simulate the whole stacked SSM model

        Args:
            u (torch.Tensor): Input sequence of shape (..., Time, in_features)

        Returns:
            torch.Tensor: Output sequence of shape (..., Time, out_features)
        """


        y = u
        for layer in self.layers:
            res = layer(y)
            if self.activation_fnc is not None:
                res = self.activation_fnc(res)
            if self.skip_connection:
                res = res + y
            y = res

        return self.static_prediction_model(y)

    def poles_zeros_gains(self):
        """Retrieve the poles, zeros, gains (range of singular values), eigenvalues scales and rotations of the model"""
        poles = []
        zeros = []
        gains = []


        for layer in self.layers:
            if isinstance(layer, (S4, S5)):
                p, z, k = layer.poles_zeros_gains()
                poles.append(p)
                zeros.append(z)
                gains.append(k)

        return poles, zeros, gains