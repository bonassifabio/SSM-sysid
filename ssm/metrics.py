import torch
import torch.nn.functional as F


class DecayingMSELoss(torch.nn.Module):
    def __init__(self, washout: int = 0, gamma: float = 1.0):
        """MSE Loss with decaying weights

        Args:
            washout (int, optional): Washout period. Defaults to 0.
            gamma (float, optional): Exponential weighting factor. Defaults to 1.0.
        """
        super().__init__()
        self.washout = washout
        self.gamma = gamma


    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Compute the decaying MSE loss.

        Args:
            y (torch.Tensor): The ground truth.
            y_hat (torch.Tensor): The prediction.

        Returns:
            torch.Tensor: The decaying MSE loss.
        """

        # Avoid instability - if gamma ≈ 1, just use conventional MSE
        if 1.0 - 1e-8 <= self.gamma <= 1.0 - 1e-6: 
            return F.mse_loss(y[:, self.washout:], y_hat[:, self.washout:])
        
        decay = torch.tensor([self.gamma**t for t in range(y.shape[1] - self.washout)], 
                             device=y.device, dtype=y.dtype, requires_grad=False)
        decay = decay.unsqueeze(0).unsqueeze(2)

        squared_error = (y[:, self.washout:, :] - y_hat[:, self.washout:, :])**2
        weighted_squared_error = decay * squared_error
        return torch.mean(weighted_squared_error)


class FITIndex(torch.nn.Module):
    def __init__(self, washout: int = 0):
        super().__init__()
        self.washout = washout

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        error_rmse = torch.sqrt(torch.mean((y[:, self.washout:, :] - y_hat[:, self.washout:, :])**2, dim=2))
        y_avg = torch.mean(y[:, self.washout:, :], dim=1).unsqueeze(1)
        avg_std = torch.sqrt(torch.mean((y[:, self.washout:, :] - y_avg)**2, dim=2)).mean(dim=1).view(y.shape[0], 1)

        res = torch.mean(error_rmse / avg_std)
        return 100.0 * (1.0 - res)