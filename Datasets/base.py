import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch

class DataLoaderExt(DataLoader):
    @property
    def nu(self):
        return self.dataset.nu

    @property
    def ny(self):
        return self.dataset.ny


class Statistics(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

class IODataset(Dataset):
    """Create dataset from data.

    Parameters
    ----------
    u, y: ndarray, shape (total_len, n_channels) or (total_len,)
        Input and output signals. It should be either a 1d array or a 2d array.
    seq_len: int (optional)
        Maximum length for a batch on, respectively. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If None,
        put the entire dataset on a single batch.

    """
    def __init__(self, u, y, seq_len=None, overlap=0.0):
        if u.ndim == 1:
            # Create batch dimension
            u = u[None, ...]
            y = y[None, ...]
        if u.ndim == 2:
            # Create feature dimension
            u = u[..., None]
            y = y[..., None]
        if seq_len is not None:
            self.u = IODataset._batchify(u.astype(np.float32), seq_len, overlap)
            self.y = IODataset._batchify(y.astype(np.float32), seq_len, overlap)
        else:
            self.u = u.astype(np.float32)
            self.y = y.astype(np.float32)
        self.ntotbatch = self.u.shape[0]
        self.seq_len = self.u.shape[1]
        self.nu = 1 if u.ndim == 1 else u.shape[1]
        self.ny = 1 if y.ndim == 1 else y.shape[1]

    def __len__(self):
        return self.ntotbatch

    def __getitem__(self, idx):
        return self.u[idx, ...], self.y[idx, ...]

    @staticmethod
    def _batchify(x, seq_len, overlap):
        # data should be a torch tensor
        # data should have size (total number of samples) times (number of signals)
        # The output has size (number of batches) times (number of signals) times (batch size)
        # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
        n_batches = x.shape[0]
        batch_seq_len = x.shape[1]
        step_size = int(np.floor((1 - overlap) * seq_len))
        n_seq_per_batch = int((batch_seq_len - seq_len)/ step_size) + 1
        out = np.zeros((n_batches*n_seq_per_batch, seq_len, x.shape[-1]), dtype=np.float32)
        for i in range(n_seq_per_batch):
            out[i*n_batches:(i+1)*n_batches, :, :] = x[:, i*step_size:i*step_size+seq_len, :]
        return out


