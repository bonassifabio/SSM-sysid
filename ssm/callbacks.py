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

import re

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from matplotlib.patches import Circle
from pytorch_lightning.utilities import grad_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ssm.LightningDynModel import LightningDynModel


class PolesZerosGainCallback(pl.Callback):

    def __init__(self, log_every_n_epochs: int = 1, dt: float = None, pz_scale: str = 'linear', show_nyquist: str = 'abs') -> None:
        """Callback to log the poles, zeros and gains of the S5 model.

        Args:
            log_every_n_epochs (int, optional): Logging frequency (epochs). Defaults to 1.
            dt (float, optional): Sampling time for plotting the Nyquist/Shannon frequency. Defaults to None.
            pz_scale (str, optional): Scale for the poles-zeros plots ['linear', 'log', 'symlog']. Defaults to 'linear'.
            show_nyquist (str, optional): Whether to show the Nyquist frequency ['abs', 'real', 'none']. Defaults to 'avs'.
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.pz_scale = pz_scale
        self.dt = dt
        self.nyquist = show_nyquist

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LightningDynModel) -> None:
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            poles, zeros, gains = pl_module.poles_zeros_gains()

            for i, (p, z, k) in enumerate(zip(poles, zeros, gains)):
                p_re, p_im = p.real, p.imag
                z_re, z_im = z.real, z.imag
                k_min, k_max = k

                # Generate a figure representing poles as 'o' and zeros as 'x' and add it to the tensorboard log
                fig, ax = plt.subplots()
                ax.scatter(p_re, p_im, marker='o', color='r')
                ax.scatter(z_re, z_im, marker="X", color='b')
                
                # Set log scale for both axes
                ax.set_xscale(self.pz_scale)
                ax.set_yscale(self.pz_scale)
                
                if self.dt is not None:
                    w_ny = 3.14 / self.dt
                    # nycircle = Circle((0.0, 0.0), radius=w_ny, fill=True, alpha=0.5, color='g', label='Nyquist')
                    # ax.add_patch(nycircle)

                    if self.nyquist == 'abs':
                        nycircle = Circle((0.0, 0.0), radius=w_ny, fill=True, alpha=0.5, color='orange')
                        ax.add_patch(nycircle)
                    elif self.nyquist == 'real':
                        ax.axvspan(-w_ny, 0, alpha=0.5, color='orange')

                ax.set_xlabel('Re')
                ax.set_ylabel('Im')
                ax.set_title(f'Layer {i}')
                ax.autoscale()
                ax.grid(True, which='both')
                ax.axhline(y=0, color='k')
                ax.axvline(x=0, color='k')

                trainer.logger.experiment.add_figure(f'Poles-Zeros/Layer {i}', fig, global_step=trainer.global_step)
                trainer.logger.experiment.add_scalars(f'Poles-Zeros-mag/Layer {i}', 
                                                      {'Re': -p_re.max().item(), 'Im': p_im.max().item()}, 
                                                      global_step=trainer.global_step)
                trainer.logger.experiment.add_scalars(f'Static-Gains/Layer {i}', {'min': k_min, 'max': k_max}, global_step=trainer.global_step)

        return super().on_validation_end(trainer, pl_module)
    

class PlotValidationTrajectories(pl.Callback):

    def __init__(self, test_dataloader: DataLoader, hidden_states: bool = True, every_n_epochs: int = 10) -> None:
        """Callback to plot the test trajectories.

        Args:
            test_dataloader (DataLoader): The dataloader for the test set.
            hidden_states (bool, optional): Whether to plot the hidden states. Defaults to True.
            every_n_epochs (int, optional): Loggin frequency (in epochs). Defaults to 10.
        """
        super().__init__()

        self.test_dataloader = test_dataloader
        self.hidden_states = hidden_states
        self.every_n_epochs = every_n_epochs

    def test_trajectories(self, trainer: pl.Trainer, pl_module: pl.LightningModule, final: bool = False) -> dict[str, float]:
        results = []
        
        # Do the actual testing - we are in eval mode!
        for batch_idx, batch in enumerate(self.test_dataloader):
            results.append(pl_module.test_step(batch, batch_idx))

        # Plot the results
        for i, r in enumerate(results):
            y, y_hat = r['y'], r['y_hat']
            metrics = r['metrics']

            # Make sure that y and y_hat are 1-dimensional, and that the batch only contains 1 sequence
            if not (y.shape[-1] == 1 and y_hat.shape[-1] == 1 and y.shape[0] == 1 and y_hat.shape[0] == 1):
                raise NotImplementedError('This Callback is only implemented for single sequence-batches with 1-dimensional output')

            metric_str = '- '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            test_title = f'Test {i} - {metric_str} {"- FINAL" if final else f"- epoch {trainer.current_epoch}"}'
            fig, ax = plt.subplots()
            ax.plot(y.squeeze().detach().numpy(), color='r', label='GT')
            ax.plot(y_hat.squeeze().detach().numpy(), color='g', label='Model')
            ax.plot((y_hat-y).squeeze().detach().numpy(), color='b', label='Residual')
            ax.set_xlabel('Time')
            ax.set_title(test_title)
            ax.grid(True, which='both')
            ax.legend()

            if self.hidden_states:
                h = r['hidden']
                for l, hl in enumerate(h):
                    figl, axl = plt.subplots()

                    for i, hli in enumerate(hl):
                        axl.plot(hli.squeeze().detach().numpy())

                    axl.set_xlabel('Time')
                    axl.set_title(f'Hidden States - Layer {l}')
                    if final:
                        trainer.logger.experiment.add_figure(f'final_testing/hidden{l}', figl, global_step=trainer.global_step)
                    else:
                        trainer.logger.experiment.add_figure(f'testing/hidden{l}', figl, global_step=trainer.global_step)

            if final:
                trainer.logger.experiment.add_figure('final_testing/outputs', fig)
            else:
                trainer.logger.experiment.add_figure(f'testing/outputs{i}', fig, global_step=trainer.global_step)

        for k, v in metrics.items():
            trainer.logger.experiment.add_scalar(f'testing/{k}', v, global_step=trainer.global_step)
            
        # Convert metrics to float
        for k in metrics.keys():
            metrics[k] = float(metrics[k])
            
        return metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.test_trajectories(trainer, pl_module)
    

class GradientLogger(pl.Callback):

    def __init__(self, norm_type: int = 2) -> None:
        """Callback to log the gradient norms.

        Args:
            norm_type (int, optional): The norm to log. Defaults to 2.
        """
        super().__init__()
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer) -> None:
        grad_log = grad_norm(pl_module, norm_type=self.norm_type)
        grad_total = None
        new_log = {}

        replace_rgx = re.compile(fr"grad_{float(self.norm_type)}_norm[/_]*")

        for k in grad_log.keys():
            new_key = replace_rgx.sub('grad_', k)

            if new_key == 'grad_total':
                grad_total = grad_log[k]
            else:
                new_log[new_key] = grad_log[k]
        
        trainer.logger.experiment.add_scalar(f'grad_total', grad_total, global_step=trainer.global_step)
        trainer.logger.experiment.add_scalars(f'grad_norm_weights', new_log, global_step=trainer.global_step)
        return super().on_before_optimizer_step(trainer, pl_module, optimizer)