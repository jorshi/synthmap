"""
SynthMap Learn a VAE model on synth presets
"""
from typing import Optional
from typing import Tuple

import lightning as L
import torch

from synthmap.model import AutoEncoder
from synthmap.params import DiscretizedNumericalParameters


class SynthMapTask(L.LightningModule):
    """
    Lightning task for estimating the tonic of a given mridangam sound

    Args:
        model: a model to produce a fixed embedding
    """

    def __init__(
        self,
        autoencoder: AutoEncoder,
        param_discretizer: Optional[DiscretizedNumericalParameters] = None,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        audio_regularizer: Optional[torch.nn.Module] = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.loss_fn = loss_fn
        self.param_discretizer = param_discretizer
        self.audio_regularizer = audio_regularizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    def _do_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        if len(batch) == 1:
            preset = batch[0]
        else:
            audio, preset = batch

        # Create a discretized representation of the parameters
        if self.param_discretizer is not None:
            y = self.param_discretizer.discretize(preset)

        y_hat, z, kl = self(preset)

        # Group the parameters back into (batch, class, parameter) if discretized
        if self.param_discretizer is not None:
            y_hat = self.param_discretizer.group_parameters(y_hat)

        # Preset reconstruction loss
        reconstruction = self.loss_fn(y_hat, y)

        loss = {
            "reconstruction": reconstruction,
            "kl": kl,
        }

        # Regularize the latent space to vary according to the audio metric
        if self.audio_regularizer is not None:
            assert len(batch) == 2, "Audio must be provided for audio regularization"
            audio_loss = self.audio_regularizer(audio, z)
            loss["audio_reg"] = audio_loss

        summed_loss = self.summed_losses(loss)
        loss["loss"] = summed_loss

        # Log the loss and return the summed loss
        self.log_loss(loss, stage)
        return summed_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "test")

    def log_loss(self, loss: dict, prefix: str):
        for key, value in loss.items():
            if value is not None:
                prog_bar = key == "loss"
                self.log(f"{prefix}/{key}", value, on_epoch=True, prog_bar=prog_bar)

    def summed_losses(self, losses: dict):
        return sum([loss for loss in losses.values() if loss is not None])
