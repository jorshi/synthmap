"""
SynthMap Learn a VAE model on synth presets
"""
from typing import Tuple

import lightning as L
import torch

from synthmap.model import AutoEncoder


class SynthMapTask(L.LightningModule):
    """
    Lightning task for estimating the tonic of a given mridangam sound

    Args:
        model: a model to produce a fixed embedding
    """

    def __init__(
        self,
        autoencoder: AutoEncoder,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.loss_fn = torch.nn.MSELoss()

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
            _, preset = batch

        y, z, kl = self(preset)

        # Preset reconstruction loss
        reconstruction = self.loss_fn(y, preset)

        loss = {
            "reconstruction": reconstruction,
            "kl": kl,
        }

        # Log the loss and return the summed loss
        self.log_loss(loss, stage)
        return self.summed_losses(loss)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._do_step(batch, "test")

    def log_loss(self, loss: dict, prefix: str):
        for key, value in loss.items():
            if value is not None:
                self.log(f"{prefix}/{key}", value, on_epoch=True, prog_bar=True)

    def summed_losses(self, losses: dict):
        return sum([loss for loss in losses.values() if loss is not None])
