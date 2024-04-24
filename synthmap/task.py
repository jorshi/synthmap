"""
SynthMap Learn a VAE model on synth presets
"""
from typing import Dict
from typing import Optional
from typing import Tuple

import lightning as L
import torch

from synthmap.params import DiscretizedNumericalParameters


class SynthMapTask(L.LightningModule):
    """
    Lightning task for estimating the tonic of a given mridangam sound

    Args:
        model: a model to produce a fixed embedding
    """

    def __init__(
        self,
        param_encoder: Optional[torch.nn.Module] = None,
        audio_encoder: Optional[torch.nn.Module] = None,
        param_decoder: Optional[torch.nn.Module] = None,
        bottleneck: Optional[torch.nn.Module] = None,
        param_discretizer: Optional[DiscretizedNumericalParameters] = None,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        audio_regularizer: Optional[torch.nn.Module] = None,
        audio_regularizer_weight: float = 1.0,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.param_encoder = param_encoder
        self.audio_encoder = audio_encoder
        self.param_decoder = param_decoder
        self.bottleneck = bottleneck
        self.param_discretizer = param_discretizer
        self.loss_fn = loss_fn
        self.audio_regularizer = audio_regularizer
        self.audio_regularizer_weight = audio_regularizer_weight
        self.lr = lr

    def forward(
        self,
        params: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        Can receive either parameters or audio, but not both.
        """
        if bool(params is not None) == bool(audio is not None):
            raise ValueError("One of params or audio must be provided")

        # Encode the parameters or audio
        if params is not None:
            z = self.param_encoder(params)
        else:
            z = self.audio_encoder(audio[:, None, :])

        # Bottleneck
        reg = None
        if self.bottleneck is not None:
            z, reg = self.bottleneck(z)

        # Decode the parameters
        param_hat = self.param_decoder(z)

        # Return the raw parameters, the latent space, and the regularization term
        return param_hat, z, reg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    def _do_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """
        Perform a forward pass and compute the loss
        """
        params = batch["params"]
        audio = batch.get("audio", None)

        # Do a parameter forward pass if we have a parameter encoder
        loss = {}
        if self.param_encoder is not None:
            y_hat, z, kl = self(params=params)

            y = params
            if self.param_discretizer is not None:
                y = self.param_discretizer.discretize(y)
                y_hat = self.param_discretizer.group_parameters(y_hat)

            reconstruction = self.loss_fn(y_hat, y)
            loss["param_recon"] = reconstruction
            loss["param_kl"] = kl

        # Do an audio forward pass if we have an audio encoder
        if self.audio_encoder is not None:
            y_hat, z, kl = self(audio=audio)

            y = params
            if self.param_discretizer is not None:
                y = self.param_discretizer.discretize(y)
                y_hat = self.param_discretizer.group_parameters(y_hat)

            reconstruction = self.loss_fn(y_hat, y)
            loss["audio_recon"] = reconstruction
            loss["audio_kl"] = kl

        # Regularize the latent space to vary according to the audio metric
        if self.audio_regularizer is not None:
            assert audio is not None, "Audio must be provided for audio regularization"
            audio_loss = self.audio_regularizer(audio, z)
            loss["audio_reg"] = audio_loss * self.audio_regularizer_weight

        summed_loss = self.summed_losses(loss)
        loss["loss"] = summed_loss

        # Check if there are GA dataloader evals to log
        if "evals" in batch:
            evals = batch["evals"]
            eval_log = {f"GA_Obj{i}": evals[i] for i in range(evals.size(0))}
            self.log_loss(eval_log, stage)

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
                self.log(
                    f"{prefix}/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=prog_bar,
                )

    def summed_losses(self, losses: dict):
        return sum([loss for loss in losses.values() if loss is not None])
