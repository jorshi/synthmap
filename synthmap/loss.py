"""
Loss functions and regularizers for training the model.
"""
from typing import Dict
from typing import List
from typing import Optional

import torch
import torchaudio

from synthmap.feature import FeatureCollection


def get_transform(name: str, **kwargs) -> torch.nn.Module:
    """
    Get a transform from the name
    """
    if name == "mel":
        return torchaudio.transforms.MelSpectrogram(**kwargs)
    if name == "mfcc":
        return torchaudio.transforms.MFCC(**kwargs)
    if name == "stft":
        return torchaudio.transforms.Spectrogram(**kwargs)
    raise ValueError(f"Unknown transform: {name}")


class AudioLatentRegularizer(torch.nn.Module):
    """
    Regularizes entire latent space to vary according to an audio metric
    """

    def __init__(
        self,
        transform: str,
        eps: float = 1.0,
        gamma: float = 1.0,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        self.tranform = get_transform(
            transform,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.eps = eps
        self.gamma = gamma

    def log_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the log distance between two tensors
        """
        x = torch.log(x + self.eps)
        y = torch.log(y + self.eps)
        return torch.mean((torch.abs(x - y)))

    def audio_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute distance matrix over a batch of audio tensors according to transform
        """
        x = self.tranform(x)
        x = torch.log(x + self.eps)
        x = torch.flatten(x, start_dim=1)
        dist = torch.cdist(x, x, p=1) / x.shape[-1]
        return dist

    def forward(self, audio: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Regularize the latent space to vary according to the audio metric
        """
        audio_dist = self.audio_distance_matrix(audio)
        latent_dist = torch.cdist(latent, latent, p=2)

        loss = torch.tanh(self.gamma * latent_dist) - audio_dist
        loss = torch.mean(torch.abs(loss))
        return loss


class TimbreRegularizer(torch.nn.Module):
    """
    Regularize the first N dimensions of the latent space to vary according to timbre
    """

    def __init__(
        self,
        extractor: FeatureCollection,
        gamma: float = 1.0,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.extractor = extractor
        self.gamma = gamma
        if weights is None:
            weights = [1.0] * len(extractor)
        self.weights = weights

    def forward(self, audio: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Regularize the latent space to vary according to the timbre metric
        """
        features = self.extractor(audio)
        loss = []
        for i in range(features.shape[-1]):
            feat = features[..., i][..., None]
            feat_dist = torch.cdist(feat, feat, p=2)

            lat = latent[..., i][..., None]
            lat_dist = torch.cdist(lat, lat, p=2)

            dist = torch.tanh(self.gamma * lat_dist) - feat_dist
            dist = torch.mean(torch.abs(dist))
            loss.append(dist * self.weights[i])

        return torch.sum(torch.stack(loss))


class CombinedLoss(torch.nn.Module):
    """
    Combine multiple losses into a single loss
    """

    def __init__(
        self, losses: torch.nn.ModuleDict, weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.losses = torch.nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined loss
        """
        loss = 0.0
        for name, loss_fn in self.losses.items():
            weight = 1.0
            if self.weights is not None and name in self.weights:
                weight = self.weights[name]
            loss += loss_fn(x, y) * weight
        return loss
