from typing import List

import auraloss
import torch
import torchaudio
from einops import repeat

from synthmap.utils.audio_utils import load_audio
from synthmap.utils.audio_utils import load_wav_dir_as_tensor


class FitnessFunctionBase(torch.nn.Module):
    """
    Base class for fitness functions
    """

    def __init__(self):
        super().__init__()

    @property
    def objective(self) -> str:
        """
        The objective of the fitness function - either "min" or "max"
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        Calculate the fitness value for a batch of parameters

        Args:
            x: A batch of parameters

        Returns:
            A batch of fitness values
        """
        raise NotImplementedError


class MelSpecFitness(FitnessFunctionBase):
    """
    Fitness function for Mel spectrogram FAD
    """

    def __init__(
        self,
        audio: str,
        sample_rate: int,
        duration: int,
        n_fft: int = 2048,
        hop_length: int = 64,
        n_mels: int = 128,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.duration = duration
        self.audio_dir = audio
        self.prepare()

    @property
    def objective(self) -> str:
        return ["min"]

    def prepare(self):
        """
        Prepare the audio for fitness calculation
        """
        self.audio = load_wav_dir_as_tensor(
            self.audio_dir, self.duration, self.sample_rate
        )
        print(f"Loaded audio with shape {self.audio.shape}")

        mel = self.mel(self.audio)
        self.register_buffer("mel_targets", mel)
        print(f"Calculated Mel spectrogram with shape {mel.shape}")

    def forward(self, x: torch.Tensor):
        # Calculate Mel spectrogram for the input audio
        x = self.mel(x)

        minimums = torch.zeros(x.shape[0], device=x.device)
        maximums = torch.zeros(x.shape[0], device=x.device)
        argmins = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        for i in range(x.shape[0]):
            diff = torch.mean(torch.abs(x[i] - self.mel_targets), dim=(-1, -2))
            minimums[i] = torch.min(diff)
            maximums[i] = torch.max(diff)
            argmins[i] = torch.argmin(diff)

        argmin = torch.unique(argmins, return_counts=False).shape[0]
        argmin = torch.tensor([argmin], device=x.device, dtype=torch.float32)
        argmin = argmin / x.shape[0]
        argmin = repeat(argmin, "() -> b", b=x.shape[0])

        return (minimums,)


class MultiScaleSpectralFitness(FitnessFunctionBase):
    """
    Fitness function to minimize the difference in multi-scale spectral
    features between the target and the synthesized audio
    """

    def __init__(
        self,
        audio: str,  # Path to an input audio sample for target
        sample_rate: int,  # Input audio sample rate
        duration: str,  # Duration of the input audio sample
        fft_sizes: List[int] = [1024, 2048, 512, 128],
        hop_sizes: List[int] = None,
        win_lengths: List[int] = None,
        sum_loss: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_path = audio
        self.sum_loss = sum_loss

        if hop_sizes is None:
            hop_sizes = [fft_size // 4 for fft_size in fft_sizes]

        if win_lengths is None:
            win_lengths = [fft_size for fft_size in fft_sizes]

        self.losses = []
        for i in range(len(fft_sizes)):
            stft = auraloss.freq.STFTLoss(
                sample_rate=sample_rate,
                fft_size=fft_sizes[i],
                hop_size=hop_sizes[i],
                win_length=win_lengths[i],
                reduction="none",
                **kwargs,
            )
            self.losses.append(stft)

        self.prepare()

    @property
    def objective(self) -> List[str]:
        if self.sum_loss:
            return ["min"]
        return ["min"] * len(self.losses)

    def prepare(self):
        """
        Prepare the audio for fitness calculation
        """
        audio = load_audio(self.audio_path, self.sample_rate, length=self.duration)
        self.register_buffer("audio", audio)
        print(f"Loaded audio with shape {self.audio.shape}")

    def forward(self, x: torch.Tensor):
        # Calculate the multi-scale spectral loss
        target = repeat(self.audio, "1 n -> b 1 n", b=x.shape[0])

        evals = []
        for i in range(len(self.losses)):
            loss = self.losses[i](x[:, None, :], target)
            loss = torch.mean(loss, dim=(-1, -2))
            evals.append(loss)

        if self.sum_loss:
            evals = torch.vstack(evals)
            evals = torch.sum(evals, dim=0)
            return (evals,)

        return evals
