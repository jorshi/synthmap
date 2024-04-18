import torch
import torchaudio
from einops import repeat

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
        sample_rate: int,
        audio: str,
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
        return ["min", "max"]

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

        return minimums, argmin
