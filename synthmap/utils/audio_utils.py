"""
Audio utils for the synthmap package
"""
from pathlib import Path
from typing import Optional

import torch
import torchaudio


def load_audio(
    file: str,
    sample_rate: int = 44100,
    mono: bool = True,
    min_length: Optional[int] = None,
    length: Optional[int] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Load an audio file and return it as a tensor

    Args:
        file: The path to the audio file
        sample_rate: The sample rate to resample to

    Returns:
        The audio waveform as a tensor
    """
    waveform, in_sample_rate = torchaudio.load(file)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform[:1]

    # Resample if necessary
    if in_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, in_sample_rate, sample_rate, lowpass_filter_width=512
        )

    # Ensure minimum length
    if min_length is not None and waveform.shape[-1] < min_length:
        waveform = torch.nn.functional.pad(
            waveform, (0, min_length - waveform.shape[-1])
        )

    # Ensure fixed length
    if length is not None:
        if waveform.shape[-1] < length:
            waveform = torch.nn.functional.pad(
                waveform, (0, length - waveform.shape[-1])
            )
        elif waveform.shape[-1] > length:
            waveform = waveform[:, :length]

    # Normalize
    if normalize:
        waveform = waveform / torch.abs(torch.max(waveform))

    return waveform


def load_wav_dir_as_list(
    dir: str,
    sample_rate: int = 44100,
    mono: bool = True,
    min_length: Optional[int] = None,
    length: Optional[int] = None,
    normalize: bool = False,
) -> list[torch.Tensor]:
    """
    Load all audio files in a directory and return them as a list of tensors
    """
    # Load all files
    files = list(Path(dir).rglob("*.wav"))
    waveforms = [
        load_audio(f, sample_rate, mono, min_length, length, normalize) for f in files
    ]
    return waveforms


def load_wav_dir_as_tensor(
    dir: str,
    length: int,
    sample_rate: int = 44100,
    mono: bool = True,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Load all audio files in a directory and return them as a tensor
    """
    waveforms = load_wav_dir_as_list(
        dir, sample_rate=sample_rate, mono=mono, length=length, normalize=normalize
    )
    return torch.vstack(waveforms)
