"""
PyTorch Lightning CLI
"""
import logging
import os
import time

from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.cli import LightningCLI

from synthmap.callback import SaveConfigCallbackWanb
from synthmap.data.fitness import MultiScaleSpectralFitness
from synthmap.data.fitness import NoveltySearch
from synthmap.data.genetic import GeneticSynthDataLoader

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    """ """
    _ = LightningCLI(save_config_callback=SaveConfigCallbackWanb)
    return


def main():
    """ """
    start_time = time.time()
    run_cli()
    end_time = time.time()
    log.info(f"Total time: {end_time - start_time} seconds")


class TimbreMatchCLI(LightningCLI):
    """
    PyTorch Lightning CLI for timbre matching
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--target", type=str, help="Path to target audio file")


def vae_match_target():
    """
    CLI entry point for the timbre VAE to match a target
    """
    cli = TimbreMatchCLI(run=False, save_config_callback=SaveConfigCallbackWanb)

    # Check if the target audio file is provided
    target = cli.config["target"]
    assert target is not None, "Target audio file must be provided"

    # Device
    accelerator = cli.config["trainer.accelerator"]
    device = "cuda" if accelerator == "gpu" else "cpu"

    # Create Fitness functions
    mel_fitness = MultiScaleSpectralFitness(
        target, 48000, 48000, fft_sizes=[2048], scale="mel", n_bins=128
    )

    stft_fitness = MultiScaleSpectralFitness(
        target,
        48000,
        48000,
        fft_sizes=[1024, 512, 256, 64],
        w_sc=0.0,
        w_log_mag=1.0,
        w_lin_mag=1.0,
        sum_loss=True,
    )

    extractor = cli.model.audio_regularizer.extractor
    novelty = NoveltySearch(extractor=extractor)

    # Create GeneticSynthDataLoader
    synth = cli.datamodule.synth
    dataloader = GeneticSynthDataLoader(
        synth,
        10,
        128,
        fitness_fns=[mel_fitness, stft_fitness, novelty],
        verbose=False,
        reset_on_epoch=False,
        device=device,
        return_evals=True,
    )

    start_time = time.time()
    cli.trainer.fit(cli.model, train_dataloaders=dataloader)
    end_time = time.time()
    log.info(f"Total time: {end_time - start_time} seconds")
