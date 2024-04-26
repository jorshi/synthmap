"""
Evaluation Script

1) Evaluate the best Genetic Algorithm result wrt to target
2) Evaluate the VAE model wrt to the target (audio)
3) Evaluate the VAE model timbre manipulation spearman correlation
"""
import argparse
import json
import sys
from collections import namedtuple
from pathlib import Path
from typing import List
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch
import torchaudio
from einops import rearrange
from einops import repeat
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.cli import LightningCLI
from scipy.stats import spearmanr

from synthmap.metrics import LogSpectralDistance
from synthmap.utils.audio_utils import load_audio


# Named tuple to hold info regarind a model run
ModelRun = namedtuple("ModelRun", ["cli", "synth", "params", "target"])


class TimbreMatchCLI(LightningCLI):
    """
    PyTorch Lightning CLI for timbre matching
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--target", type=str, help="Path to target audio file")


class ModelEvaluation:
    """
    Struct for model evaluation
    """

    def __init__(self, name: str = None):
        self.name = name
        self.target_file = None
        self.ga_lsd = LogSpectralDistance()
        self.vae_target_lsd = LogSpectralDistance()
        self.timbre_correlation = {}
        self.audio = {}
        self.timbre_plot = {}


def load_run(run: Path) -> ModelRun:
    """
    Load the model from a run
    """
    # Find the checkpoint file
    checkpoint = list(run.rglob("*.ckpt"))
    assert len(checkpoint) == 1, "More than one checkpoint found"
    checkpoint = checkpoint[0]

    # Find the config file
    config = run.joinpath("config.yaml")
    assert config.exists(), "Config file not found"

    # Load the model
    with patch.object(
        sys,
        "argv",
        [
            "",
            "-c",
            str(config),
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    ):
        cli = TimbreMatchCLI(run=False)

    # Load model from checkpoint
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    cli.model.load_state_dict(state_dict)
    cli.model.eval()

    # Find the GA parameter file
    param_file = run.joinpath("params/pareto_front_params.pt")
    assert param_file.exists(), "GA parameter file not found"
    params = torch.load(param_file, map_location="cpu")

    # Pull out the synth model
    synth = cli.datamodule.synth

    run = ModelRun(cli=cli, synth=synth, params=params, target=None)
    return run


def load_target_audio(run: ModelRun):
    """
    Load the target audio
    """
    target_file = Path(run.cli.config["target"])
    assert target_file.exists(), f"Target audio file not found: {target_file}"

    # Load the target audio
    audio = load_audio(target_file, sample_rate=48000, length=48000)
    return ModelRun(cli=run.cli, params=run.params, synth=run.synth, target=audio)


def evaluate_genetic_algorithm(run: ModelRun, results: ModelEvaluation):
    """
    Evaluate the best Genetic Algorithm result wrt to target
    """
    # Load the GA parameters
    params = run.params[:1]
    y_hat = run.synth(params)
    results.ga_lsd.update(run.target[None], y_hat[None])
    results.audio["target_ga"] = y_hat.clone()
    return results


def evaluate_vae_target_reconstruction(run: ModelRun, results: ModelEvaluation):
    """
    Evaluate the VAE model for reconstruction
    """
    params = run.params[:1]

    with torch.no_grad():
        # Run the VAE model
        p_hat, _, _ = run.cli.model(params=params)
        y_hat = run.synth(torch.clamp(p_hat, 0.0, 1.0))

    results.vae_target_lsd.update(run.target[None], y_hat[None])
    results.audio["target_vae"] = y_hat.clone()
    return results


def evaluate_vae_timbre_control(run: ModelRun, results: ModelEvaluation):
    """
    Evaluate the VAE model for timbre manipulation
    """

    # Params for the target
    params = run.params[:1].clone()

    # Get the number and names of regularized dimensions
    controls = []
    extractor = run.cli.model.audio_regularizer.extractor
    y = run.synth(params)
    for feat in extractor.features:
        x = feat.get_as_dict(y)
        controls.extend(x.keys())

    # Get the latent variable
    with torch.no_grad():
        z = run.cli.model.param_encoder(params)
        z, _ = run.cli.model.bottleneck(z)

        for i, control in enumerate(controls):
            n = torch.linspace(-1.0, 1.0, 10)
            z_mod = repeat(z, "1 l -> n l", n=n.shape[0]).clone()
            z_mod[:, i] = z_mod[:, i] + n

            p_hat = run.cli.model.param_decoder(z_mod)
            y_hat = run.synth(torch.clamp(p_hat, 0.0, 1.0))

            features = extractor(y_hat)
            rank = spearmanr(features[:, i].numpy(), n)[0]
            results.timbre_correlation[control] = rank

            # Store the audio
            y_hat = rearrange(y_hat, "b n -> 1 (b n)")
            results.audio[f"{control.lower()}_control"] = y_hat.clone()

    return results


def generate_timbre_plot_data(run: ModelRun, results: ModelEvaluation):
    """
    Generate data for plotting timbre results from the GA outpout and similar
    number of random VAE outputs
    """
    # Generate output for the entire GA output
    params = run.params

    with torch.no_grad():
        y_ga = run.synth(params)

        # Generate VAE output
        z_ga = run.cli.model.param_encoder(params)
        z_ga, _ = run.cli.model.bottleneck(z_ga)
        z_sample = torch.randn_like(z_ga)
        p_hat = run.cli.model.param_decoder(z_sample)
        y_vae = run.synth(torch.clamp(p_hat, 0.0, 1.0))

        extractor = run.cli.model.audio_regularizer.extractor
        results.timbre_plot["ga"] = extractor(y_ga).cpu().numpy()
        results.timbre_plot["vae"] = extractor(y_vae).cpu().numpy()
        results.timbre_plot["target"] = extractor(run.target).cpu().numpy()
        results.timbre_plot["target (ga)"] = extractor(y_ga[:1]).cpu().numpy()

    return results


def evaluate_run(run_path: Path):
    """
    Evaluate a single run
    """
    # Load run
    run = load_run(run_path)

    # Load the target audio
    run = load_target_audio(run)

    # Evaluate the genetic algorithm
    results = ModelEvaluation(name=run_path.name)
    results.target_file = run.cli.config["target"]
    results = evaluate_genetic_algorithm(run, results)

    # Evaluate the VAE model for target reconstruction
    results = evaluate_vae_target_reconstruction(run, results)

    # Evaluate the VAE model for timbre manipulation
    results = evaluate_vae_timbre_control(run, results)

    # Save the target audio
    results.audio["target"] = run.target.clone()

    # Generate data for plotting
    results = generate_timbre_plot_data(run, results)

    return results


def find_runs(log_dir: Path) -> List[Path]:
    """
    Search for all the lightning logs in a directory
    """
    runs = sorted(list(log_dir.rglob("version_*")))
    runs = [run for run in runs if run.is_dir()]
    return runs


def save_timbre_plots(results: List[ModelEvaluation], outdir: Path):
    """
    Save all the timbre plots
    """
    outdir = outdir.joinpath("figures")
    outdir.mkdir(parents=True, exist_ok=True)
    features = list(results[0].timbre_correlation.keys())

    for result in results:
        keys = list(result.timbre_plot.keys())
        for i in range(0, len(features), 2):
            fig, ax = plt.subplots()
            for key in keys:
                x = result.timbre_plot[key][:, i]
                y = result.timbre_plot[key][:, i + 1]
                marker = "*" if key.startswith("target") else "o"
                size = 250 if key.startswith("target") else 50
                alpha = 1.0 if key.startswith("target") else 0.4
                ax.scatter(x, y, label=key.upper(), marker=marker, s=size, alpha=alpha)

            ax.set_xlabel(features[i])
            ax.set_ylabel(features[i + 1])
            ax.legend()
            fig.savefig(
                outdir.joinpath(f"{result.name}_{features[i]}_{features[i+1]}.png")
            )


def save_result_audio(results: List[ModelEvaluation], outdir: Path):
    """
    Save all the audio results
    """
    outdir = outdir.joinpath("audio")
    outdir.mkdir(parents=True, exist_ok=True)

    for result in results:
        for key, audio in result.audio.items():
            audio_file = outdir.joinpath(f"{result.name}_{key}.wav")
            torchaudio.save(str(audio_file), audio, sample_rate=48000)


def save_result_json(results: List[ModelEvaluation], outdir: Path):
    """
    Save all the audio results
    """
    final = {}
    for result in results:
        final[result.name] = {
            "name": result.name,
            "target_file": result.target_file,
            "ga_lsd": result.ga_lsd.compute().item(),
            "vae_target_lsd": result.vae_target_lsd.compute().item(),
            "timbre_correlation": result.timbre_correlation,
        }

    outdir = outdir.joinpath("results.json")
    with open(outdir, "w") as f:
        json.dump(final, f, indent=4)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_dir", help="Directoy of log files", type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)
    parser.add_argument("--save_audio", help="Save audio files", action="store_true")

    args = parser.parse_args(arguments)

    # Find all the runs in the directory
    runs = find_runs(Path(args.log_dir))
    results = [evaluate_run(run) for run in runs]

    # Save the results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_audio:
        save_result_audio(results, output_dir)

    save_result_json(results, output_dir)
    save_timbre_plots(results, output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
