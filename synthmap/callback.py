"""
PyTorch Lightning Callbacks
"""
from pathlib import Path

import lightning as L
import torch
import torchaudio
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger


class SaveAudioCallback(L.Callback):
    def __init__(self, num_samples: int = 16):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        data = trainer.train_dataloader
        self.render_audio(module, data)

    def render_audio(self, module, data):
        if not hasattr(data, "synth"):
            print("Data loader does not have a synth attribute")
            return

        outdir = Path(module.logger.log_dir).joinpath("audio")
        outdir.mkdir(exist_ok=True)

        # Get a batch of data
        batch = next(iter(data))

        if len(batch) == 2:
            audio, params = batch
        else:
            (params,) = batch

        params = params[: self.num_samples]

        # Generate audio
        with torch.no_grad():
            y = data.synth(params)

        # Predicted audio
        p_hat, _, _ = module(params=params.to(module.device))
        if module.param_discretizer is not None:
            p_hat = module.param_discretizer.group_parameters(p_hat)
            p_hat = module.param_discretizer.inverse(p_hat)

        p_hat = torch.clamp(p_hat, 0.0, 1.0)
        y_hat = data.synth(p_hat.to("cpu"))

        # Interleave the audio
        audio = []
        for i in range(y_hat.shape[0]):
            audio.extend([y[i], y_hat[i]])

        audio = torch.hstack(audio)[None]

        # Save the audio
        torchaudio.save(
            outdir.joinpath(f"audio_{module.current_epoch}.wav"),
            audio,
            sample_rate=data.synth.sample_rate,
        )


class SaveParetoFrontParameters(L.Callback):
    """
    Custom callback to save the parameters of the Pareto front
    """

    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """
        Save the parameters and evals of the final population sorted by the Pareto front
        """
        data = trainer.train_dataloader
        size = data.ga.population.evals.shape[0]
        params = data.ga.population.take_best(size)

        outdir = Path(module.logger.log_dir).joinpath("params")
        outdir.mkdir(exist_ok=True)

        torch.save(params.values, outdir.joinpath("pareto_front_params.pt"))
        torch.save(params.evals, outdir.joinpath("pareto_front_evals.pt"))


class SaveConfigCallbackWanb(SaveConfigCallback):
    """
    Custom callback to move the config file saved by LightningCLI to the
    experiment directory created by WandbLogger. This has a few benefits:
    1. The config file is saved in the same directory as the other files created
         by wandb, so it's easier to find.
    2. The config file is uploaded to wandb and can be viewed in the UI.
    3. Subsequent runs won't be blocked by the config file already existing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        super().setup(trainer, pl_module, stage)
        if trainer.global_rank == 0:
            if isinstance(trainer.logger, WandbLogger):
                config = Path(trainer.log_dir).joinpath("config.yaml")
                print(f"config is:{config}")
                assert config.exists()
                experiment_dir = Path(trainer.logger.experiment.dir)
                print(f"experiment_dir is:{config}")
                # If this is the first time using wandb logging on this machine,
                # the experiment directory won't exist yet.
                if not experiment_dir.exists():
                    experiment_dir.mkdir(parents=True)

                config.rename(experiment_dir.joinpath("model-config.yaml"))
