import sys
from typing import List
from typing import Optional
from unittest.mock import patch

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI


def load_model(
    config: str,
    ckpt: Optional[str] = None,
    device: str = "cpu",
    return_synth: bool = False,
    extra_args: Optional[List[str]] = None,
    load_data: bool = True,
):
    """
    Load a model from a checkpoint using a config file.
    """
    args = ["fit", "-c", str(config), "--trainer.accelerator", device]
    if extra_args is not None:
        args.extend(extra_args)

    datamodule = None
    if not load_data:
        datamodule = L.LightningDataModule

    with patch.object(sys, "argv", args):
        cli = LightningCLI(run=False, datamodule_class=datamodule)
        model = cli.model

    if ckpt is not None:
        state_dict = torch.load(ckpt, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)

    if return_synth:
        synth = cli.datamodule.synth
        return model, synth

    return model
