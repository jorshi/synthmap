"""
PyTorch Lightning CLI
"""
import logging
import os
import time

from lightning.pytorch.cli import LightningCLI

from synthmap.callback import SaveConfigCallbackWanb

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
