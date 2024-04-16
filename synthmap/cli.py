"""
PyTorch Lightning CLI
"""
import logging
import os
import time

from lightning.pytorch.cli import LightningCLI

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    """ """
    _ = LightningCLI()
    return


def main():
    """ """
    start_time = time.time()
    run_cli()
    end_time = time.time()
    log.info(f"Total time: {end_time - start_time} seconds")
