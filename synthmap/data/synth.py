import lightning as L
import torch

from synthmap.synth import AbstractSynth


class SynthesizerDataset(torch.utils.data.Dataset):
    """
    A dataset that returns pairs of synthesizer sounds and the their
    corresponding preset values.
    """

    def __init__(
        self, synth: AbstractSynth, size: int, seed: int = 649418400, return_sound=True
    ):
        super().__init__()

        self.synth = synth
        self.num_params = synth.get_num_params()
        self.size = size
        self.seed = seed
        self.return_sound = return_sound
        self.generator = torch.Generator()

    def __len__(self):
        """
        Number of samples in the dataset
        """
        return self.size

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        """
        self.generator.manual_seed(self.seed + idx)
        preset = torch.rand(1, self.num_params, generator=self.generator)

        if self.return_sound:
            sound = self.synth(preset).squeeze(0)
            return sound, preset.squeeze(0)

        return (preset.squeeze(0),)


class SynthesizerDataModule(L.LightningDataModule):
    """
    A data module for synthesizer datasets
    """

    def __init__(
        self,
        synth: AbstractSynth,
        batch_size: int,
        num_train: int = 1000000,
        num_val: int = 10000,
        num_test: int = 10000,
        seed: int = 649418400,
        return_sound: bool = True,
    ):
        super().__init__()
        self.synth = synth
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        self.return_sound = return_sound

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """

        if stage == "fit":
            self.train_dataset = SynthesizerDataset(
                self.synth, self.num_train, self.seed, self.return_sound
            )

        if stage in ["fit", "validate"]:
            seed = self.seed + self.num_train
            self.val_dataset = SynthesizerDataset(
                self.synth, self.num_val, seed, self.return_sound
            )
        elif stage == "test":
            seed = self.seed + self.num_train + self.num_val
            self.test_dataset = SynthesizerDataset(
                self.synth, self.num_test, seed, self.return_sound
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        """
        Returns the train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader
        """
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        """
        Returns the test dataloader
        """
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )


class BatchedSynthDataLoader(torch.nn.Module):
    """
    A dataloader that returns batches of synthesizer sounds, optionally running
    directly on the GPU.
    """

    def __init__(
        self,
        synth: AbstractSynth,
        size: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 649418400,
        return_sound=True,
    ):
        super().__init__()
        self.synth = synth
        self.batch_size = batch_size

        print(
            f"BatchedSynthDataLoader: Generating {size} samples"
            f" with batch size {batch_size} on {device}"
        )

        # Number of samples in the dataset is the size
        self.size = size
        self.steps = size // batch_size

        # Check if the device is valid
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("Dataloader: CUDA is not available, using CPU")
            self.device = "cpu"

        self.seed = seed
        self.return_sound = return_sound
        self.generator = torch.Generator(device="cpu")
        self.current = 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        # Step iterator or raise StopIteration
        if self.current >= self.steps:
            raise StopIteration
        self.current += 1

        # Generate the batch of parameters (on the CPU for reproducibility)
        self.generator.manual_seed(self.seed + (self.current * self.batch_size))
        params = torch.rand(
            self.batch_size,
            self.synth.get_num_params(),
            generator=self.generator,
            device="cpu",
        )
        params = params.to(self.device)

        if self.return_sound:
            sound = self.synth(params)
            return sound, params

        return (params,)


class BatchedSynthesizerDataModule(L.LightningDataModule):
    """
    A data module for synthesizer datasets
    """

    def __init__(
        self,
        synth: AbstractSynth,
        batch_size: int,
        device: str = "cpu",
        num_train: int = 1000000,
        num_val: int = 10000,
        num_test: int = 10000,
        seed: int = 649418400,
        return_sound: bool = True,
    ):
        super().__init__()
        self.synth = synth
        self.batch_size = batch_size
        self.device = device
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        self.return_sound = return_sound
        # print(f"Num train {num_train}, num val {num_val}, num test {num_test}")

    def train_dataloader(self):
        """
        Returns the train dataloader
        """
        return BatchedSynthDataLoader(
            self.synth,
            self.num_train,
            self.batch_size,
            device=self.device,
            seed=self.seed,
            return_sound=self.return_sound,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader
        """
        return BatchedSynthDataLoader(
            self.synth,
            self.num_val,
            self.batch_size,
            device=self.device,
            seed=self.seed + self.num_train,
            return_sound=self.return_sound,
        )

    def test_dataloader(self):
        """
        Returns the test dataloader
        """
        return BatchedSynthDataLoader(
            self.synth,
            self.num_test,
            self.batch_size,
            device=self.device,
            seed=self.seed + self.num_train + self.num_val,
            return_sound=self.return_sound,
        )
