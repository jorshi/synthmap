"""
Genetic Algorithm for Generating Synthesizer Datasets
"""
from typing import List
from typing import Union

import lightning as L
import torch
from evotorch import Problem
from evotorch.algorithms import SteadyStateGA
from evotorch.operators import GaussianMutation
from evotorch.operators import SimulatedBinaryCrossOver

from synthmap.data.fitness import FitnessFunctionBase
from synthmap.synth import AbstractSynth


class GeneticSynthDataLoader(torch.nn.Module):
    """
    Uses a genetic algorithm to evolve synthesizer parameters during training
    for datasets.
    """

    def __init__(
        self,
        synth: AbstractSynth,
        size: int,
        batch_size: int,
        fitness_fns: List[FitnessFunctionBase],
        device: str = "cpu",
        return_sound=True,
        reset_on_epoch=True,
    ):
        super().__init__()
        self.synth = synth
        self.size = size
        self.batch_size = batch_size
        self.fitness_fns = fitness_fns
        self.device = device
        self.return_sound = return_sound
        self.reset_on_epoch = reset_on_epoch
        self.current = 0

        # Create the problem
        self.create_problem()

    def __len__(self):
        return self.size

    def __iter__(self):
        self.current = 0
        if self.reset_on_epoch:
            print("Resetting on epoch")
            self.create_problem()
        return self

    def __next__(self):
        # Step iterator or raise StopIteration
        if self.current >= self.size:
            raise StopIteration
        self.current += 1

        # Step the genetic algorithm
        self.ga.step()

        # Get the solution
        params = self.ga.population.values.clone()

        return (params,)

    def create_problem(self):
        """
        Create a problem for the genetic algorithm
        """
        # Create the problem
        opt = []
        for fit in self.fitness_fns:
            opt.extend(fit.objective)

        self.problem = Problem(
            opt,
            self.fitness,
            initial_bounds=(0.0, 1.0),
            bounds=(0.0, 1.0),
            solution_length=self.synth.get_num_params(),
            vectorized=True,
            device=self.device,
        )

        # Create the genetic algorithm
        self.ga = SteadyStateGA(self.problem, popsize=self.batch_size, re_evaluate=True)
        self.ga.use(
            SimulatedBinaryCrossOver(
                self.problem,
                tournament_size=4,
                cross_over_rate=1.0,
                eta=8,
            )
        )
        self.ga.use(GaussianMutation(self.problem, stdev=0.3))
        # self.logger = StdOutLogger(self.ga)

    def fitness(self, x: torch.Tensor):
        """
        Fitness function for the genetic algorithm
        """
        # Evaluate the synthesizer
        sounds = self.synth(torch.clamp(x, 0.0, 1.0))

        fitness = []
        for fit in self.fitness_fns:
            fitness.extend(fit(sounds))

        return torch.stack(fitness, dim=-1)


class GeneticSynthesizerDataModule(L.LightningDataModule):
    """
    A data module for synthesizer datasets
    """

    def __init__(
        self,
        synth: AbstractSynth,
        fitness_fns: Union[FitnessFunctionBase, list[FitnessFunctionBase]],
        batch_size: int,
        num_train: int = 1000000,
        num_val: int = 10000,
        num_test: int = 10000,
        return_sound: bool = True,
        reset_on_epoch: bool = False,
    ):
        super().__init__()
        self.synth = synth
        self.batch_size = batch_size
        self.train_steps = num_train // batch_size
        self.val_steps = num_val // batch_size
        self.test_steps = num_test // batch_size
        self.return_sound = return_sound
        self.reset_on_epoch = reset_on_epoch
        self.fitness_fns = fitness_fns

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        pass

    def train_dataloader(self):
        """
        Returns the train dataloader
        """
        return GeneticSynthDataLoader(
            self.synth,
            self.train_steps,
            self.batch_size,
            fitness_fns=self.fitness_fns,
            return_sound=self.return_sound,
            reset_on_epoch=self.reset_on_epoch,
        )

    # def val_dataloader(self):
    #     """
    #     Returns the validation dataloader
    #     """
    #     return GeneticSynthDataLoader(
    #         self.synth,
    #         self.val_steps,
    #         self.batch_size,
    #         fitness_fns=self.fitness_fns,
    #         return_sound=self.return_sound,
    #         reset_on_epoch=self.reset_on_epoch,
    #     )

    # def test_dataloader(self):
    #     """
    #     Returns the test dataloader
    #     """
    #     return GeneticSynthDataLoader(
    #         self.synth,
    #         self.test_steps,
    #         self.batch_size,
    #         fitness_fns=self.fitness_fns,
    #         return_sound=self.return_sound,
    #         reset_on_epoch=self.reset_on_epoch,
    #     )
