"""
Genetic Algorithm for Generating Synthesizer Datasets
"""
from typing import List
from typing import Union

import lightning as L
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import PolynomialMutation
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
        return_audio=True,
        return_evals=False,
        reset_on_epoch=True,
        verbose=False,
    ):
        super().__init__()
        self.synth = synth
        self.size = size
        self.batch_size = batch_size
        self.fitness_fns = fitness_fns
        self.device = device
        self.return_audio = return_audio
        self.return_evals = return_evals
        self.reset_on_epoch = reset_on_epoch
        self.verbose = verbose
        self.current = 0
        self.audio = None

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

        batch = {"params": params}
        if self.return_audio:
            batch["audio"] = self.audio
        if self.return_evals:
            # Get best value for each objective
            evals = torch.zeros(len(self.fitness_fns))
            for i in range(len(self.fitness_fns)):
                evals[i] = self.ga.population.evals[self.ga.population.argbest(i), i]
            batch["evals"] = evals

        return batch

    def create_problem(self):
        """
        Create a problem for the genetic algorithm
        """
        # Create the problem
        opt = []
        for i, fit in enumerate(self.fitness_fns):
            print(f"Creating problem for fitness function {i}, device {self.device}")
            self.fitness_fns[i].to(self.device)
            self.fitness_fns[i].reset()
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
        self.ga = GeneticAlgorithm(
            self.problem,
            popsize=self.batch_size,
            elitist=True,
            operators=[
                SimulatedBinaryCrossOver(
                    self.problem,
                    tournament_size=4,
                    cross_over_rate=1.0,
                    eta=8,
                ),
                PolynomialMutation(
                    self.problem,
                    eta=5.0,
                    mutation_probability=0.5,
                ),
            ],
        )

        if self.verbose:
            self.logger = StdOutLogger(self.ga)

    def fitness(self, x: torch.Tensor):
        """
        Fitness function for the genetic algorithm
        """
        # Evaluate the synthesizer
        audios = self.synth(torch.clamp(x, 0.0, 1.0))
        self.audio = audios

        fitness = []
        for fit in self.fitness_fns:
            fitness.extend(fit(audios))

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
        return_audio: bool = True,
        return_evals: bool = False,
        reset_on_epoch: bool = False,
    ):
        super().__init__()
        self.synth = synth
        self.batch_size = batch_size
        self.train_steps = num_train // batch_size
        self.val_steps = num_val // batch_size
        self.test_steps = num_test // batch_size
        self.return_audio = return_audio
        self.return_evals = return_evals
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
            return_audio=self.return_audio,
            return_evals=self.return_evals,
            reset_on_epoch=self.reset_on_epoch,
        )
