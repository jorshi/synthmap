"""
Functions for handling synthesizer parameters
"""
import torch
from einops import rearrange


class DiscretizedNumericalParameters(torch.nn.Module):
    """
    Handle the discretization and conversion of continuous numerical parameters
    """

    def __init__(self, num_params: int, steps_per_param: int):
        super().__init__()
        self.num_params = num_params
        self.steps_per_param = steps_per_param

    @property
    def num_discrete_params(self):
        return self.num_params * self.steps_per_param

    def discretize(self, x: torch.Tensor):
        """
        Discretize the input tensor and convert to a one-hot representation
        """
        assert x.shape[-1] == self.num_params
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert x.ndim == 2

        x = torch.split(x, 1, dim=-1)
        x = [torch.floor(p * self.steps_per_param).long() for p in x]
        x = [torch.nn.functional.one_hot(p, self.steps_per_param).float() for p in x]
        x = torch.cat(x, dim=-2)
        x = rearrange(x, "b p c-> b c p")
        return x

    def group_parameters(self, x: torch.Tensor):
        """
        Group a flattened set of discrete parameters into a batch of parameters
        Returns parameters in format expected by CrossEntropyLoss (logits)
        (batch_class, classes, parameter)
        """
        assert x.shape[-1] == self.num_discrete_params
        assert x.ndim == 2
        x = rearrange(x, "b (p c) -> b c p", c=self.steps_per_param)
        return x

    def flatten_parameters(self, x: torch.Tensor):
        """
        Flatten a batch of parameters into a single tensor
        """
        assert x.ndim == 3
        x = rearrange(x, "b c p -> b (c p)")
        return x

    def inverse(self, x: torch.Tensor):
        """
        Inverse of the discretize function
        """
        assert x.ndim == 3
        x = rearrange(x, "b c p -> b p c")
        x = torch.argmax(x, dim=-1).float() / self.steps_per_param
        return x
