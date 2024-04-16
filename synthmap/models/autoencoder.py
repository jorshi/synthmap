"""
Models
"""
import torch


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        bottleneck: bool = False,
        beta: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if bottleneck:
            self.bottleneck = GaussianVAE(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        kl = None
        if hasattr(self, "bottleneck"):
            z, kl = self.bottleneck(z)
        y = self.decoder(z)
        return y, z, kl


class GaussianVAE(torch.nn.Module):
    """
    Sets an encoder bottleneck for reparametrization of posterior as
    multimodal gaussian (diagonal covariance).
    The prior is assumed to be N(z;0,1)
    """

    def __init__(self, beta: float = 0.2):
        super().__init__()
        self.beta = beta

    def reparametrize(self, z: torch.tensor):
        """Sample from a parameterized gaussian given as an input.
        Args:
        z (torch.tensor): A batch of inputs where the parameterized Gaussian
            is at dim=1.
        Returns:
        A tuple containing the sampled vector (with dim=2 halved),
        and the kl divergence.
        """
        mean, scale = torch.chunk(z, 2, -1)
        std = torch.nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = 0.5 * (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl * self.beta

    def forward(self, z: torch.Tensor):
        z, kl = self.reparametrize(z)
        return z, kl
