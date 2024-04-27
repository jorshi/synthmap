"""
Transformer Architecture
"""
import torch
from einops import rearrange
from einops import repeat

from synthmap.models.components import ParameterScaler


class TransformerAggegrate(torch.nn.Module):
    """
    A transformer layer that aggregates a sequence of input vectors input
    a single output vector.
    Based on: https://arxiv.org/pdf/2204.11479.pdf
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        clip_length: int,
        num_layers: int = 4,
        nhead: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_length = clip_length

        # Create the transormer encoder
        tlayer = torch.nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            activation="gelu",
            batch_first=True,
            dim_feedforward=512,
        )
        self.transformer = torch.nn.TransformerEncoder(
            tlayer, num_layers=num_layers, norm=torch.nn.LayerNorm(self.input_dim)
        )

        # Output projection
        self.proj = torch.nn.Linear(self.input_dim, self.output_dim)

        # Class tokens
        self.num_tokens = 2
        self.cls_token = torch.nn.Parameter(
            torch.zeros(1, self.num_tokens, self.input_dim)
        )

        # Positional encoding
        self.pos_emb = torch.nn.Parameter(
            torch.zeros(1, self.clip_length + self.num_tokens, self.input_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = rearrange(x, "b d n -> b n d")

        # Add class token and append to the beginning of the input sequence
        tokens = repeat(self.cls_token, "() n d -> b n d", b=x.shape[0])
        x = torch.cat((tokens, x), dim=1)

        # Apply positional encoding
        x = x + self.pos_emb

        out = self.transformer(x)
        out = self.proj(out[:, 0, :])
        return out


class TransformerEncoder(torch.nn.Module):
    """
    A transformer encoder that receives parameters from a single input vector
    and generates a single latent representation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        clip_length: int,
        num_layers: int = 4,
        nhead: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.clip_length = clip_length
        self.in_proj = torch.nn.Linear(1, input_dim)
        self.transformer = TransformerAggegrate(
            input_dim, output_dim, clip_length, num_layers, nhead
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., None]
        x = self.in_proj(x)
        x = rearrange(x, "b n d -> b d n")
        return self.transformer(x)


class TransformerDecoder(torch.nn.Module):
    """
    A transformer decoder that takes a single input vector and
    generates a sequence of output vectors.
    """

    def __init__(
        self,
        input_dim: int,
        clip_length: int,
        num_layers: int = 4,
        nhead: int = 8,
        scale_output: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.clip_length = clip_length
        if scale_output:
            self.scale_output = ParameterScaler(
                max_value=1.0, threshold=0.0, exponent=10.0
            )

        # Create the transormer decoder
        tlayer = torch.nn.TransformerDecoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            activation="gelu",
            batch_first=True,
            dim_feedforward=512,
        )
        self.transformer = torch.nn.TransformerDecoder(
            tlayer, num_layers=num_layers, norm=torch.nn.LayerNorm(self.input_dim)
        )

        # Output projection
        self.proj = torch.nn.Linear(self.input_dim, 1)

        # Positional encoding
        self.pos_emb = torch.nn.Parameter(
            torch.zeros(1, self.clip_length, self.input_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = rearrange(x, "b f -> b 1 f")

        pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=x.shape[0])

        y = self.transformer(pos_emb, x)
        y = self.proj(y)
        if hasattr(self, "scale_output"):
            y = self.scale_output(y)

        return y[..., 0]
