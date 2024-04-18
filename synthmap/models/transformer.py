"""
Transformer Architecture
"""
import torch
from einops import rearrange
from einops import repeat


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
        num_layers: int = 6,
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
        x = rearrange(x, "b f s -> b s f")

        # Add class token and append to the beginning of the input sequence
        tokens = repeat(self.cls_token, "() n d -> b n d", b=x.shape[0])
        x = torch.cat((tokens, x), dim=1)

        # Apply positional encoding
        x = x + self.pos_emb

        out = self.transformer(x)
        out = self.proj(out[:, 0, :])
        return out
