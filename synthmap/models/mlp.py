import torch


class MLP(torch.nn.Module):
    """
    Configurable multilayer perceptron
    """

    def __init__(
        self,
        in_size: int,  # Input parameter size
        hidden_size: int,  # Hidden layer size
        out_size: int,  # Output parameter size
        num_layers: int,  # Number of hidden layers
        activation: torch.nn.Module = torch.nn.Sigmoid(),  # Activation function
        scale_output: bool = False,  # Scale output to [-1, 1]
        input_bias: float = 0.0,  # Bias for the input layer
        layer_norm: bool = False,  # Use layer normalization
        normalize_input: bool = False,  # Normalize input
        init_std: float = 1e-3,  # Standard deviation of initial weights
    ):
        super().__init__()
        channels = [in_size] + (num_layers) * [hidden_size]
        net = []
        for i in range(num_layers):
            net.append(torch.nn.Linear(channels[i], channels[i + 1]))
            if layer_norm:
                net.append(
                    torch.nn.LayerNorm(channels[i + 1], elementwise_affine=False)
                )
            net.append(activation)

        net.append(torch.nn.Linear(channels[-1], out_size))
        self.in_size = in_size
        self.net = torch.nn.Sequential(*net)
        self.scale_output = scale_output
        self.input_bias = input_bias
        self.normalize_input = normalize_input
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.normal_(m.weight, 0, self.init_std)

    def forward(self, x: torch.Tensor):
        x = x + self.input_bias
        x = self.net(x)
        if self.scale_output:
            x = torch.tanh(x)
        return x
