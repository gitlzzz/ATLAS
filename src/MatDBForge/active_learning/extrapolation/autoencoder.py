"""Autoencoder model for dimensionality reduction and data reconstruction."""

import pathlib as pl

import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder model for dimensionality reduction and data reconstruction.

    This class implements a simple feedforward autoencoder model using PyTorch's
    `nn.Module`. It consists of two main components:

    - Encoder: Compresses the input data to a lower-dimensional
               bottleneck (latent space).
    - Decoder: Reconstructs the input data from the bottleneck representation.

    Attributes
    ----------
    encoder : nn.Sequential
        A neural network stack that reduces the input dimensionality down to the
        specified bottleneck dimension, capturing important features in a
        compressed form.

    decoder : nn.Sequential
        A neural network stack that reconstructs the input from the bottleneck
        representation, attempting to match the original input as closely as possible.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    l1_dim : int
        Dimension of the first hidden layer in the encoder (and the last hidden layer
         in the decoder).
    l2_dim : int
        Dimension of the second hidden layer in the encoder (and the second-to-last
         hidden layer in the decoder).
    bottleneck_dim : int, optional, default=2
        Dimension of the bottleneck layer (latent space) where input is compressed.
    bias_flag : bool, optional, default=True
        Whether to include a bias term in each linear layer.
        bias_flag set to false in principle is less dependant on the training data
         and is more generalizable to unseen data.



    Methods
    -------
    forward(x):
        Performs a forward pass through the encoder and decoder, compressing
        the input to the bottleneck and then reconstructing it back to its
         original dimension.

    Example Usage:
    --------------
    >>> model = Autoencoder(
    ...     input_dim=100, l1_dim=64, l2_dim=32, bottleneck_dim=10
    ... )
    >>> input_data = torch.randn(1, 100)
    >>> reconstructed_data = model(input_data)
    """

    def __init__(
        self,
        input_dim: int,
        l1_dim: int,
        l2_dim: int,
        bottleneck_dim: int = 2,
        bias_flag=True,
    ):
        super().__init__()

        # Encoder: Input -> Bottleneck (latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, l1_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l1_dim, l2_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l2_dim, bottleneck_dim, bias_flag),
        )

        # Decoder: Bottleneck -> Input (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, l2_dim, bias_flag),
            nn.ReLU(),
            nn.Linear(l2_dim, l1_dim, bias_flag),
            nn.ReLU(),
            nn.Linear(l1_dim, input_dim, bias_flag),
        )

    def forward(self, x):
        # Pass input through the encoder to obtain the latent space (bottleneck)
        # and reconstruct the input from the latent space
        return self.decoder(self.encoder(x))


def load_autoencoder_model(model_path: str, data_arr: np.ndarray):
    state_dict = torch.load(model_path)

    input_dim = data_arr.shape[1]
    l1_dim = state_dict["encoder.0.weight"].shape[0]
    l2_dim = state_dict["encoder.2.weight"].shape[0]

    # Check if the model already exists
    if pl.Path(model_path).exists():
        # model = torch.load(model_path)
        model = Autoencoder(
            input_dim=input_dim,
            l1_dim=l1_dim,
            l2_dim=l2_dim,
        )
        model.load_state_dict(state_dict)
        print("\nModel loaded successfully!")

    return model
