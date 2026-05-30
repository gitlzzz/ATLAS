"""Autoencoder model for dimensionality reduction and data reconstruction."""

import logging
import pathlib as pl
import warnings

import numpy as np
import torch
import torch.nn as nn

from atlas.core.code_utils import custom_print

# Silencing specific warnings and log messages
warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')

# Force third party loggers to only show errors and critical messages
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


def locate_standarization_files(autoencoder_path=pl.Path('.')):
    """Check for the existence of standardization files.

    These files are stored as numpy arrays containing values necessary
    to apply same standarization scale.

    Parameters
    ----------
    autoencoder_path: pl.Path
        Path for the autoencoder model

    Returns
    -------
    mean_vals: np.ndarray or False
        Array of mean values for each feature, or False if not found.
    """
    if isinstance(autoencoder_path, str):
        autoencoder_path = pl.Path(autoencoder_path)

    mean_vals, std_vals = None, None

    for file in autoencoder_path.parent.glob('*vals.npy'):
        if 'mean' in file.name:
            mean_vals = np.load(file)
        if 'std' in file.name:
            std_vals = np.load(file)

    return mean_vals, std_vals


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
            nn.Linear(l2_dim, bottleneck_dim, bias=bias_flag),
        )

        # Decoder: Bottleneck -> Input (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, l2_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l2_dim, l1_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l1_dim, input_dim, bias=bias_flag),
        )

    def forward(self, x):
        # Pass input through the encoder to obtain the latent space (bottleneck)
        # and reconstruct the input from the latent space
        return self.decoder(self.encoder(x))


def load_autoencoder_model(model_path: str, data_arr: np.ndarray = None):
    from torch.serialization import safe_globals

    # Only my classes and pytorch classes are allowed.
    # Only bugs in my classes should be dangerous.
    trusted_classes = [
        Autoencoder,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.activation.ReLU,
    ]

    # To allow loading of custom model classes without changing
    # to weights_only = False
    # Context manager is used to avoid affecting global state
    # and preventing other security risks.
    with safe_globals(trusted_classes):
        try:
            model = torch.load(model_path)
        except Exception as e:
            custom_print(
                f'Error loading model from {model_path}. '
                f'Please check the file path and format. Error: {e}',
                'error',
            )
            raise

    state_dict = model if isinstance(model, dict) else False

    if state_dict:
        input_dim = state_dict['encoder.0.weight'].shape[1]
        l1_dim = state_dict['encoder.0.weight'].shape[0]
        l2_dim = state_dict['encoder.2.weight'].shape[0]

        # Check if the model already exists
        if pl.Path(model_path).exists() and isinstance(model, dict):
            # Extract bottleneck_dim from the final encoder layer (layer 4)
            bottleneck_dim = state_dict['encoder.4.weight'].shape[0]

            # Check if bias keys exist in the state dict
            has_bias = 'encoder.0.bias' in state_dict

            model = Autoencoder(
                input_dim=input_dim,
                l1_dim=l1_dim,
                l2_dim=l2_dim,
                bottleneck_dim=bottleneck_dim,
                bias_flag=has_bias,
            )
            model.load_state_dict(state_dict)

    custom_print('Model loaded successfully!', 'done')

    return model


def get_latent_space_autoencoder(
    model,
    descriptor_dict: dict,
    device: str = None,
    dtype=torch.float32,
    quiet: bool = False,
    standardize_data: bool = False,
    autoencoder_path: str | pl.Path = None,
):
    if quiet:
        init_print_type = 'debug'
        final_print_type = 'debug'
    else:
        init_print_type = 'info'
        final_print_type = 'done'

    # Changing device to available GPU if available, else CPU.
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # Remember that I must call model.eval() to set dropout and batch
    # normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Standardize the data if required
    should_standardize: bool = standardize_data and (autoencoder_path is not None)

    # Loading standardization parameters from arrays in path
    if should_standardize:
        autoencoder_path = pl.Path(autoencoder_path)
        mean_vals, std_vals = locate_standarization_files(autoencoder_path)
        custom_print('Loaded standardization parameters.', init_print_type)

    # Reduce the dimensionality of the input points to 2D
    custom_print('Computing latent space for all structures...', init_print_type)

    # No need to compute gradients for inference
    with torch.no_grad():
        uuids = list(descriptor_dict.keys())

        # Gather all descriptors into a single list
        all_descrs = [descriptor_dict[u]['descriptors'][0] for u in uuids]
        all_lengths = [descriptor_dict[u]['descriptors'][0].shape[0] for u in uuids]
        descr_tensor = torch.tensor(np.vstack(all_descrs), device=device, dtype=dtype)

        # Standardize batched data
        if should_standardize and mean_vals is not None and std_vals is not None:
            mean_tensor = torch.tensor(mean_vals, device=device, dtype=dtype)
            std_tensor = torch.tensor(std_vals, device=device, dtype=dtype)
            descr_tensor = (descr_tensor - mean_tensor) / std_tensor

        # Run the ENTIRE batch through the model at once
        latent_batch = model.encoder(descr_tensor).cpu().numpy()

        # Map the results back to the dictionary
        ini_posc_desc_list = 0
        fin_posc_desc_list = 0
        for i, uuid in enumerate(uuids):
            curr_struct_length = all_lengths[i]
            fin_posc_desc_list += curr_struct_length

            descriptor_dict[uuid]['latent_space'] = latent_batch[
                ini_posc_desc_list:fin_posc_desc_list
            ]
            ini_posc_desc_list = fin_posc_desc_list

    custom_print('Computed latent space!', final_print_type)
    return descriptor_dict
