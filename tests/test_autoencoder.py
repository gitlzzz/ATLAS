"""Tests for the dimensionality-reduction Autoencoder model."""

import torch

from atlas.active_learning.extrapolation.autoencoder import Autoencoder


class TestAutoencoder:
    """Shape/contract tests for the autoencoder forward pass."""

    def test_encoder_maps_to_bottleneck_dim(self):
        torch.manual_seed(0)
        model = Autoencoder(input_dim=16, l1_dim=8, l2_dim=4, bottleneck_dim=2)
        x = torch.randn(5, 16)
        latent = model.encoder(x)
        assert latent.shape == (5, 2)

    def test_forward_reconstructs_input_shape(self):
        torch.manual_seed(0)
        model = Autoencoder(input_dim=16, l1_dim=8, l2_dim=4, bottleneck_dim=2)
        x = torch.randn(5, 16)
        out = model(x)
        assert out.shape == x.shape

    def test_no_bias_option(self):
        model = Autoencoder(
            input_dim=8, l1_dim=4, l2_dim=4, bottleneck_dim=2, bias_flag=False
        )
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert linears
        assert all(layer.bias is None for layer in linears)
