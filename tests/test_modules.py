"""Tests for module.py components: Embedder, MLP, ARPredictor, SIGReg."""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from module import Embedder, MLP, SIGReg, ARPredictor


class TestEmbedder:
    """Test the Embedder with various smoothed_dim values."""

    def test_default_smoothed_dim(self):
        """Original behavior: smoothed_dim=10."""
        emb = Embedder(input_dim=10, smoothed_dim=10, emb_dim=192)
        x = torch.randn(4, 8, 10)  # (B, T, D)
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_large_smoothed_dim(self):
        """Fixed behavior: smoothed_dim matches input_dim for delta poses."""
        emb = Embedder(input_dim=45, smoothed_dim=45, emb_dim=192)
        x = torch.randn(4, 8, 45)  # frameskip=5 * action_dim=9
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_smoothed_dim_bottleneck(self):
        """Regression test: smoothed_dim=10 with input_dim=45 still works (just compresses)."""
        emb = Embedder(input_dim=45, smoothed_dim=10, emb_dim=192)
        x = torch.randn(4, 8, 45)
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_max_smoothed_dim_logic(self):
        """Test the max(effective_act_dim, 10) logic from train.py."""
        for act_dim in [6, 10, 35, 45]:
            sd = max(act_dim, 10)
            emb = Embedder(input_dim=act_dim, smoothed_dim=sd, emb_dim=128)
            x = torch.randn(2, 4, act_dim)
            out = emb(x)
            assert out.shape == (2, 4, 128)


class TestMLP:
    """Test the MLP module."""

    def test_basic(self):
        mlp = MLP(input_dim=192, output_dim=192, hidden_dim=2048)
        x = torch.randn(16, 192)
        out = mlp(x)
        assert out.shape == (16, 192)

    def test_with_batchnorm(self):
        mlp = MLP(input_dim=768, output_dim=192, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        x = torch.randn(32, 768)
        out = mlp(x)
        assert out.shape == (32, 192)

    def test_dimension_mismatch(self):
        """Projector-like: large input, small output."""
        mlp = MLP(input_dim=1024, output_dim=192, hidden_dim=4096)
        x = torch.randn(8, 1024)
        out = mlp(x)
        assert out.shape == (8, 192)


class TestSIGReg:
    """Test the SIGReg regularizer."""

    def test_basic_shape(self):
        sigreg = SIGReg(knots=17, num_proj=1024)
        proj = torch.randn(4, 32, 192)  # (T, B, D)
        loss = sigreg(proj)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gaussian_input(self):
        """SIGReg loss should be lower for Gaussian-distributed input."""
        sigreg = SIGReg(knots=17, num_proj=1024)

        # Standard Gaussian input
        gaussian = torch.randn(4, 128, 192)
        loss_gaussian = sigreg(gaussian)

        # Highly non-Gaussian (constant + noise)
        degenerate = torch.ones(4, 128, 192) * 5.0 + torch.randn(4, 128, 192) * 0.01
        loss_degenerate = sigreg(degenerate)

        # Gaussian should have lower loss
        assert loss_gaussian.item() < loss_degenerate.item()

    def test_different_embed_dims(self):
        """Test SIGReg works with various embedding dimensions."""
        for dim in [64, 192, 384, 768]:
            sigreg = SIGReg(knots=17, num_proj=1024)
            proj = torch.randn(4, 32, dim)
            loss = sigreg(proj)
            assert loss.shape == ()


class TestARPredictor:
    """Test the autoregressive predictor."""

    def test_basic_shape(self):
        pred = ARPredictor(
            num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
            depth=2, heads=4, mlp_dim=512, dim_head=32,
        )
        x = torch.randn(4, 3, 192)  # (B, T, D)
        c = torch.randn(4, 3, 192)  # (B, T, D) conditioning
        out = pred(x, c)
        assert out.shape == (4, 3, 192)

    def test_decoupled_dims(self):
        """Test with input_dim != hidden_dim (VJEPA scenario)."""
        pred = ARPredictor(
            num_frames=3, input_dim=192, hidden_dim=384, output_dim=384,
            depth=2, heads=4, mlp_dim=512, dim_head=32,
        )
        x = torch.randn(4, 3, 192)
        c = torch.randn(4, 3, 192)
        out = pred(x, c)
        assert out.shape == (4, 3, 384)

    def test_truncated_sequence(self):
        """Predictor should handle T < num_frames via pos_embedding slicing."""
        pred = ARPredictor(
            num_frames=5, input_dim=192, hidden_dim=192, output_dim=192,
            depth=1, heads=4, mlp_dim=256, dim_head=32,
        )
        x = torch.randn(2, 2, 192)  # T=2 < num_frames=5
        c = torch.randn(2, 2, 192)
        out = pred(x, c)
        assert out.shape == (2, 2, 192)
