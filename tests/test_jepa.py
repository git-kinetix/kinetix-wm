"""Tests for JEPA dual-path encode (CLS vs mean pooling)."""

import torch
import torch.nn as nn
from einops import rearrange

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


class MockHFEncoder(nn.Module):
    """Mock HuggingFace ViT encoder returning an object with last_hidden_state."""

    def __init__(self, hidden_dim=192, num_patches=16):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.linear = nn.Linear(3 * 224 * 224, hidden_dim)  # dummy

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        B = pixel_values.shape[0]
        # Simulate (B, num_patches+1, hidden_dim) with CLS token at position 0
        tokens = torch.randn(B, self.num_patches + 1, self.hidden_dim, device=pixel_values.device)
        return type("Output", (), {"last_hidden_state": tokens})()


class MockVJEPAEncoder(nn.Module):
    """Mock VJEPA encoder returning a raw tensor of patch tokens (no CLS)."""

    def __init__(self, embed_dim=768, num_patches=196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.dummy = nn.Linear(1, 1)  # ensure parameters() is non-empty

    def forward(self, x):
        # x could be (B, C, H, W) or (B, C, T, H, W)
        B = x.shape[0]
        return torch.randn(B, self.num_patches, self.embed_dim, device=x.device)


class MockTimmEncoder(nn.Module):
    """Mock timm ViT encoder with num_classes=0 returning 2D pooled output."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        B = x.shape[0]
        return torch.randn(B, self.embed_dim, device=x.device)


def _build_jepa(encoder, pooling, hidden_dim, embed_dim=192, action_dim=10):
    """Helper to build a JEPA with the given encoder and pooling strategy."""
    predictor = ARPredictor(
        num_frames=3,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=1, heads=4, mlp_dim=256, dim_head=32,
    )
    action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=256)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=256)

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        pooling=pooling,
    )


class TestJEPAEncodeCLS:
    """Test the CLS pooling path (HuggingFace ViT)."""

    def test_encode_shape(self):
        encoder = MockHFEncoder(hidden_dim=192)
        model = _build_jepa(encoder, pooling="cls", hidden_dim=192)

        B, T = 2, 4
        info = {
            "pixels": torch.randn(B, T, 3, 224, 224),
            "action": torch.randn(B, T, 10),
        }
        output = model.encode(info)

        assert "emb" in output
        assert output["emb"].shape == (B, T, 192)
        assert "act_emb" in output
        assert output["act_emb"].shape == (B, T, 192)

    def test_encode_no_action(self):
        encoder = MockHFEncoder(hidden_dim=192)
        model = _build_jepa(encoder, pooling="cls", hidden_dim=192)

        info = {"pixels": torch.randn(2, 3, 3, 224, 224)}
        output = model.encode(info)

        assert "emb" in output
        assert "act_emb" not in output


class TestJEPAEncodeMean:
    """Test the mean pooling path (VJEPA)."""

    def test_encode_3d_output(self):
        """VJEPA returning (B, num_patches, D) — needs mean pooling."""
        encoder = MockVJEPAEncoder(embed_dim=768)
        model = _build_jepa(encoder, pooling="mean", hidden_dim=768)

        B, T = 2, 4
        info = {
            "pixels": torch.randn(B, T, 3, 224, 224),
            "action": torch.randn(B, T, 10),
        }
        output = model.encode(info)

        assert output["emb"].shape == (B, T, 192)
        assert output["act_emb"].shape == (B, T, 192)

    def test_encode_2d_output(self):
        """timm ViT with num_classes=0 returning (B, D) — already pooled."""
        encoder = MockTimmEncoder(embed_dim=768)
        model = _build_jepa(encoder, pooling="mean", hidden_dim=768)

        B, T = 2, 4
        info = {
            "pixels": torch.randn(B, T, 3, 224, 224),
            "action": torch.randn(B, T, 10),
        }
        output = model.encode(info)

        assert output["emb"].shape == (B, T, 192)


class TestJEPAPredict:
    """Test the predict method."""

    def test_predict_shape(self):
        encoder = MockHFEncoder(hidden_dim=192)
        model = _build_jepa(encoder, pooling="cls", hidden_dim=192)

        B, T = 2, 3
        emb = torch.randn(B, T, 192)
        act_emb = torch.randn(B, T, 192)

        preds = model.predict(emb, act_emb)
        assert preds.shape == (B, T, 192)


class TestJEPARollout:
    """Test the rollout method."""

    def test_rollout_shape(self):
        encoder = MockHFEncoder(hidden_dim=192)
        model = _build_jepa(encoder, pooling="cls", hidden_dim=192)

        B, S, T = 1, 2, 6  # 1 batch, 2 samples, 6 timesteps
        H = 3  # history size

        info = {
            "pixels": torch.randn(B, S, H, 3, 224, 224),
        }
        action_seq = torch.randn(B, S, T, 10)

        output = model.rollout(info, action_seq, history_size=H)

        assert "predicted_emb" in output
        # predicted_emb should have T+1 timesteps (initial H + T-H predictions + 1 final)
        assert output["predicted_emb"].shape[0] == B
        assert output["predicted_emb"].shape[1] == S
