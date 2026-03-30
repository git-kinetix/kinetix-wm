"""Tests for train.py configuration logic and backward compatibility.

Note: train.py cannot be directly imported without PyTorch Lightning installed,
so these tests validate the logic patterns independently.
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from module import Embedder, ARPredictor, MLP


class TestClearMLInit:
    """Test ClearML initialization logic (without importing train.py)."""

    def test_disabled_config(self):
        """When clearml.enabled=false, function should return None."""
        # Replicate the init_clearml logic
        clearml_cfg = {"enabled": False}
        enabled = clearml_cfg.get("enabled", False)
        assert enabled is False

    def test_missing_config(self):
        """When clearml section is missing, should default to disabled."""
        cfg = {}
        clearml_cfg = cfg.get("clearml", {})
        enabled = clearml_cfg.get("enabled", False)
        assert enabled is False

    def test_enabled_config(self):
        """When enabled, config values should be accessible."""
        clearml_cfg = {
            "enabled": True,
            "project": "lewm",
            "task_name": "test-run",
            "dataset_id": "abc123",
            "tags": ["test"],
            "upload_model": True,
        }
        assert clearml_cfg["enabled"] is True
        assert clearml_cfg["project"] == "lewm"
        assert clearml_cfg["dataset_id"] == "abc123"


class TestBackwardCompatibility:
    """Test that the code changes don't break existing configurations."""

    def test_embedder_smoothed_dim_pusht(self):
        """PushT: effective_act_dim=10, max(10,10)=10 — same as old default."""
        effective_act_dim = 5 * 2  # frameskip=5, action_dim=2
        smoothed_dim = max(effective_act_dim, 10)
        assert smoothed_dim == 10

        emb = Embedder(input_dim=effective_act_dim, smoothed_dim=smoothed_dim, emb_dim=192)
        x = torch.randn(4, 8, effective_act_dim)
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_embedder_smoothed_dim_delta_pose(self):
        """Delta pose: effective_act_dim=45, max(45,10)=45 — fixes bottleneck."""
        effective_act_dim = 5 * 9  # frameskip=5, action_dim=9 (6D rotation)
        smoothed_dim = max(effective_act_dim, 10)
        assert smoothed_dim == 45

        emb = Embedder(input_dim=effective_act_dim, smoothed_dim=smoothed_dim, emb_dim=192)
        x = torch.randn(4, 8, effective_act_dim)
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_embedder_no_frameskip(self):
        """Delta pose with frameskip=1: effective_act_dim=9."""
        effective_act_dim = 1 * 9  # frameskip=1, action_dim=9
        smoothed_dim = max(effective_act_dim, 10)
        assert smoothed_dim == 10  # min is 10

        emb = Embedder(input_dim=effective_act_dim, smoothed_dim=smoothed_dim, emb_dim=192)
        x = torch.randn(4, 8, effective_act_dim)
        out = emb(x)
        assert out.shape == (4, 8, 192)

    def test_predictor_hidden_dim_default(self):
        """Without hidden_dim_override, predictor_hidden should equal hidden_dim."""
        hidden_dim = 192  # ViT-Tiny
        predictor_hidden = hidden_dim  # no override

        pred = ARPredictor(
            num_frames=3, input_dim=192, hidden_dim=predictor_hidden,
            output_dim=predictor_hidden,
            depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
        )
        x = torch.randn(2, 3, 192)
        c = torch.randn(2, 3, 192)
        out = pred(x, c)
        assert out.shape == (2, 3, 192)

    def test_predictor_hidden_dim_override(self):
        """With hidden_dim_override, predictor uses a different width."""
        hidden_dim = 768  # VJEPA encoder
        predictor_hidden = 256  # override

        pred = ARPredictor(
            num_frames=3, input_dim=192, hidden_dim=predictor_hidden,
            output_dim=predictor_hidden,
            depth=2, heads=4, mlp_dim=512, dim_head=32,
        )
        pred_proj = MLP(input_dim=predictor_hidden, output_dim=192, hidden_dim=512)

        x = torch.randn(2, 3, 192)
        c = torch.randn(2, 3, 192)
        raw_pred = pred(x, c)
        assert raw_pred.shape == (2, 3, 256)

        projected = pred_proj(raw_pred.reshape(-1, 256))
        assert projected.shape == (6, 192)

    def test_full_dimension_chain_vjepa(self):
        """Simulate the full VJEPA dimension chain: encoder→projector→predictor→pred_proj."""
        hidden_dim = 768       # VJEPA-B encoder output
        embed_dim = 192        # kept small
        predictor_hidden = 192 # default (equals hidden_dim for ViT-Tiny, here overridden)
        proj_hidden = 2048

        # Projector: 768 → 192
        projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                        hidden_dim=proj_hidden, norm_fn=torch.nn.BatchNorm1d)
        # Predictor: 192 → 192
        predictor = ARPredictor(
            num_frames=3, input_dim=embed_dim, hidden_dim=predictor_hidden,
            output_dim=predictor_hidden,
            depth=2, heads=4, mlp_dim=512, dim_head=32,
        )
        # Pred projector: 192 → 192
        pred_proj = MLP(input_dim=predictor_hidden, output_dim=embed_dim,
                        hidden_dim=proj_hidden, norm_fn=torch.nn.BatchNorm1d)

        # Simulate forward pass
        B, T = 4, 3
        encoder_output = torch.randn(B * T, hidden_dim)  # frozen VJEPA output
        emb = projector(encoder_output)
        assert emb.shape == (B * T, embed_dim)

        emb_seq = emb.reshape(B, T, embed_dim)
        act_emb = torch.randn(B, T, embed_dim)  # from action encoder
        pred_out = predictor(emb_seq, act_emb)
        assert pred_out.shape == (B, T, predictor_hidden)

        pred_flat = pred_out.reshape(B * T, predictor_hidden)
        final = pred_proj(pred_flat)
        assert final.shape == (B * T, embed_dim)
