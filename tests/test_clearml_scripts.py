"""Tests for the ClearML helper scripts (dataset, train launcher)."""

import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# clearml_dataset.py calls sys.exit(1) at import time if clearml is not installed.
# Detect this and skip those tests.
try:
    import clearml
    _has_clearml = True
except ImportError:
    _has_clearml = False


@pytest.mark.skipif(not _has_clearml, reason="clearml package not installed")
class TestClearMLDatasetCLI:
    """Test the clearml_dataset.py argument parsing."""

    def test_upload_args(self):
        from clearml_dataset import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "upload",
            "--dataset-name", "prod_beta0",
            "--project", "lewm/data",
            "--h5-path", "/tmp/test.h5",
            "--tags", "synthetic", "beta-0",
        ])
        assert args.command == "upload"
        assert args.dataset_name == "prod_beta0"
        assert args.project == "lewm/data"
        assert args.h5_path == "/tmp/test.h5"
        assert args.tags == ["synthetic", "beta-0"]

    def test_download_args_by_id(self):
        from clearml_dataset import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "download",
            "--dataset-id", "abc123",
            "--output-dir", "/tmp/data",
        ])
        assert args.command == "download"
        assert args.dataset_id == "abc123"
        assert args.output_dir == "/tmp/data"

    def test_download_args_by_name(self):
        from clearml_dataset import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "download",
            "--dataset-name", "prod_beta0",
            "--dataset-project", "lewm/data",
            "--output-dir", "/tmp/data",
        ])
        assert args.command == "download"
        assert args.dataset_name == "prod_beta0"
        assert args.dataset_project == "lewm/data"

    def test_list_args(self):
        from clearml_dataset import build_parser
        parser = build_parser()
        args = parser.parse_args(["list", "--project", "lewm/data"])
        assert args.command == "list"
        assert args.project == "lewm/data"


class TestClearMLTrainCLI:
    """Test the clearml_train.py argument parsing and helpers."""

    def test_parse_args(self):
        from clearml_train import parse_args
        args = parse_args([
            "--project", "lewm",
            "--task-name", "vjepa-frozen-beta0",
            "--queue", "gpu",
            "--data", "prod_beta0",
            "--overrides", "encoder_type=vjepa", "encoder_frozen=true",
            "--tags", "experiment", "vjepa",
        ])
        assert args.project == "lewm"
        assert args.task_name == "vjepa-frozen-beta0"
        assert args.queue == "gpu"
        assert args.data == "prod_beta0"
        assert args.overrides == ["encoder_type=vjepa", "encoder_frozen=true"]
        assert args.tags == ["experiment", "vjepa"]

    def test_looks_like_clearml_id(self):
        from clearml_train import _looks_like_clearml_id
        assert _looks_like_clearml_id("a" * 32)
        assert _looks_like_clearml_id("0123456789abcdef0123456789abcdef")
        assert not _looks_like_clearml_id("short")
        assert not _looks_like_clearml_id("/path/to/file.pth")
        assert not _looks_like_clearml_id("a" * 31)
        assert not _looks_like_clearml_id("a" * 33)

    def test_parse_args_defaults(self):
        from clearml_train import parse_args
        args = parse_args([])
        assert args.project == "lewm"
        assert args.task_name == "lewm-train"
        assert args.queue is None
        assert args.dataset_id is None
        assert args.data is None
        assert args.overrides == []
