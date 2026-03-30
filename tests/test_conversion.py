"""Tests for the NPZ→HDF5 conversion script rotation helpers and I/O."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from convert_npz_to_hdf5 import (
    quaternion_to_rotation_matrix,
    euler_to_rotation_matrix,
    rotation_matrix_to_6d,
    rotation_matrix_to_axis_angle,
    convert_rotation,
    format_delta_pose,
    write_hdf5,
    validate_hdf5,
)


class TestQuaternionToMatrix:
    def test_identity(self):
        """Identity quaternion (1,0,0,0) should give identity matrix."""
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_90_deg_z(self):
        """90-degree rotation about Z axis."""
        q = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])
        R = quaternion_to_rotation_matrix(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R[0], expected, atol=1e-10)

    def test_batch(self):
        """Batch of quaternions."""
        q = np.random.randn(10, 4)
        R = quaternion_to_rotation_matrix(q)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self):
        """Rotation matrices should be orthogonal (R^T R = I)."""
        q = np.random.randn(5, 4)
        R = quaternion_to_rotation_matrix(q)
        for i in range(5):
            np.testing.assert_allclose(R[i].T @ R[i], np.eye(3), atol=1e-8)


class TestEulerToMatrix:
    def test_identity(self):
        euler = np.array([[0.0, 0.0, 0.0]])
        R = euler_to_rotation_matrix(euler)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_90_deg_z(self):
        euler = np.array([[0.0, 0.0, np.pi / 2]])
        R = euler_to_rotation_matrix(euler)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R[0], expected, atol=1e-10)

    def test_orthogonality(self):
        euler = np.random.randn(5, 3) * 0.5
        R = euler_to_rotation_matrix(euler)
        for i in range(5):
            np.testing.assert_allclose(R[i].T @ R[i], np.eye(3), atol=1e-8)


class TestRotationMatrix6D:
    def test_shape(self):
        R = np.eye(3).reshape(1, 3, 3)
        r6d = rotation_matrix_to_6d(R)
        assert r6d.shape == (1, 6)

    def test_identity(self):
        R = np.eye(3).reshape(1, 3, 3)
        r6d = rotation_matrix_to_6d(R)
        # R[:, :2] = [[1,0],[0,1],[0,0]], flattened = [1,0,0,1,0,0]
        expected = np.array([[1, 0, 0, 1, 0, 0]], dtype=float)
        np.testing.assert_allclose(r6d, expected, atol=1e-10)

    def test_batch(self):
        R = np.random.randn(10, 3, 3)
        r6d = rotation_matrix_to_6d(R)
        assert r6d.shape == (10, 6)


class TestAxisAngle:
    def test_identity(self):
        R = np.eye(3).reshape(1, 3, 3)
        aa = rotation_matrix_to_axis_angle(R)
        np.testing.assert_allclose(aa[0], np.zeros(3), atol=1e-6)

    def test_small_rotation(self):
        """Small rotation should give small axis-angle."""
        angle = 0.1
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]).reshape(1, 3, 3)
        aa = rotation_matrix_to_axis_angle(R)
        np.testing.assert_allclose(np.linalg.norm(aa[0]), angle, atol=1e-6)


class TestConvertRotation:
    def test_quaternion_to_6d(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        r6d = convert_rotation(q, "quaternion", "6d")
        assert r6d.shape == (1, 6)
        assert r6d.dtype == np.float32

    def test_euler_to_6d(self):
        euler = np.array([[0.0, 0.0, 0.0]])
        r6d = convert_rotation(euler, "euler", "6d")
        assert r6d.shape == (1, 6)

    def test_quaternion_to_axis_angle(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        aa = convert_rotation(q, "quaternion", "axis_angle")
        assert aa.shape == (1, 3)
        np.testing.assert_allclose(aa[0], np.zeros(3), atol=1e-6)

    def test_identity_passthrough(self):
        """quaternion source with quaternion target should return input."""
        q = np.array([[0.7071, 0.0, 0.7071, 0.0]])
        out = convert_rotation(q, "quaternion", "quaternion")
        np.testing.assert_allclose(out, q.astype(np.float32), atol=1e-4)


class TestFormatDeltaPose:
    def test_quaternion_input(self):
        """7D input (xyz + quaternion) converted to 6D rotation."""
        dp = np.random.randn(10, 7)
        dp[:, 3:7] = dp[:, 3:7] / np.linalg.norm(dp[:, 3:7], axis=1, keepdims=True)
        out = format_delta_pose(dp, rotation_repr="6d")
        assert out.shape == (10, 9)  # 3 translation + 6 rotation
        assert out.dtype == np.float32

    def test_euler_input(self):
        """6D input (xyz + euler) converted to 6D rotation."""
        dp = np.random.randn(10, 6)
        out = format_delta_pose(dp, rotation_repr="6d")
        assert out.shape == (10, 9)

    def test_passthrough_euler(self):
        """6D input with euler target should passthrough."""
        dp = np.random.randn(10, 6)
        out = format_delta_pose(dp, rotation_repr="euler")
        assert out.shape == (10, 6)

    def test_invalid_dim(self):
        """Should raise for unexpected dimension."""
        dp = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="expected 6 or 7"):
            format_delta_pose(dp, rotation_repr="6d")


class TestWriteAndValidateHDF5:
    def test_roundtrip(self, tmp_path):
        """Write HDF5, validate, and check contents."""
        episodes = [
            {
                "actions": np.random.randn(20, 9).astype(np.float32),
                "pixels": np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
            },
            {
                "actions": np.random.randn(15, 9).astype(np.float32),
                "pixels": np.random.randint(0, 255, (15, 64, 64, 3), dtype=np.uint8),
            },
        ]

        output_path = tmp_path / "test_dataset.h5"
        write_hdf5(episodes, output_path)

        assert output_path.exists()
        assert validate_hdf5(output_path)

        # Verify contents
        with h5py.File(output_path, "r") as f:
            assert f["ep_len"][:].tolist() == [20, 15]
            assert f["ep_offset"][:].tolist() == [0, 20]
            assert f["pixels"].shape == (35, 64, 64, 3)
            assert f["action"].shape == (35, 9)

            # Check NaN at episode boundaries
            assert np.all(np.isnan(f["action"][19]))  # end of ep 0
            assert np.all(np.isnan(f["action"][34]))  # end of ep 1

            # Non-boundary actions should not be NaN
            assert not np.any(np.isnan(f["action"][0]))
            assert not np.any(np.isnan(f["action"][20]))

    def test_single_episode(self, tmp_path):
        episodes = [
            {
                "actions": np.random.randn(10, 6).astype(np.float32),
                "pixels": np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
            },
        ]
        output_path = tmp_path / "single_ep.h5"
        write_hdf5(episodes, output_path)
        assert validate_hdf5(output_path)
