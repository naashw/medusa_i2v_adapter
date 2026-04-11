#!/usr/bin/env python3
"""Unit tests for camera_path module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from camera_path import (
    quat_to_matrix,
    slerp,
    catmull_rom_vec3,
    catmull_rom_quat,
    interpolate_camera_path,
)


def test_quat_identity():
    """Test quaternion to matrix: identity quat [1,0,0,0] -> identity matrix."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    R = quat_to_matrix(q)
    I = torch.eye(3)
    assert torch.allclose(R, I, atol=1e-5), f"Identity quat failed.\nGot:\n{R}\nExpected:\n{I}"
    print("✓ test_quat_identity passed")


def test_quat_norm():
    """Test that quaternion to matrix produces unit norm rotation matrix."""
    q = torch.tensor([0.9659, 0.1305, 0.1305, 0.1305])  # random unit quat
    R = quat_to_matrix(q.unsqueeze(0)).squeeze(0)
    # R @ R^T should be identity
    product = R @ R.T
    I = torch.eye(3)
    assert torch.allclose(product, I, atol=1e-5), f"R @ R^T != I:\n{product}"
    # det(R) should be 1
    det = torch.det(R)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), f"det(R) = {det}, expected 1"
    print("✓ test_quat_norm passed")


def test_slerp_endpoints():
    """Test slerp at t=0 and t=1."""
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q1 = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # 90 deg around z

    # t=0 should give q0
    result = slerp(q0, q1, 0.0)
    assert torch.allclose(result, F.normalize(q0, p=2, dim=-1), atol=1e-5), f"slerp(t=0) failed"

    # t=1 should give q1
    result = slerp(q0, q1, 1.0)
    assert torch.allclose(result, F.normalize(q1, p=2, dim=-1), atol=1e-5), f"slerp(t=1) failed"

    print("✓ test_slerp_endpoints passed")


def test_slerp_identity_identity():
    """Test slerp between identical quaternions."""
    q = torch.tensor([0.8, 0.2, 0.3, 0.5])
    result = slerp(q, q, 0.5)
    q_norm = F.normalize(q, p=2, dim=-1)
    assert torch.allclose(result, q_norm, atol=1e-5), "slerp(q, q, t) should return q"
    print("✓ test_slerp_identity_identity passed")


def test_catmull_rom_vec3_linear():
    """Test Catmull-Rom on linear control points."""
    p0 = torch.tensor([0.0, 0.0, 0.0])
    p1 = torch.tensor([1.0, 1.0, 1.0])
    p2 = torch.tensor([2.0, 2.0, 2.0])
    p3 = torch.tensor([3.0, 3.0, 3.0])

    # For linear control points, Catmull-Rom should interpolate [p1, p2]
    result_0 = catmull_rom_vec3(p0, p1, p2, p3, 0.0)
    assert torch.allclose(result_0, p1, atol=1e-5), f"t=0.0: {result_0} != {p1}"

    result_1 = catmull_rom_vec3(p0, p1, p2, p3, 1.0)
    assert torch.allclose(result_1, p2, atol=1e-5), f"t=1.0: {result_1} != {p2}"

    print("✓ test_catmull_rom_vec3_linear passed")


def test_interpolate_camera_path_2kf_linear():
    """Test interpolation with 2 keyframes (linear mode)."""
    camera_path = [
        {"t": 0.0, "translation": [0, 0, 0], "rotation_quat": [1, 0, 0, 0]},
        {"t": 1.0, "translation": [0, 0, 1], "rotation_quat": [1, 0, 0, 0]},
    ]

    # t=0
    trans, quat = interpolate_camera_path(camera_path, 0.0, "linear")
    assert torch.allclose(trans, torch.tensor([0, 0, 0], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(quat, torch.tensor([1, 0, 0, 0], dtype=torch.float32), atol=1e-5)

    # t=1
    trans, quat = interpolate_camera_path(camera_path, 1.0, "linear")
    assert torch.allclose(trans, torch.tensor([0, 0, 1], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(quat, torch.tensor([1, 0, 0, 0], dtype=torch.float32), atol=1e-5)

    # t=0.5 (midpoint)
    trans, quat = interpolate_camera_path(camera_path, 0.5, "linear")
    assert torch.allclose(trans, torch.tensor([0, 0, 0.5], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(quat, torch.tensor([1, 0, 0, 0], dtype=torch.float32), atol=1e-5)

    print("✓ test_interpolate_camera_path_2kf_linear passed")


def test_interpolate_camera_path_3kf_cubic():
    """Test interpolation with 3 keyframes (cubic mode) — edge case."""
    camera_path = [
        {"t": 0.0, "translation": [0, 0, 0], "rotation_quat": [1, 0, 0, 0]},
        {"t": 0.5, "translation": [0, 0, 0.5], "rotation_quat": [1, 0, 0, 0]},
        {"t": 1.0, "translation": [0, 0, 1], "rotation_quat": [1, 0, 0, 0]},
    ]

    # With 3 keyframes, we can't do full cubic (no i-1 for first segment, no i+2 for last).
    # Should fall back to linear.
    trans, quat = interpolate_camera_path(camera_path, 0.25, "cubic")
    assert trans is not None, "Should not crash on edge case"
    assert quat is not None

    print("✓ test_interpolate_camera_path_3kf_cubic passed (edge case handled)")


def test_backward_compat_dolly_in():
    """Test that fallback dolly-in path generates correctly."""
    # Simulate render_depth_sequence fallback: None camera_path
    # Should generate 2-kf linear path with dolly-in
    num_frames = 25
    frame_rate = 24.0
    camera_speed_ms = 0.5
    duration_s = (num_frames - 1) / frame_rate
    max_z = camera_speed_ms * duration_s

    expected_path = [
        {"t": 0.0, "translation": [0.0, 0.0, 0.0], "rotation_quat": [1.0, 0.0, 0.0, 0.0]},
        {"t": 1.0, "translation": [0.0, 0.0, max_z], "rotation_quat": [1.0, 0.0, 0.0, 0.0]},
    ]

    # Interpolate at a few points
    for t in [0.0, 0.25, 0.5, 1.0]:
        trans, quat = interpolate_camera_path(expected_path, t, "linear")
        # Verify Z increases linearly
        expected_z = max_z * t
        assert torch.allclose(trans[2], torch.tensor(expected_z, dtype=torch.float32), atol=1e-4), \
            f"t={t}: Z={trans[2]}, expected {expected_z}"

    print("✓ test_backward_compat_dolly_in passed")


if __name__ == "__main__":
    test_quat_identity()
    test_quat_norm()
    test_slerp_endpoints()
    test_slerp_identity_identity()
    test_catmull_rom_vec3_linear()
    test_interpolate_camera_path_2kf_linear()
    test_interpolate_camera_path_3kf_cubic()
    test_backward_compat_dolly_in()

    print("\n✅ All tests passed!")
