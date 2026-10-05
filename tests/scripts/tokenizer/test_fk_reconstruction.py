import importlib.util
from pathlib import Path

import numpy as np

path = (
    Path(__file__).resolve().parents[3]
    / "scripts/tokenizer/evaluate_fk_reconstruction.py"
)
spec = importlib.util.spec_from_file_location("fk_evaluation", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_rotation_geodesic_identity_quarter_half_turn():
    angles = np.array([0, np.pi / 2, np.pi])
    rot = mod.rotation([0, 0, 1], angles)
    np.testing.assert_allclose(
        mod.geodesic_deg(np.eye(3), rot), [0, 90, 180], atol=1e-10
    )


def test_absolute_targets_ignore_q0_but_deltas_share_initial_state():
    a = np.ones((2, 32, 14)) * 0.1
    q0 = np.full((2, 14), 2.0)
    np.testing.assert_array_equal(mod.to_joint_trajectory(a, q0), a)
    delta = mod.to_joint_trajectory(a, q0, "delta")
    np.testing.assert_allclose(delta[:, -1, 0], 5.2)
    np.testing.assert_allclose(delta[:, 0, 7], 2.1)
    np.testing.assert_allclose(delta[:, :, [6, 13]], 0.1)


def test_shared_translation_cancels_relative_error_and_grippers_stay_separate():
    poses = np.broadcast_to(np.eye(4), (2, 32, 2, 4, 4)).copy()
    pred_pose = poses.copy()
    pred_pose[..., 0, 3] += 0.01
    a = np.zeros((2, 32, 14))
    b = a.copy()
    b[..., 6] = 0.2
    m = mod.chunk_metrics(a, b, poses, pred_pose)
    np.testing.assert_allclose(m["left_position_cm"], 1)
    np.testing.assert_allclose(m["right_position_cm"], 1)
    np.testing.assert_allclose(m["relative_position_cm"], 0)
    np.testing.assert_allclose(m["left_gripper_mae"], 0.2)
    np.testing.assert_allclose(m["right_gripper_mae"], 0)
    np.testing.assert_allclose(m["action_mse"], 0.04 / 14)


def test_relative_rotation_is_left_transpose_right():
    gt = np.broadcast_to(np.eye(4), (2, 32, 2, 4, 4)).copy()
    gt[..., 0, :3, :3] = mod.rotation([1, 0, 0], 0.6)
    gt[..., 1, :3, :3] = mod.rotation([0, 1, 0], 0.4)
    pred = gt.copy()
    shared = mod.rotation([0, 0, 1], 0.8)
    pred[..., :3, :3] = shared @ gt[..., :3, :3]
    a = np.zeros((2, 32, 14))
    m = mod.chunk_metrics(a, a, gt, pred)
    np.testing.assert_allclose(m["relative_rotation_deg"], 0, atol=1e-12)
    np.testing.assert_allclose(m["left_rotation_deg"], np.rad2deg(0.8))


def test_statistics_keep_chunk_weighting_and_cluster_by_episode():
    result = mod.summarize({"x": np.array([0.0, 0.0, 9.0])}, np.array([1, 1, 2]))
    assert result["x"]["mean"] == 3.0
    assert result["x"]["ci95_half_width"] > 0
    constant = mod.summarize({"x": np.ones(3)}, np.array([1, 1, 2]))
    assert constant["x"]["ci95_half_width"] == 0
