#!/usr/bin/env python3
"""Independently check saved Task-1 FK against RoboTwin's SAPIEN environment.

Run with the robotwin environment's Python. Visual/collision geometry is removed
from a temporary URDF to support a headless CPU check; all kinematics are kept.
"""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    cfg = yaml.safe_load(Path(manifest["robot_config"]).read_text())
    data = np.load(args.output_dir / "shared_samples.npz")
    # Spread checks across chunks, episodes, and all 32 waypoint positions.
    flat_q = data["gt_joint_trajectory"].reshape(-1, 14)
    flat_pose = data["gt_ee_pose"].reshape(-1, 2, 4, 4)
    indices = np.random.default_rng(2026).choice(
        len(flat_q), min(256, len(flat_q)), replace=False
    )
    tree = ET.parse(manifest["urdf"])
    for link in tree.getroot().findall("link"):
        for element in list(link):
            if element.tag in ("visual", "collision"):
                link.remove(element)
    with tempfile.TemporaryDirectory(prefix="task1_fk_") as tmp:
        stripped = Path(tmp) / "kinematic.urdf"
        tree.write(stripped)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(str(stripped))
        base = cfg["robot_pose"][0]
        quat = np.array(base[3:])
        quat /= np.linalg.norm(quat)
        robot.set_root_pose(sapien.Pose(base[:3], quat))
        active = {j.name: i for i, j in enumerate(robot.get_active_joints())}
        ees = [robot.find_joint_by_name(n) for n in cfg["ee_joints"]]
        worst_p, worst_r = 0.0, 0.0
        for index in indices:
            q = np.zeros(robot.dof, dtype=np.float32)
            for arm in range(2):
                for k, name in enumerate(cfg["arm_joints_name"][arm]):
                    q[active[name]] = flat_q[index, arm * 7 + k]
            robot.set_qpos(q)
            for arm, ee in enumerate(ees):
                pose = ee.global_pose
                rot = (
                    t3d.quaternions.quat2mat(pose.q)
                    @ np.array(cfg["global_trans_matrix"])
                    @ np.array(cfg["delta_matrix"])
                )
                pos = pose.p + rot @ np.array([cfg["gripper_bias"] - 0.12, 0, 0])
                expected = flat_pose[index, arm]
                worst_p = max(worst_p, float(np.linalg.norm(pos - expected[:3, 3])))
                worst_r = max(worst_r, float(np.max(np.abs(rot - expected[:3, :3]))))
    result = {
        "backend": "SAPIEN PhysxCpuSystem; original URDF kinematics",
        "waypoints_checked": len(indices),
        "arms_checked": len(indices) * 2,
        "max_position_discrepancy_m": worst_p,
        "max_rotation_matrix_element_discrepancy": worst_r,
        "position_tolerance_m": 1e-5,
        "rotation_matrix_tolerance": 1e-5,
        "passed": worst_p < 1e-5 and worst_r < 1e-5,
    }
    (args.output_dir / "fk_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("FK does not match RoboTwin's SAPIEN end-effector poses")


if __name__ == "__main__":
    main()
