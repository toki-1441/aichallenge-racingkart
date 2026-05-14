#!/usr/bin/env python3
"""
Extract (bev, traj_gt, mode_id) samples from rosbag2 — scaffold.

Expected topics (align with running stack):
  - bev_scene_stack/tensor (std_msgs/Float32MultiArray)
  - /localization/kinematic_state (nav_msgs/Odometry) for ego pose history
  - /planning/scenario_planning/trajectory (autoware_auto_planning_msgs/Trajectory) as teacher, OR
    logged MPC reference to be defined.

Implementation note: use `ros2 bag` + rclpy deserialization or offline mcap reader
(`mcap` + `rosbags`). This file is a placeholder so CI / repo structure stays complete.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", type=str, required=True, help="Path to .mcap or rosbag2 dir")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for .npz")
    p.add_argument("--horizon", type=int, default=40)
    args = p.parse_args()
    raise NotImplementedError(
        f"rosbag extraction not implemented yet. bag={args.bag} out={args.out_dir}. "
        "Use prepare_data.py synthetic for development, or implement mcap reader here."
    )


if __name__ == "__main__":
    main()
