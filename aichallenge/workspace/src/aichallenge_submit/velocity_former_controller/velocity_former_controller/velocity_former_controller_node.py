#!/usr/bin/env python3
"""ROS 2 node that runs VelocityFormer inference and publishes control commands."""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from autoware_auto_planning_msgs.msg import Trajectory
from autoware_auto_control_msgs.msg import AckermannControlCommand

from velocity_former_controller_core import VelocityFormerCore
from model.preprocess import sample_trajectory_points


class VelocityFormerNode(Node):
    """Subscribes to a Trajectory and publishes AckermannControlCommand inferred by VelocityFormer."""

    def __init__(self):
        super().__init__("velocity_former_controller_node")

        self.declare_parameter("log_interval_sec", 5.0)
        self.declare_parameter("model.velocity_onnx_path", "")
        self.declare_parameter("model.steering_onnx_path", "")
        self.declare_parameter("model.input_size", 12)
        self.declare_parameter("trajectory.interval", 10)
        self.declare_parameter("trajectory.minimum_num", 150)
        self.declare_parameter("control_mode.mode", "velocity_only")
        self.declare_parameter("output.fallback_steering", 0.0)
        self.declare_parameter("output.fallback_velocity", 0.0)
        self.declare_parameter("output.max_velocity", 30.0)
        self.declare_parameter("output.min_velocity", 0.0)
        self.declare_parameter("output.max_steering", 0.7)
        self.declare_parameter("output.min_steering", -0.7)
        self.declare_parameter("debug", False)

        velocity_onnx_path = self.get_parameter("model.velocity_onnx_path").value
        steering_onnx_path = self.get_parameter("model.steering_onnx_path").value
        self.input_size = int(self.get_parameter("model.input_size").value)
        self.interval = int(self.get_parameter("trajectory.interval").value)
        self.minimum_num = int(self.get_parameter("trajectory.minimum_num").value)
        control_mode = self.get_parameter("control_mode.mode").value

        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval = float(self.get_parameter("log_interval_sec").value)

        try:
            self.core = VelocityFormerCore(
                velocity_onnx_path=velocity_onnx_path,
                steering_onnx_path=steering_onnx_path,
                input_size=self.input_size,
                control_mode=control_mode,
                fallback_velocity=float(self.get_parameter("output.fallback_velocity").value),
                fallback_steering=float(self.get_parameter("output.fallback_steering").value),
                min_velocity=float(self.get_parameter("output.min_velocity").value),
                max_velocity=float(self.get_parameter("output.max_velocity").value),
                min_steering=float(self.get_parameter("output.min_steering").value),
                max_steering=float(self.get_parameter("output.max_steering").value),
            )
            self.get_logger().info(
                f"Core initialized. mode={control_mode}, input_size={self.input_size}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise

        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_traj = self.create_subscription(
            Trajectory,
            "/planning/scenario_planning/trajectory",
            self.trajectory_callback,
            qos,
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("VelocityFormerNode is ready.")

    def trajectory_callback(self, msg: Trajectory):
        if len(msg.points) < self.minimum_num:
            if self.debug:
                self.get_logger().warning(
                    f"Skipping: trajectory too short ({len(msg.points)} < {self.minimum_num})"
                )
            return

        start_time = time.monotonic()

        traj_xy = np.empty((len(msg.points), 2), dtype=np.float32)
        for i, p in enumerate(msg.points):
            traj_xy[i, 0] = p.pose.position.x
            traj_xy[i, 1] = p.pose.position.y

        sampled = sample_trajectory_points(traj_xy, self.input_size, self.interval)
        velocity, steering = self.core.process(sampled)

        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.stamp = cmd.stamp
        cmd.longitudinal.speed = float(velocity)
        cmd.longitudinal.acceleration = 0.0
        cmd.lateral.stamp = cmd.stamp
        cmd.lateral.steering_tire_angle = float(steering)
        self.pub_control.publish(cmd)

        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = float(np.mean(self.inference_times))
                max_time = float(np.max(self.inference_times))
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0
                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | Max: {max_time:.2f}ms"
                )
                self.inference_times.clear()
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = VelocityFormerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
