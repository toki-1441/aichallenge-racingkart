"""Core inference logic for the VelocityFormer controller.

Framework-agnostic: takes a sampled trajectory (x, y) array and produces
(velocity, steering) commands via two ONNX models (one per output head).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from model.preprocess import trajectory_to_token_ids
from model.velocity_former_onnx import VelocityFormerOnnxRunner


class VelocityFormerCore:
    """Manages ONNX VelocityFormer models and produces control commands."""

    VALID_MODES = ("velocity_only", "steering_only", "both")

    def __init__(
        self,
        velocity_onnx_path: str = "",
        steering_onnx_path: str = "",
        input_size: int = 12,
        control_mode: str = "velocity_only",
        fallback_velocity: float = 0.0,
        fallback_steering: float = 0.0,
        min_velocity: float = 0.0,
        max_velocity: float = 30.0,
        min_steering: float = -0.7,
        max_steering: float = 0.7,
    ):
        if control_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown control_mode '{control_mode}'. Must be one of {self.VALID_MODES}."
            )

        self.input_size = input_size
        self.control_mode = control_mode
        self.fallback_velocity = fallback_velocity
        self.fallback_steering = fallback_steering
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_steering = min_steering
        self.max_steering = max_steering
        self.logger = logging.getLogger(__name__)

        self.velocity_runner: Optional[VelocityFormerOnnxRunner] = None
        self.steering_runner: Optional[VelocityFormerOnnxRunner] = None

        need_velocity = control_mode in ("velocity_only", "both")
        need_steering = control_mode in ("steering_only", "both")

        if need_velocity:
            if not velocity_onnx_path:
                raise ValueError("velocity_onnx_path must be set for control_mode that predicts velocity.")
            self.velocity_runner = VelocityFormerOnnxRunner(velocity_onnx_path)
            self.logger.info(f"Loaded velocity ONNX: {velocity_onnx_path}")

        if need_steering:
            if not steering_onnx_path:
                raise ValueError("steering_onnx_path must be set for control_mode that predicts steering.")
            self.steering_runner = VelocityFormerOnnxRunner(steering_onnx_path)
            self.logger.info(f"Loaded steering ONNX: {steering_onnx_path}")

    def process(self, sampled_traj_xy: np.ndarray) -> Tuple[float, float]:
        """Run inference on a sampled trajectory.

        Args:
            sampled_traj_xy: (input_size, 2) float array of (x, y) trajectory points.

        Returns:
            Tuple of (velocity [m/s], steering [rad]).
        """
        if sampled_traj_xy.shape != (self.input_size, 2):
            raise ValueError(
                f"Expected sampled trajectory shape ({self.input_size}, 2), got {sampled_traj_xy.shape}."
            )

        token_ids = trajectory_to_token_ids(sampled_traj_xy)
        token_ids_batched = token_ids[None, :]  # (1, input_size)

        if self.velocity_runner is not None:
            velocity = float(self.velocity_runner.run(token_ids_batched).reshape(-1)[0])
        else:
            velocity = self.fallback_velocity

        if self.steering_runner is not None:
            steering = float(self.steering_runner.run(token_ids_batched).reshape(-1)[0])
        else:
            steering = self.fallback_steering

        velocity = float(np.clip(velocity, self.min_velocity, self.max_velocity))
        steering = float(np.clip(steering, self.min_steering, self.max_steering))
        return velocity, steering
