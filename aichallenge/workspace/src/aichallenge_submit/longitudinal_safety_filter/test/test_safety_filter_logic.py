#!/usr/bin/env python3
import math

import rclpy

from longitudinal_safety_filter.longitudinal_safety_filter_node import (
    LongitudinalSafetyFilterNode,
    RiskMetrics,
)


def _make_node():
    if not rclpy.ok():
        rclpy.init()
    return LongitudinalSafetyFilterNode()


def _destroy_node(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_predicted_close_distance_does_not_trigger_emergency_stop():
    node = _make_node()
    try:
        speed = 20.0 / 3.6
        decision = node._decision_from_metrics(
            speed,
            RiskMetrics(
                front_distance_current_m=8.0,
                predicted_ttc_s=1.2,
                min_predicted_distance_m=0.5,
                predicted_collision_distance_m=speed * 1.2,
                prediction_source="selected_trajectory",
            ),
        )

        assert decision.state in ("AVOID_WITH_SLOWDOWN", "BRAKE_FOR_COMMIT")
        assert decision.state != "EMERGENCY_STOP"
    finally:
        _destroy_node(node)


def test_current_hard_stop_distance_triggers_emergency_stop():
    node = _make_node()
    try:
        decision = node._decision_from_metrics(
            20.0 / 3.6,
            RiskMetrics(
                front_distance_current_m=0.8,
                predicted_ttc_s=math.inf,
                min_predicted_distance_m=math.inf,
                predicted_collision_distance_m=math.inf,
                prediction_source="selected_trajectory",
            ),
        )

        assert decision.state == "EMERGENCY_STOP"
        assert decision.speed_limit_mps == 0.0
        assert decision.acceleration_limit_mps2 < 0.0
    finally:
        _destroy_node(node)


def test_path_clear_with_obstacle_keeps_avoid_state_without_speed_cap():
    node = _make_node()
    try:
        decision = node._decision_from_metrics(
            20.0 / 3.6,
            RiskMetrics(
                front_distance_current_m=8.0,
                predicted_ttc_s=math.inf,
                min_predicted_distance_m=3.0,
                predicted_collision_distance_m=math.inf,
                prediction_source="selected_trajectory",
            ),
        )

        assert decision.state == "AVOID"
        assert decision.risk_class == "SAFE"
        assert math.isinf(decision.speed_limit_mps)
        assert math.isinf(decision.acceleration_limit_mps2)
    finally:
        _destroy_node(node)


def test_far_ttc_starts_slowdown_before_close_range():
    node = _make_node()
    try:
        speed = 20.0 / 3.6
        decision = node._decision_from_metrics(
            speed,
            RiskMetrics(
                front_distance_current_m=18.0,
                predicted_ttc_s=3.5,
                min_predicted_distance_m=0.5,
                predicted_collision_distance_m=speed * 3.5,
                prediction_source="selected_trajectory",
            ),
        )

        assert decision.state == "AVOID_WITH_SLOWDOWN"
        assert decision.speed_limit_mps >= 5.0 / 3.6
        assert decision.acceleration_limit_mps2 < 0.0
    finally:
        _destroy_node(node)


def test_brake_for_commit_can_command_full_stop_without_race_speed_floor():
    node = _make_node()
    try:
        decision = node._decision_from_metrics(
            20.0 / 3.6,
            RiskMetrics(
                front_distance_current_m=2.0,
                predicted_ttc_s=0.8,
                min_predicted_distance_m=0.4,
                predicted_collision_distance_m=1.0,
                prediction_source="selected_trajectory",
            ),
        )

        assert decision.state == "BRAKE_FOR_COMMIT"
        assert decision.speed_limit_mps == 0.0
        assert decision.acceleration_limit_mps2 < 0.0
    finally:
        _destroy_node(node)


def test_no_odom_slow_policy_limits_speed_and_acceleration():
    node = _make_node()
    try:
        decision = node._decision_for_fail_policy("no_odom", "slow")

        assert decision.state == "AVOID_WITH_SLOWDOWN"
        assert decision.reason == "no_odom"
        assert decision.speed_limit_mps > 0.0
        assert decision.acceleration_limit_mps2 < 0.0
    finally:
        _destroy_node(node)
