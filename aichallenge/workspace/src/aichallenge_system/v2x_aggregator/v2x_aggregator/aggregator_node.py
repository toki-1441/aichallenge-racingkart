"""V2X PointStamped aggregator node.

Subscribes /v2x/vehicle_position (geometry_msgs/PointStamped where
header.frame_id encodes the source vehicle as "d{N}") and republishes
v2x_msgs/V2XVehiclePositionArray on /v2x/vehicle_positions at 20 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from v2x_msgs.msg import V2XVehiclePosition, V2XVehiclePositionArray


ARRAY_HEADER_FRAME_ID = "map"
POSITION_FRAME_ID = "map"
DEFAULT_STALE_TIMEOUT_S = 1.0
DEFAULT_PUBLISH_PERIOD_S = 0.05  # 20 Hz


@dataclass
class _Entry:
    msg: PointStamped
    last_seen_s: float


class V2XAggregatorState:
    """Holds the latest PointStamped per vehicle_id and produces snapshots."""

    def __init__(self, stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S):
        self._stale_timeout_s = stale_timeout_s
        self._entries: Dict[str, _Entry] = {}

    def update(self, vehicle_id: str, msg, now_s: float) -> None:
        self._entries[vehicle_id] = _Entry(msg=msg, last_seen_s=now_s)

    def update_from_message(self, msg, now_s: float) -> None:
        self.update(msg.header.frame_id, msg, now_s)

    def snapshot(self, now_s: float) -> List[Tuple[str, object]]:
        active = []
        for vid, entry in self._entries.items():
            if now_s - entry.last_seen_s <= self._stale_timeout_s:
                active.append((vid, entry.msg))
        return active

    def drop_stale(self, now_s: float) -> None:
        cutoff = now_s - self._stale_timeout_s
        self._entries = {
            vid: e for vid, e in self._entries.items()
            if e.last_seen_s >= cutoff
        }


def build_vehicle_position_payload(vehicle_id: str, source_msg) -> dict:
    """Pure helper for unit tests; mirrors the field copy done in V2XAggregatorNode._on_timer."""
    return {
        "vehicle_id": vehicle_id,
        "position_frame_id": POSITION_FRAME_ID,
        "stamp_sec": source_msg.header.stamp.sec,
        "stamp_nanosec": source_msg.header.stamp.nanosec,
        "x": source_msg.point.x,
        "y": source_msg.point.y,
        "z": source_msg.point.z,
    }


class V2XAggregatorNode(Node):
    def __init__(self):
        super().__init__("v2x_aggregator")
        self.declare_parameter("stale_timeout_s", DEFAULT_STALE_TIMEOUT_S)
        self.declare_parameter("publish_period_s", DEFAULT_PUBLISH_PERIOD_S)

        stale = self.get_parameter("stale_timeout_s").value
        period = self.get_parameter("publish_period_s").value

        self._state = V2XAggregatorState(stale_timeout_s=stale)
        self._sub = self.create_subscription(
            PointStamped, "/v2x/vehicle_position", self._on_msg, 10)
        self._pub = self.create_publisher(
            V2XVehiclePositionArray, "/v2x/vehicle_positions", 10)
        self._timer = self.create_timer(period, self._on_timer)

    def _now_s(self) -> float:
        t = self.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9

    def _on_msg(self, msg: PointStamped) -> None:
        vehicle_id = msg.header.frame_id
        self._state.update(vehicle_id, msg, self._now_s())

    def _on_timer(self) -> None:
        now = self._now_s()
        snap = self._state.snapshot(now)

        out = V2XVehiclePositionArray()
        out.header = Header()
        out.header.frame_id = ARRAY_HEADER_FRAME_ID
        out.header.stamp = self.get_clock().now().to_msg()

        for vehicle_id, src in snap:
            elem = V2XVehiclePosition()
            elem.vehicle_id = vehicle_id
            elem.position = PointStamped()
            elem.position.header = Header()
            elem.position.header.frame_id = POSITION_FRAME_ID
            elem.position.header.stamp = src.header.stamp
            elem.position.point.x = src.point.x
            elem.position.point.y = src.point.y
            elem.position.point.z = src.point.z
            out.vehicles.append(elem)

        self._pub.publish(out)
        self._state.drop_stale(now)


def main(args=None):
    rclpy.init(args=args)
    node = V2XAggregatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
