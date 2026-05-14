#!/usr/bin/env python3
"""Ego-centric BEV tensor: lane OSM, trajectory, obstacles (+ V2X opponent GNSS), ego."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from autoware_auto_planning_msgs.msg import Trajectory
from geometry_msgs.msg import PointStamped, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    from tf2_geometry_msgs import do_transform_point
except ImportError:  # pragma: no cover
    do_transform_point = None  # type: ignore[misc, assignment]


def quat_to_rot_child_to_parent(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Rotation R such that v_parent = R @ v_child (column vectors)."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def apply_pose_to_map_point(pose: Pose, lx: float, ly: float, lz: float) -> Tuple[float, float, float]:
    q = pose.orientation
    r = quat_to_rot_child_to_parent(q.w, q.x, q.y, q.z)
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    p = r @ np.array([lx, ly, lz], dtype=np.float64) + t
    return float(p[0]), float(p[1]), float(p[2])


def quat_to_rot_base_to_map(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Rotation R such that v_map = R @ v_base (odometry child to map)."""
    return quat_to_rot_child_to_parent(qw, qx, qy, qz)


def map_point_to_base(odom: Odometry, x_map: float, y_map: float, z_map: float) -> Tuple[float, float, float]:
    """Transform position from map to base_link: p_base = R^T (p_map - t)."""
    q = odom.pose.pose.orientation
    r = quat_to_rot_base_to_map(q.w, q.x, q.y, q.z)
    t = np.array(
        [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ],
        dtype=np.float64,
    )
    p_map = np.array([x_map, y_map, z_map], dtype=np.float64)
    p_base = r.T @ (p_map - t)
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


def load_lane_boundary_polylines_osm(osm_path: str) -> List[List[Tuple[float, float]]]:
    """Parse Lanelet2 OSM: road lanelets' left/right member ways -> polylines in map (local_x/local_y)."""
    tree = ET.parse(osm_path)
    root = tree.getroot()
    nodes: Dict[int, Tuple[float, float]] = {}
    for el in root.findall("node"):
        nid_s = el.get("id")
        if not nid_s:
            continue
        nid = int(nid_s)
        lx = ly = None
        for tg in el.findall("tag"):
            if tg.get("k") == "local_x":
                lx = float(tg.get("v", "nan"))
            elif tg.get("k") == "local_y":
                ly = float(tg.get("v", "nan"))
        if lx is not None and ly is not None and math.isfinite(lx) and math.isfinite(ly):
            nodes[nid] = (lx, ly)

    ways: Dict[int, List[int]] = {}
    for el in root.findall("way"):
        wid_s = el.get("id")
        if not wid_s:
            continue
        wid = int(wid_s)
        is_parking = False
        nds: List[int] = []
        for child in el:
            if child.tag == "nd":
                ref = child.get("ref")
                if ref:
                    nds.append(int(ref))
            elif child.tag == "tag" and child.get("k") == "type" and child.get("v") == "parking_space":
                is_parking = True
        if not is_parking and len(nds) >= 2:
            ways[wid] = nds

    polylines: List[List[Tuple[float, float]]] = []
    used_ways: set[int] = set()

    for rel in root.findall("relation"):
        rtype = next((t.get("v") for t in rel.findall("tag") if t.get("k") == "type"), None)
        if rtype != "lanelet":
            continue
        subtype = next((t.get("v") for t in rel.findall("tag") if t.get("k") == "subtype"), "")
        if subtype in ("parking", "crosswalk", "walkway"):
            continue
        for m in rel.findall("member"):
            if m.get("type") != "way":
                continue
            role = m.get("role") or ""
            if role not in ("left", "right", "centerline"):
                continue
            ref_s = m.get("ref")
            if not ref_s:
                continue
            wid = int(ref_s)
            if wid in used_ways or wid not in ways:
                continue
            pts: List[Tuple[float, float]] = []
            for nid in ways[wid]:
                if nid in nodes:
                    pts.append(nodes[nid])
            if len(pts) >= 2:
                polylines.append(pts)
                used_ways.add(wid)
    return polylines


def _as_str_list(val: object) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val else []
    try:
        return [str(x) for x in list(val)]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return [str(val)]


class BevSceneStackNode(Node):
    # Channel indices (Float32MultiArray layout)
    CH_LANE = 0
    CH_TRAJ = 1
    CH_OBS = 2
    CH_EGO = 3
    NUM_CHANNELS = 4

    def __init__(self) -> None:
        super().__init__("bev_scene_stack")

        self.declare_parameter("odom_topic", "/localization/kinematic_state")
        self.declare_parameter("trajectory_topic", "/planning/scenario_planning/trajectory")
        self.declare_parameter("objects_topic", "/aichallenge/objects")
        self.declare_parameter("use_v2x_opponent_poses", True)
        self.declare_parameter(
            "v2x_opponent_pose_topics",
            ["/v2x/gnss/pose_with_covariance"],
        )
        self.declare_parameter("v2x_pose_map_frame", "map")
        # If > 0, skip V2X poses whose map position is within this distance of ego (own echo).
        self.declare_parameter("v2x_skip_if_near_ego_m", 0.0)
        self.declare_parameter("use_lanelet_osm_lane_lines", True)
        self.declare_parameter("lanelet_osm_path", "")
        self.declare_parameter("vector_map_marker_topic", "/map/vector_map_marker")
        self.declare_parameter("use_vector_map_markers", False)
        # "*" alone = all namespaces (except excludes). Lanelet viz often sends incremental MarkerArrays.
        self.declare_parameter(
            "vector_map_marker_namespace_substrings",
            ["*"],
        )
        self.declare_parameter(
            "vector_map_marker_namespace_exclude_substrings",
            ["parking", "lanelet_id", "arrow"],
        )
        self.declare_parameter("lane_raster_half_width", 0.12)

        self.declare_parameter("tensor_topic", "bev_scene_stack/tensor")
        self.declare_parameter("debug_image", False)
        self.declare_parameter("debug_image_topic", "bev_scene_stack/debug_image")

        self.declare_parameter("grid_x_min", -12.0)
        self.declare_parameter("grid_x_max", 52.0)
        self.declare_parameter("grid_y_min", -18.0)
        self.declare_parameter("grid_y_max", 18.0)
        self.declare_parameter("resolution", 0.25)
        self.declare_parameter("path_half_width", 0.45)
        self.declare_parameter("publish_rate_hz", 15.0)
        # Axis-aligned ego footprint in base_link (x forward, y left). Defaults match
        # racing_kart_description/config/vehicle_info.param.yaml (rear axle at base_link).
        self.declare_parameter("ego_bbox_rear_x_m", -0.51)
        self.declare_parameter("ego_bbox_front_x_m", 1.554)
        self.declare_parameter("ego_bbox_right_y_m", -0.65)
        self.declare_parameter("ego_bbox_left_y_m", 0.65)

        odom_topic = str(self.get_parameter("odom_topic").value)
        traj_topic = str(self.get_parameter("trajectory_topic").value)
        objects_topic = str(self.get_parameter("objects_topic").value)
        self._use_v2x = bool(self.get_parameter("use_v2x_opponent_poses").value)
        self._v2x_topics = _as_str_list(self.get_parameter("v2x_opponent_pose_topics").value)
        if self._use_v2x and not self._v2x_topics:
            self._v2x_topics = ["/v2x/gnss/pose_with_covariance"]
        self._v2x_map_frame = str(self.get_parameter("v2x_pose_map_frame").value).strip() or "map"
        self._v2x_skip_near = float(self.get_parameter("v2x_skip_if_near_ego_m").value)
        vm_topic = str(self.get_parameter("vector_map_marker_topic").value)
        self._use_osm = bool(self.get_parameter("use_lanelet_osm_lane_lines").value)
        self._lane_osm_path_param = str(self.get_parameter("lanelet_osm_path").value).strip()
        self._use_vm = bool(self.get_parameter("use_vector_map_markers").value)
        self._vm_ns_inc = _as_str_list(self.get_parameter("vector_map_marker_namespace_substrings").value)
        self._vm_ns_exc = _as_str_list(self.get_parameter("vector_map_marker_namespace_exclude_substrings").value)
        self._lane_hw = float(self.get_parameter("lane_raster_half_width").value)

        tensor_topic = str(self.get_parameter("tensor_topic").value)
        self._debug_image = bool(self.get_parameter("debug_image").value)
        dbg_topic = str(self.get_parameter("debug_image_topic").value)

        self._x_min = float(self.get_parameter("grid_x_min").value)
        self._x_max = float(self.get_parameter("grid_x_max").value)
        self._y_min = float(self.get_parameter("grid_y_min").value)
        self._y_max = float(self.get_parameter("grid_y_max").value)
        self._res = float(self.get_parameter("resolution").value)
        self._path_hw = float(self.get_parameter("path_half_width").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        rx = float(self.get_parameter("ego_bbox_rear_x_m").value)
        fx = float(self.get_parameter("ego_bbox_front_x_m").value)
        ry = float(self.get_parameter("ego_bbox_right_y_m").value)
        ly = float(self.get_parameter("ego_bbox_left_y_m").value)
        self._ego_x_min, self._ego_x_max = (rx, fx) if rx <= fx else (fx, rx)
        self._ego_y_min, self._ego_y_max = (ry, ly) if ry <= ly else (ly, ry)

        self._h = max(1, int(math.ceil((self._x_max - self._x_min) / self._res)))
        self._w = max(1, int(math.ceil((self._y_max - self._y_min) / self._res)))
        self._c = self.NUM_CHANNELS

        self._odom: Optional[Odometry] = None
        self._traj: Optional[Trajectory] = None
        self._objects: List[float] = []
        self._v2x_poses: Dict[str, PoseWithCovarianceStamped] = {}
        self._v2x_frame_warn_last_ns: int = 0
        self._marker_store: Dict[Tuple[str, int], Marker] = {}
        self._markers: List[Marker] = []
        self._marker_count_logged = False
        self._lane_polylines_map: List[List[Tuple[float, float]]] = []

        self._tf_buffer: Optional[Buffer] = None
        self._tf_listener = None
        if self._use_vm:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        self._load_lane_polylines()

        # EKF odom: RELIABLE. Trajectory from simple_trajectory_generator: BEST_EFFORT (must match or no data).
        qos_odom = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
        qos_traj = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        qos_objects = rclpy.qos.QoSProfile(
            depth=5,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        qos_vm = rclpy.qos.QoSProfile(
            depth=50,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        qos_v2x = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Odometry, odom_topic, self._on_odom, qos_odom)
        self.create_subscription(Trajectory, traj_topic, self._on_traj, qos_traj)
        self.create_subscription(Float64MultiArray, objects_topic, self._on_objects, qos_objects)
        if self._use_vm:
            self.create_subscription(MarkerArray, vm_topic, self._on_markers, qos_vm)
        seen_v2x: set[str] = set()
        for tp in self._v2x_topics:
            if tp in seen_v2x:
                continue
            seen_v2x.add(tp)
            self.create_subscription(
                PoseWithCovarianceStamped,
                tp,
                lambda m, topic=tp: self._on_v2x_pose(topic, m),
                qos_v2x,
            )

        self._pub_tensor = self.create_publisher(Float32MultiArray, tensor_topic, qos_odom)
        self._pub_image: Optional[object] = None
        if self._debug_image:
            self._pub_image = self.create_publisher(Image, dbg_topic, qos_odom)
            self.get_logger().info(f"debug_image=true: publishing RGB visualization on '{dbg_topic}'")

        period = 1.0 / max(0.1, rate_hz)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"BEV grid {self._c}x{self._h}x{self._w} "
            f"(x[{self._x_min},{self._x_max}] y[{self._y_min},{self._y_max}] res={self._res}m), "
            f"tensor topic '{tensor_topic}', "
            f"osm_lane_lines={len(self._lane_polylines_map)} polylines, "
            f"vector_map_markers={'on' if self._use_vm else 'off'}, "
            f"ego_bbox x[{self._ego_x_min:.3f},{self._ego_x_max:.3f}] y[{self._ego_y_min:.3f},{self._ego_y_max:.3f}], "
            f"v2x_opponents={'on [' + ','.join(self._v2x_topics) + ']' if self._use_v2x and self._v2x_topics else 'off'}"
        )

    def _load_lane_polylines(self) -> None:
        self._lane_polylines_map = []
        if not self._use_osm:
            return
        path = self._lane_osm_path_param
        if not path:
            try:
                path = os.path.join(
                    get_package_share_directory("aichallenge_submit_launch"),
                    "map",
                    "lanelet2_map.osm",
                )
            except LookupError:
                self.get_logger().warning(
                    "use_lanelet_osm_lane_lines=true but lanelet_osm_path empty and "
                    "aichallenge_submit_launch not found; lane channel will be empty unless markers enabled."
                )
                return
        if not os.path.isfile(path):
            self.get_logger().warning(f"Lanelet OSM not found: {path}")
            return
        try:
            self._lane_polylines_map = load_lane_boundary_polylines_osm(path)
            self.get_logger().info(
                f"Loaded {len(self._lane_polylines_map)} lane boundary polylines from {path}"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to parse Lanelet OSM {path}: {exc}")

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_traj(self, msg: Trajectory) -> None:
        self._traj = msg

    def _on_objects(self, msg: Float64MultiArray) -> None:
        self._objects = list(msg.data)

    def _on_v2x_pose(self, topic: str, msg: PoseWithCovarianceStamped) -> None:
        self._v2x_poses[topic] = msg

    def _on_markers(self, msg: MarkerArray) -> None:
        # Lanelet2MapVisualization publishes incremental updates; merge like RViz.
        for mk in msg.markers:
            key = (mk.ns, mk.id)
            if mk.action == Marker.DELETE:
                self._marker_store.pop(key, None)
            elif mk.action == Marker.DELETEALL:
                self._marker_store.clear()
            else:
                self._marker_store[key] = mk
        self._markers = list(self._marker_store.values())

    def _namespace_use_marker(self, ns: str) -> bool:
        if any(ex in ns for ex in self._vm_ns_exc):
            return False
        if not self._vm_ns_inc:
            return True
        if len(self._vm_ns_inc) == 1 and self._vm_ns_inc[0] == "*":
            return True
        return any(inc in ns for inc in self._vm_ns_inc)

    def _base_to_indices(self, x_b: float, y_b: float) -> Optional[Tuple[int, int]]:
        if not (self._x_min <= x_b <= self._x_max and self._y_min <= y_b <= self._y_max):
            return None
        row = int((self._x_max - x_b) / self._res)
        col = int((self._y_max - y_b) / self._res)
        row = max(0, min(self._h - 1, row))
        col = max(0, min(self._w - 1, col))
        return row, col

    def _raster_segment(self, grid: np.ndarray, ch: int, p0: Tuple[float, float], p1: Tuple[float, float]) -> None:
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(length / (self._res * 0.5))))
        for s in range(n + 1):
            t = s / float(n)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            ij = self._base_to_indices(x, y)
            if ij is not None:
                grid[ch, ij[0], ij[1]] = 1.0

    def _raster_path_tube(self, grid: np.ndarray, ch: int, points_b: List[Tuple[float, float]], half_width: float) -> None:
        if len(points_b) < 2:
            return
        hw = half_width
        for i in range(len(points_b) - 1):
            x0, y0 = points_b[i]
            x1, y1 = points_b[i + 1]
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy) or 1.0
            px, py = -dy / ln * hw, dx / ln * hw
            self._raster_segment(grid, ch, (x0 + px, y0 + py), (x1 + px, y1 + py))
            self._raster_segment(grid, ch, (x0 - px, y0 - py), (x1 - px, y1 - py))
            self._raster_segment(grid, ch, (x0, y0), (x1, y1))

    def _raster_obstacles(self, grid: np.ndarray, ch: int, odom: Odometry) -> None:
        data = self._objects
        for i in range(0, len(data) - 3, 4):
            x_m, y_m, z_m, rad = data[i], data[i + 1], data[i + 2], data[i + 3]
            x_b, y_b, _ = map_point_to_base(odom, x_m, y_m, z_m)
            r_pix = int(math.ceil(max(rad, self._res) / self._res)) + 1
            center = self._base_to_indices(x_b, y_b)
            if center is None:
                continue
            cr, cc = center
            for dr in range(-r_pix, r_pix + 1):
                for dc in range(-r_pix, r_pix + 1):
                    rr, ccl = cr + dr, cc + dc
                    if 0 <= rr < self._h and 0 <= ccl < self._w:
                        y_cell = self._y_max - (ccl + 0.5) * self._res
                        x_cell = self._x_max - (rr + 0.5) * self._res
                        if math.hypot(x_cell - x_b, y_cell - y_b) <= max(rad, self._res * 0.5):
                            grid[ch, rr, ccl] = 1.0

    @staticmethod
    def _point_in_polygon(px: float, py: float, poly: Sequence[Tuple[float, float]]) -> bool:
        """Ray casting; poly closed implicitly (last -> first)."""
        n = len(poly)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersects = (yi > py) != (yj > py) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _opponent_footprint_corners_base(
        self, odom: Odometry, msg: PoseWithCovarianceStamped
    ) -> Optional[List[Tuple[float, float]]]:
        """Opponent vehicle bbox in ego base_link: same half-extents as ego, oriented with V2X pose (map)."""
        fid = (msg.header.frame_id or "").strip()
        if fid and fid != self._v2x_map_frame:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._v2x_frame_warn_last_ns > 5_000_000_000:
                self.get_logger().warning(
                    f"V2X pose frame_id '{fid}' != '{self._v2x_map_frame}'; "
                    "skipping until frame matches (transform not implemented)."
                )
                self._v2x_frame_warn_last_ns = now_ns
            return None
        pose = msg.pose.pose
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        if abs(qw) + abs(qx) + abs(qy) + abs(qz) < 1e-9:
            return None
        corners_body = (
            (self._ego_x_min, self._ego_y_min, 0.0),
            (self._ego_x_min, self._ego_y_max, 0.0),
            (self._ego_x_max, self._ego_y_max, 0.0),
            (self._ego_x_max, self._ego_y_min, 0.0),
        )
        base_xy: List[Tuple[float, float]] = []
        for bx, by, bz in corners_body:
            xm, ym, zm = apply_pose_to_map_point(pose, bx, by, bz)
            xbb, ybb, _ = map_point_to_base(odom, xm, ym, zm)
            base_xy.append((xbb, ybb))
        return base_xy

    def _raster_v2x_opponents(self, grid: np.ndarray, ch: int, odom: Odometry) -> None:
        if not self._use_v2x or not self._v2x_topics:
            return
        ex = odom.pose.pose.position.x
        ey = odom.pose.pose.position.y
        poses = list(self._v2x_poses.items())
        for _topic, msg in poses:
            ox = msg.pose.pose.position.x
            oy = msg.pose.pose.position.y
            if self._v2x_skip_near > 0.0 and math.hypot(ox - ex, oy - ey) < self._v2x_skip_near:
                continue
            quad = self._opponent_footprint_corners_base(odom, msg)
            if quad is None:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            pad = self._res
            minx, maxx = min(xs) - pad, max(xs) + pad
            miny, maxy = min(ys) - pad, max(ys) + pad
            for r in range(self._h):
                x_cell = self._x_max - (r + 0.5) * self._res
                if x_cell < minx or x_cell > maxx:
                    continue
                for c in range(self._w):
                    y_cell = self._y_max - (c + 0.5) * self._res
                    if y_cell < miny or y_cell > maxy:
                        continue
                    if self._point_in_polygon(x_cell, y_cell, quad):
                        grid[ch, r, c] = 1.0

    def _raster_ego_bbox(self, grid: np.ndarray, ch: int) -> None:
        """Fill channel cells whose centers lie inside ego axis-aligned bbox in base_link."""
        xa, xb = self._ego_x_min, self._ego_x_max
        ya, yb = self._ego_y_min, self._ego_y_max
        for r in range(self._h):
            x_cell = self._x_max - (r + 0.5) * self._res
            if not (xa <= x_cell <= xb):
                continue
            for c in range(self._w):
                y_cell = self._y_max - (c + 0.5) * self._res
                if ya <= y_cell <= yb:
                    grid[ch, r, c] = 1.0

    def _marker_point_to_base(self, odom: Odometry, mk: Marker, lx: float, ly: float, lz: float) -> Optional[Tuple[float, float, float]]:
        xm, ym, zm = apply_pose_to_map_point(mk.pose, lx, ly, lz)
        src_frame = mk.header.frame_id or odom.header.frame_id
        if self._tf_buffer is None or src_frame == odom.header.frame_id:
            return map_point_to_base(odom, xm, ym, zm)
        if do_transform_point is None:
            return map_point_to_base(odom, xm, ym, zm)
        ps = PointStamped()
        ps.header.frame_id = src_frame
        ps.header.stamp = rclpy.time.Time().to_msg()
        ps.point.x = xm
        ps.point.y = ym
        ps.point.z = zm
        try:
            tf = self._tf_buffer.lookup_transform(
                odom.child_frame_id,
                src_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.15),
            )
            out = do_transform_point(ps, tf)
            return (float(out.point.x), float(out.point.y), float(out.point.z))
        except Exception:  # noqa: BLE001
            return None

    def _raster_lane_markers(self, grid: np.ndarray, ch: int, odom: Odometry) -> None:
        for mk in self._markers:
            if mk.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            if not self._namespace_use_marker(mk.ns):
                continue
            if mk.type == Marker.LINE_STRIP:
                pts = mk.points
                if len(pts) < 2:
                    continue
                base_pts: List[Tuple[float, float]] = []
                for p in pts:
                    bb = self._marker_point_to_base(odom, mk, p.x, p.y, p.z)
                    if bb is not None:
                        base_pts.append((bb[0], bb[1]))
                self._raster_path_tube(grid, ch, base_pts, self._lane_hw)
            elif mk.type == Marker.LINE_LIST:
                pts = mk.points
                for i in range(0, len(pts) - 1, 2):
                    p0 = pts[i]
                    p1 = pts[i + 1]
                    b0 = self._marker_point_to_base(odom, mk, p0.x, p0.y, p0.z)
                    b1 = self._marker_point_to_base(odom, mk, p1.x, p1.y, p1.z)
                    if b0 is None or b1 is None:
                        continue
                    self._raster_path_tube(
                        grid,
                        ch,
                        [(b0[0], b0[1]), (b1[0], b1[1])],
                        self._lane_hw,
                    )

    def _raster_lane_osm(self, grid: np.ndarray, ch: int, odom: Odometry) -> None:
        for poly in self._lane_polylines_map:
            pts_b: List[Tuple[float, float]] = []
            for xm, ym in poly:
                xb, yb, _ = map_point_to_base(odom, xm, ym, 0.0)
                pts_b.append((xb, yb))
            self._raster_path_tube(grid, ch, pts_b, self._lane_hw)

    def _build_tensor(self, odom: Odometry) -> np.ndarray:
        """Shape (4,H,W): lane (OSM or markers), trajectory tube, obstacles (+V2X), ego bbox."""
        grid = np.zeros((self._c, self._h, self._w), dtype=np.float32)

        if self._lane_polylines_map:
            self._raster_lane_osm(grid, self.CH_LANE, odom)
        elif self._use_vm and self._markers:
            self._raster_lane_markers(grid, self.CH_LANE, odom)
            if not self._marker_count_logged and self._markers:
                self._marker_count_logged = True
                self.get_logger().info(
                    f"vector map markers merged: {len(self._markers)} markers for BEV lane channel"
                )

        traj = self._traj
        if traj is not None and len(traj.points) > 0:
            pts_b: List[Tuple[float, float]] = []
            for p in traj.points:
                xb, yb, _ = map_point_to_base(odom, p.pose.position.x, p.pose.position.y, p.pose.position.z)
                pts_b.append((xb, yb))
            self._raster_path_tube(grid, self.CH_TRAJ, pts_b, self._path_hw)

        self._raster_obstacles(grid, self.CH_OBS, odom)
        self._raster_v2x_opponents(grid, self.CH_OBS, odom)

        self._raster_ego_bbox(grid, self.CH_EGO)
        return grid

    def _tensor_to_rgb(self, tensor_chw: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8: R=lane, G=trajectory, B=obstacles (ego folded into G/B max)."""
        h, w = self._h, self._w
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        order: Sequence[Tuple[int, int]] = (
            (0, self.CH_LANE),
            (1, self.CH_TRAJ),
            (2, self.CH_OBS),
        )
        for out_ch, src_ch in order:
            ch = tensor_chw[src_ch].astype(np.float64)
            vmax = float(ch.max()) if float(ch.max()) > 1e-6 else 1.0
            rgb[:, :, out_ch] = np.clip(ch / vmax * 255.0, 0.0, 255.0).astype(np.uint8)
        ego = tensor_chw[self.CH_EGO]
        rgb[:, :, 1] = np.clip(np.maximum(rgb[:, :, 1].astype(np.float64), ego * 255.0), 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip(np.maximum(rgb[:, :, 2].astype(np.float64), ego * 255.0), 0, 255).astype(np.uint8)
        return rgb

    def _on_timer(self) -> None:
        odom = self._odom
        if odom is None:
            return

        tensor = self._build_tensor(odom)
        layout = MultiArrayLayout()
        layout.dim = [
            MultiArrayDimension(label="channel", size=self._c, stride=self._h * self._w),
            MultiArrayDimension(label="height", size=self._h, stride=self._w),
            MultiArrayDimension(label="width", size=self._w, stride=1),
        ]
        layout.data_offset = 0
        msg = Float32MultiArray()
        msg.layout = layout
        msg.data = tensor.reshape(-1, order="C").astype(np.float32).tolist()
        self._pub_tensor.publish(msg)

        if self._debug_image and self._pub_image is not None:
            rgb = self._tensor_to_rgb(tensor)
            img = Image()
            img.header.stamp = self.get_clock().now().to_msg()
            img.header.frame_id = odom.child_frame_id or "base_link"
            img.height = self._h
            img.width = self._w
            img.encoding = "rgb8"
            img.is_bigendian = 0
            img.step = self._w * 3
            img.data = rgb.tobytes()
            self._pub_image.publish(img)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = BevSceneStackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
