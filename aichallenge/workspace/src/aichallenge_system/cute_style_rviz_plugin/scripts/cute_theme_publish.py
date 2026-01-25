#!/usr/bin/env python3
import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ThemePublisher(Node):
    def __init__(self, topic: str, qss: str):
        super().__init__("cute_theme_publisher")
        self.pub = self.create_publisher(String, topic, 1)
        self.qss = qss
        self.topic = topic

    def publish_once(self):
        msg = String()
        msg.data = self.qss
        self.pub.publish(msg)
        self.get_logger().info(f"Published QSS to {self.topic} ({len(self.qss)} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cute_style_rviz_plugin/theme_qss")
    ap.add_argument("--qss", help="QSS string to publish")
    ap.add_argument("--qss-file", help="Path to .qss file to publish")
    args = ap.parse_args()

    if not args.qss and not args.qss_file:
        ap.error("Specify --qss or --qss-file")

    qss = args.qss if args.qss is not None else Path(args.qss_file).read_text(encoding="utf-8")

    rclpy.init()
    node = ThemePublisher(args.topic, qss)
    node.publish_once()
    rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
