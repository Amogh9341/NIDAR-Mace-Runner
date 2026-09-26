#!/usr/bin/env python3
"""
mapper_node.py

Combines LiDAR reading AND occupancy-grid mapping into a single node, per
the architecture diagram: LIDAR --PC--> MAPPER, with MAPPER exchanging
OG / FC VIO with the POSE node, and broadcasting OG onward to PLANNER
and RF.

I/O (matches the Mapper Node spec):
    Input  : LiDAR scan (/scan, sensor_msgs/LaserScan)
    Input  : pose estimate from the Pose node (/pose, geometry_msgs/
             PoseStamped) — this is the "FC VIO" arrow coming INTO Mapper
             in the diagram. Kept as a generic ROS2 type here rather than
             a PX4-specific one, since Mapper's job is scan+pose -> grid,
             not talking to the flight controller.
    Output : Occupancy grid (/map, nav_msgs/OccupancyGrid) — this is the
             "OG" arrow going OUT to both Planner and RF. One publish on
             /map reaches every subscriber; no special fan-out code
             needed.

Message format (nav_msgs/OccupancyGrid, per spec doc):
    header            : timestamp + frame_id ("map")
    info.map_load_time: set to current time (no persisted map load here)
    info.resolution   : meters/cell
    info.width/height : cell dimensions
    info.origin       : real-world pose of the grid's bottom-left cell
    data[]            : int8, 0-100 = occupancy probability, -1 = unknown

Mapping method: scan-to-grid raycasting (Bresenham) — same core technique
used by hector_slam / slam_toolbox-style pipelines, inlined here as a
single custom node instead of depending on an external SLAM package.
Deliberately simple (no probabilistic log-odds smoothing yet) so it's
easy to verify against real hardware before optimizing further.

Requires: rclpy, sensor_msgs, nav_msgs, geometry_msgs, numpy.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def yaw_from_quaternion(q):
    """Extract yaw (rotation about Z) from a geometry_msgs/Quaternion.
    Only Z-rotation matters for a 2D grid."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MapperNode(Node):
    def __init__(self):
        super().__init__("mapper_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("pose_topic", "/pose")
        self.declare_parameter("map_size_m", 40.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("map_frame", "map")

        self.map_frame = self.get_parameter("map_frame").value
        self.map_size_m = self.get_parameter("map_size_m").value
        self.resolution = self.get_parameter("resolution_m").value
        self.grid_size = int(self.map_size_m / self.resolution)

        self.grid = np.full((self.grid_size, self.grid_size), UNKNOWN, dtype=np.int8)
        self.origin_x = -self.map_size_m / 2.0
        self.origin_y = -self.map_size_m / 2.0

        # Lightweight callbacks: store only. Fusion + grid update happens
        # on the timer below, not inline in either callback.
        self.latest_scan: LaserScan | None = None
        self.latest_pose = None  # (x, y, yaw)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value,
            self.scan_callback, sensor_qos,
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("pose_topic").value,
            self.pose_callback, sensor_qos,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)

        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self.update_and_publish)

        self.get_logger().info(
            f"mapper_node up. grid={self.grid_size}x{self.grid_size} "
            f"({self.resolution}m/cell). Waiting for scan + pose..."
        )

    # ---- Lightweight callbacks ----

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def pose_callback(self, msg: PoseStamped):
        yaw = yaw_from_quaternion(msg.pose.orientation)
        self.latest_pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    # ---- Timer: fusion + grid update + publish ----

    def update_and_publish(self):
        if self.latest_scan is None or self.latest_pose is None:
            return

        scan = self.latest_scan
        px, py, yaw = self.latest_pose

        sensor_cell = self.world_to_grid(px, py)
        if sensor_cell is None:
            self.get_logger().warn("Pose outside map bounds — skipping this update")
            return

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                world_angle = yaw + angle
                hit_x = px + r * math.cos(world_angle)
                hit_y = py + r * math.sin(world_angle)
                hit_cell = self.world_to_grid(hit_x, hit_y)
                if hit_cell is not None:
                    self.mark_line_free(sensor_cell, hit_cell)
                    self.grid[hit_cell[1], hit_cell[0]] = OCCUPIED
            angle += scan.angle_increment

        self.publish_map()

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return (gx, gy)
        return None

    def mark_line_free(self, start, end):
        """Bresenham's line — marks cells between sensor and hit point as
        free, excluding the hit cell itself (set OCCUPIED by the caller)."""
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while (x, y) != (x1, y1):
            self.grid[y, x] = FREE
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def publish_map(self):
        msg = OccupancyGrid()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = now
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.data = self.grid.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
