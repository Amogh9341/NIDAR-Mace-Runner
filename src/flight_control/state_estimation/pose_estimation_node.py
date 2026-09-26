#!/usr/bin/env python3
"""
pose_estimation_node.py

Fuses IMU (for high-rate motion prediction between scans) with scan-to-map
matching against Mapper's occupancy grid (for drift correction) into a
pose estimate, published to PX4 in the EXACT VehicleOdometry format your
spec doc requires, and also republished as PoseStamped on /pose for
mapper_node/planner_node (closing the diagram's Mapper<->Pose loop).

Structure (standard SLAM front-end pattern):
    PREDICT (high rate, every IMU message): integrate gyro yaw rate to
        keep orientation current between scans. Position prediction uses
        the last scan-match-derived velocity estimate — this is
        deliberately coarse; it only has to hold until the next
        correction, not be accurate on its own.
    CORRECT (every new /scan + current /map): search a small window of
        (dx, dy, dtheta) around the predicted pose and score each
        candidate by how well the scan's points land on OCCUPIED cells
        in the current occupancy grid. Best-scoring candidate becomes the
        corrected pose. This is a brute-force correlative scan matcher —
        the same idea hector_slam uses, just without its Gauss-Newton
        optimization for speed. Fine as a first working version; the
        known upgrade path if this is too slow on the Jetson/RPi is
        replacing the brute-force search with a gradient-based matcher.

Inputs:
    /scan  (sensor_msgs/LaserScan)      — from sllidar_ros2
    /map   (nav_msgs/OccupancyGrid)     — from mapper_node
    /fmu/out/sensor_combined (px4_msgs/SensorCombined) — IMU. NOTE: your
        FC->RPi table doesn't list a raw IMU topic by name; this is PX4's
        standard raw-IMU output. Flag if your doc specifies a different one.
    /fmu/out/estimator_status_flags (px4_msgs/EstimatorStatusFlags) —
        cross-check only, used to widen the correction search window when
        PX4's own fusion confidence is low (see error_correct_gain below)
    /fmu/out/estimator_innovations (px4_msgs/EstimatorInnovations) —
        cross-check only, same purpose
    /fmu/out/vehicle_local_position, /fmu/out/vehicle_odometry —
        cross-check reference only, not used in the core estimate

Outputs:
    /fmu/in/vehicle_visual_odometry (px4_msgs/VehicleOdometry) — exact
        field set per spec, streamed at 30-50Hz
    /pose (geometry_msgs/PoseStamped) — feeds mapper_node + planner_node

Requires: rclpy, sensor_msgs, nav_msgs, geometry_msgs, px4_msgs, numpy.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from px4_msgs.msg import (
    SensorCombined,
    EstimatorStatusFlags,
    EstimatorInnovations,
    VehicleLocalPosition,
    VehicleOdometry,
)

OCCUPIED_THRESHOLD = 50

# uXRCE-DDS QoS — confirmed necessary for RPi-side sub/pub to actually see
# FC messages, per your spec doc.
FC_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def enu_to_ned(x_enu, y_enu, z_enu, yaw_enu):
    """Inverse of the ned_to_enu conversion used elsewhere in this
    project — our internal working frame (shared with mapper_node) is
    ENU; PX4's VehicleOdometry wants NED (pose_frame = 1)."""
    x_ned = y_enu
    y_ned = x_enu
    z_ned = -z_enu
    yaw_ned = (math.pi / 2.0) - yaw_enu
    yaw_ned = math.atan2(math.sin(yaw_ned), math.cos(yaw_ned))
    return x_ned, y_ned, z_ned, yaw_ned


def yaw_to_quaternion_wxyz(yaw):
    """Hamiltonian convention, order (w, x, y, z) — per spec."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        self.declare_parameter("scan_window_dx_dy_m", 0.3)
        self.declare_parameter("scan_window_dtheta_rad", 0.2)
        self.declare_parameter("scan_window_steps", 5)  # per-axis search steps
        self.declare_parameter("jump_threshold_m", 1.0)  # triggers reset_counter
        self.declare_parameter("publish_rate_hz", 30.0)

        self.window_xy = self.get_parameter("scan_window_dx_dy_m").value
        self.window_theta = self.get_parameter("scan_window_dtheta_rad").value
        self.window_steps = self.get_parameter("scan_window_steps").value
        self.jump_threshold = self.get_parameter("jump_threshold_m").value

        # Internal working pose, ENU meters + radians (same frame mapper_node uses).
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0  # from last scan-match delta, used for prediction
        self.last_predict_time = None
        self.reset_counter = 0
        self.last_match_quality = 0  # 0-100, set by correct_pose() from match confidence

        self.latest_scan: LaserScan | None = None
        self.latest_map: OccupancyGrid | None = None

        # Cross-check-only state (not part of the core estimate)
        self.estimator_flags = None
        self.estimator_innovations = None
        self.fc_local_position = None
        self.fc_odometry = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(LaserScan, "/scan", self.scan_callback, sensor_qos)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, sensor_qos)
        self.create_subscription(
            SensorCombined, "/fmu/out/sensor_combined", self.imu_callback, FC_QOS
        )
        self.create_subscription(
            EstimatorStatusFlags, "/fmu/out/estimator_status_flags",
            self.estimator_flags_callback, FC_QOS,
        )
        self.create_subscription(
            EstimatorInnovations, "/fmu/out/estimator_innovations",
            self.estimator_innovations_callback, FC_QOS,
        )
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self.fc_local_position_callback, FC_QOS,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry",
            self.fc_odometry_callback, FC_QOS,
        )

        self.odom_pub = self.create_publisher(
            VehicleOdometry, "/fmu/in/vehicle_visual_odometry", FC_QOS
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/pose", sensor_qos)

        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self.predict_and_publish)

        self.get_logger().info("pose_estimation_node up. Waiting for scan + map + IMU...")

    # ---- Lightweight callbacks ----

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.correct_pose()  # correction runs on new scan arrival, not on the predict timer

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def imu_callback(self, msg: SensorCombined):
        # Gyro Z (yaw rate) integrated in predict_and_publish via dt; here
        # we just store it. gyro_rad is body-frame [x, y, z] rad/s.
        self.last_gyro_z = msg.gyro_rad[2]

    def estimator_flags_callback(self, msg: EstimatorStatusFlags):
        self.estimator_flags = msg

    def estimator_innovations_callback(self, msg: EstimatorInnovations):
        self.estimator_innovations = msg

    def fc_local_position_callback(self, msg: VehicleLocalPosition):
        self.fc_local_position = msg

    def fc_odometry_callback(self, msg: VehicleOdometry):
        self.fc_odometry = msg

    # ---- Predict step: IMU-driven, runs at publish_rate_hz ----

    def predict_and_publish(self):
        now = self.get_clock().now()
        if self.last_predict_time is not None:
            dt = (now - self.last_predict_time).nanoseconds / 1e9
            gyro_z = getattr(self, "last_gyro_z", 0.0)
            self.yaw += gyro_z * dt
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
            self.x += self.vx * dt
            self.y += self.vy * dt
        self.last_predict_time = now

        self.publish_odometry()
        self.publish_pose()

    # ---- Correct step: scan-to-map matching, runs on new scan ----

    def correct_pose(self):
        if self.latest_map is None or self.latest_scan is None:
            return

        grid_msg = self.latest_map
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        width, height = grid_msg.info.width, grid_msg.info.height
        grid = np.array(grid_msg.data, dtype=np.int8).reshape((height, width))

        scan = self.latest_scan
        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                points.append((r, angle))
            angle += scan.angle_increment
        if not points:
            return

        # Widen the search window when PX4's own estimator confidence is
        # low (per your doc: Pose Estimation should "error correct
        # aggressively" when estimator_status_flags/innovations indicate
        # a problem).
        gain = 1.0
        if self.estimator_flags is not None and not self._fusion_looks_healthy():
            gain = 2.0

        best_score = -1
        best_pose = (self.x, self.y, self.yaw)

        steps = self.window_steps
        for i in range(-steps, steps + 1):
            dx = (i / steps) * self.window_xy * gain
            for j in range(-steps, steps + 1):
                dy = (j / steps) * self.window_xy * gain
                for k in range(-steps, steps + 1):
                    dtheta = (k / steps) * self.window_theta * gain
                    cand_x = self.x + dx
                    cand_y = self.y + dy
                    cand_yaw = self.yaw + dtheta

                    score = self._score_pose(cand_x, cand_y, cand_yaw, points,
                                              grid, ox, oy, res, width, height)
                    if score > best_score:
                        best_score = score
                        best_pose = (cand_x, cand_y, cand_yaw)

        new_x, new_y, new_yaw = best_pose
        jump = math.hypot(new_x - self.x, new_y - self.y)
        if jump > self.jump_threshold:
            self.reset_counter = (self.reset_counter + 1) % 256
            self.get_logger().warn(
                f"Pose jump of {jump:.2f}m detected — incrementing reset_counter "
                f"to {self.reset_counter} so EKF2 doesn't treat this as real motion"
            )

        # Match confidence: fraction of scan points that landed on an
        # occupied cell at the winning pose, scaled to 0-100. Low quality
        # means either the scan is genuinely ambiguous (featureless area)
        # or the pose has drifted far enough that matching is unreliable —
        # both are useful for EKF2/downstream consumers to know about.
        self.last_match_quality = int(100 * best_score / len(points)) if points else 0

        # Velocity estimate for the next predict step, derived from this
        # correction's displacement over the time since the last one.
        if self.last_predict_time is not None:
            dt = max(1e-3, 1.0 / self.get_parameter("publish_rate_hz").value)
            self.vx = (new_x - self.x) / dt
            self.vy = (new_y - self.y) / dt

        self.x, self.y, self.yaw = new_x, new_y, new_yaw

    def _fusion_looks_healthy(self):
        # Placeholder check against whatever boolean fields
        # EstimatorStatusFlags actually exposes for "fusion active" —
        # adjust field name(s) to match px4_msgs once confirmed.
        flags = self.estimator_flags
        return getattr(flags, "cs_ev_pos_fault", False) is False

    def _score_pose(self, x, y, yaw, points, grid, ox, oy, res, width, height):
        score = 0
        for r, angle in points:
            world_angle = yaw + angle
            hit_x = x + r * math.cos(world_angle)
            hit_y = y + r * math.sin(world_angle)
            gx = int((hit_x - ox) / res)
            gy = int((hit_y - oy) / res)
            if 0 <= gx < width and 0 <= gy < height:
                if grid[gy, gx] >= OCCUPIED_THRESHOLD:
                    score += 1
        return score

    # ---- Publishing ----

    def publish_odometry(self):
        x_ned, y_ned, z_ned, yaw_ned = enu_to_ned(self.x, self.y, 0.0, self.yaw)
        w, qx, qy, qz = yaw_to_quaternion_wxyz(yaw_ned)

        msg = VehicleOdometry()
        now_us = self.get_clock().now().nanoseconds // 1000
        msg.timestamp = now_us
        # timestamp_sample: capture time of the scan this correction used,
        # not arrival time — matters for EKF2_EV_DELAY tuning per your doc.
        if self.latest_scan is not None:
            stamp = self.latest_scan.header.stamp
            msg.timestamp_sample = int(stamp.sec * 1_000_000 + stamp.nanosec / 1000)
        else:
            msg.timestamp_sample = now_us

        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED  # 1
        msg.position = [x_ned, y_ned, z_ned]
        msg.q = [w, qx, qy, qz]  # Hamiltonian, (w, x, y, z)

        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED  # 1
        vx_ned, vy_ned, _, _ = enu_to_ned(self.vx, self.vy, 0.0, 0.0)
        msg.velocity = [vx_ned, vy_ned, float("nan")]
        msg.angular_velocity = [
            float("nan"), float("nan"), getattr(self, "last_gyro_z", float("nan"))
        ]

        msg.position_variance = [0.05, 0.05, 0.1]
        msg.orientation_variance = [0.02, 0.02, 0.05]
        msg.velocity_variance = [0.1, 0.1, 0.1]

        msg.reset_counter = self.reset_counter
        # Quality tied to match confidence (fraction of scan points that
        # landed on occupied cells at the winning pose) rather than a
        # constant — inspired by ahmedeltaher/Autonomous-drone-navigation's
        # vision_pose_estimator, which ties its EKF2-facing output quality
        # to actual estimator confidence instead of a fixed value.
        msg.quality = self.last_match_quality

        self.odom_pub.publish(msg)

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
