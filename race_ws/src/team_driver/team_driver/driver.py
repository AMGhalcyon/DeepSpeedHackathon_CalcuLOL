#!/usr/bin/env python3
"""ROS wiring for the team driver. The algorithm lives in `driver_core.py`.

This node does four things and nothing else: declare parameters, subscribe to
the sensors, hand each message to `DriverCore`, and publish whatever comes back
on `/drive`. Every control decision is made in `DriverCore`, so the offline
harness can drive exactly the same code the judge runs.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from .driver_core import DriverConfig, DriverCore
from .mpc_controller import MPCController
from .vehicle_model import VehicleModel

DT_REPORT_PERIOD = 5.0   # [s] how often to log dt_measured statistics


def _stamp_s(stamp):
    """A ROS time message as float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


class Driver(Node):

    def __init__(self):
        # The judge runs a bare `ros2 run team_driver driver`, so sim time is
        # set here rather than trusted to a params file.
        super().__init__('driver', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p crawl_speed:=2.0
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('crawl_speed', 1.0)        # [m/s]
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here
        self.declare_parameter('v_max', 3.0)              # [m/s] speed cap on the profile
        self.declare_parameter('lat_accel', 0.8)         # [m/s^2] cornering budget
        self.declare_parameter('a_brake', 0.8)            # [m/s^2] braking budget
        self.declare_parameter('steer_limit', 0.4)       # [rad]

        # Declare MPC parameters
        self.declare_parameter('mpc.prediction_horizon', 8)
        self.declare_parameter('mpc.control_horizon', 3)
        # Move blocking: the 8 prediction steps share 3 free moves (6 variables).
        self.declare_parameter('mpc.block_map', [0, 1, 1, 2, 2, 2, 2, 2])
        self.declare_parameter('mpc.maxiter', 25)
        self.declare_parameter('mpc.solve_budget_ms', 15.0)
        self.declare_parameter('mpc.dt', 0.025)
        self.declare_parameter('mpc.q_pos', 2.0)
        self.declare_parameter('mpc.q_yaw', 2.0)
        self.declare_parameter('mpc.q_vel', 0.5)
        self.declare_parameter('mpc.r_steer', 0.08)
        self.declare_parameter('mpc.r_accel', 0.01)
        self.declare_parameter('mpc.r_delta_steer', 0.8)
        self.declare_parameter('mpc.r_delta_accel', 0.005)
        self.declare_parameter('control_period', 0.025)   # [s] must equal mpc.dt

        self.control_period = self.get_parameter('control_period').value
        if abs(self.control_period - self.get_parameter('mpc.dt').value) > 1e-9:
            raise ValueError(
                f'control_period {self.control_period} must equal mpc.dt '
                f'{self.get_parameter("mpc.dt").value}: the model assumes each '
                'command is held for exactly one MPC step')

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        # Depth-1 best-effort: the queue never holds a stale scan, so the
        # callbacks are pure setters and the timer always reads the freshest data.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback,
            sensor_qos)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback,
            sensor_qos)

        mpc_controller = MPCController(
            vehicle_model=VehicleModel(),
            prediction_horizon=self.get_parameter('mpc.prediction_horizon').value,
            control_horizon=self.get_parameter('mpc.control_horizon').value,
            dt=self.get_parameter('mpc.dt').value,
            q_pos=self.get_parameter('mpc.q_pos').value,
            q_yaw=self.get_parameter('mpc.q_yaw').value,
            q_vel=self.get_parameter('mpc.q_vel').value,
            r_steer=self.get_parameter('mpc.r_steer').value,
            r_accel=self.get_parameter('mpc.r_accel').value,
            r_delta_steer=self.get_parameter('mpc.r_delta_steer').value,
            r_delta_accel=self.get_parameter('mpc.r_delta_accel').value,
            block_map=list(self.get_parameter('mpc.block_map').value),
            maxiter=self.get_parameter('mpc.maxiter').value,
            solve_budget_ms=self.get_parameter('mpc.solve_budget_ms').value,
        )

        self.core = DriverCore(
            cfg=DriverConfig(
                crawl_speed=self.get_parameter('crawl_speed').value,
                max_range=self.get_parameter('max_range').value,
                v_max=self.get_parameter('v_max').value,
                lat_accel=self.get_parameter('lat_accel').value,
                a_brake=self.get_parameter('a_brake').value,
                steer_limit=self.get_parameter('steer_limit').value,
            ),
            mpc_controller=mpc_controller,
            logger=self.get_logger(),
        )

        self._marker_divisor = 0
        self._last_dt_report = self.get_clock().now()

        # A regular hold period is what mpc.dt assumes. With use_sim_time this
        # is 25 ms of plant time; on the wall clock it would drift with RTF.
        self.create_timer(self.control_period, self._control_step)
        self.get_logger().info(
            f'team_driver is up with MPC controller! control_period='
            f'{self.control_period * 1e3:.1f} ms')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # The bridge publishes (v cos beta, v sin beta, yaw_rate), so the
        # sideslip angle and the yaw rate come free alongside the speed.
        lx = msg.twist.twist.linear.x
        ly = msg.twist.twist.linear.y
        self.core.set_odom(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            math.hypot(lx, ly),
            beta=math.atan2(ly, lx),
            yaw_rate=msg.twist.twist.angular.z,
            t_sim=_stamp_s(msg.header.stamp),
        )

    # ------------------------------------------------------------------
    # The control loop, on a fixed timer at mpc.dt. Callbacks only store.
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        self.core.set_scan(scan.ranges, scan.angle_min, scan.angle_increment,
                           t_sim=_stamp_s(scan.header.stamp))

    def _control_step(self):
        t = self.core.t_scan
        cmd = self.core.step(t_sim=t)
        self.publish(cmd)
        self.core.note_published(cmd, t_sim=t)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(cmd.steering_angle)
        self._report_dt()

    def _report_dt(self):
        now = self.get_clock().now()
        if (now - self._last_dt_report).nanoseconds * 1e-9 < DT_REPORT_PERIOD:
            return
        self._last_dt_report = now
        mean, p95, worst = self.core.dt_summary()
        self.get_logger().info(
            f'dt_measured mean={mean * 1e3:.1f} p95={p95 * 1e3:.1f} '
            f'max={worst * 1e3:.1f} ms')

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, cmd):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(cmd.steering_angle)
        msg.drive.speed = float(cmd.speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
