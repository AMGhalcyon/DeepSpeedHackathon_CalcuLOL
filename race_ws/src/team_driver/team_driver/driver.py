#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

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
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import os

# Import our MPC controller
from .vehicle_model import VehicleModel
from .mpc_controller import MPCController


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

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

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.v_max = self.get_parameter('v_max').value
        self.lat_accel = self.get_parameter('lat_accel').value
        self.a_brake = self.get_parameter('a_brake').value
        self.steer_limit = self.get_parameter('steer_limit').value

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        # Initialize MPC controller with parameters from YAML
        self.vehicle_model = VehicleModel()
        # Declare MPC parameters
        self.declare_parameter('mpc.prediction_horizon', 8)
        self.declare_parameter('mpc.control_horizon', 8)
        self.declare_parameter('mpc.dt', 0.025)
        self.declare_parameter('mpc.q_pos', 2.0)
        self.declare_parameter('mpc.q_yaw', 2.0)
        self.declare_parameter('mpc.q_vel', 0.5)
        self.declare_parameter('mpc.r_steer', 0.08)
        self.declare_parameter('mpc.r_accel', 0.01)
        self.declare_parameter('mpc.r_delta_steer', 0.8)
        self.declare_parameter('mpc.r_delta_accel', 0.005)

        # Get MPC parameters
        prediction_horizon = self.get_parameter('mpc.prediction_horizon').value
        control_horizon = self.get_parameter('mpc.control_horizon').value
        dt = self.get_parameter('mpc.dt').value
        q_pos = self.get_parameter('mpc.q_pos').value
        q_yaw = self.get_parameter('mpc.q_yaw').value
        q_vel = self.get_parameter('mpc.q_vel').value
        r_steer = self.get_parameter('mpc.r_steer').value
        r_accel = self.get_parameter('mpc.r_accel').value
        r_delta_steer = self.get_parameter('mpc.r_delta_steer').value
        r_delta_accel = self.get_parameter('mpc.r_delta_accel').value

        self.mpc_controller = MPCController(
            vehicle_model=self.vehicle_model,
            prediction_horizon=prediction_horizon,
            control_horizon=control_horizon,
            dt=dt,
            q_pos=q_pos,
            q_yaw=q_yaw,
            q_vel=q_vel,
            r_steer=r_steer,
            r_accel=r_accel,
            r_delta_steer=r_delta_steer,
            r_delta_accel=r_delta_accel
        )

        # Load reference path
        self.mpc_dt = self.mpc_controller.dt
        self._load_reference_path()

        self._marker_divisor = 0
        self.get_logger().info(
            'team_driver is up with MPC controller!')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    # ------------------------------------------------------------------
    # Reference path loading
    # ------------------------------------------------------------------
    def _centerline_candidates(self):
        """Places icra26_centerline.csv can legitimately live.

        The package is installed under race_ws/install/... but the map ships at
        the repo root, so the obvious 'share directory' guess does not exist.
        Try the ROS way first, then walk up from this file to the repo root.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = []

        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('team_driver'), 'maps'))
        except Exception:
            pass

        # Walk up looking for a maps/ dir holding the file. Seven levels covers
        # install/team_driver/lib/pythonX.Y/site-packages/team_driver/.
        node = here
        for _ in range(8):
            node = os.path.dirname(node)
            candidates.append(os.path.join(node, 'maps'))

        return [os.path.join(c, 'icra26_centerline.csv') for c in candidates]

    def _load_reference_path(self):
        """Load centerline as reference path."""
        try:
            centerline_path = next(
                (p for p in self._centerline_candidates() if os.path.exists(p)), None)

            if centerline_path is None:
                self.get_logger().error(
                    'icra26_centerline.csv not found in any of: '
                    f'{self._centerline_candidates()}. MPC will NOT be used; '
                    'falling back to the reactive gap-following controller.')
                self.reference_path = None
                return

            # The file is commented CSV; np.loadtxt needs comments='#' and the
            # header lines dropped.
            centerline_data = np.loadtxt(centerline_path, delimiter=',', comments='#')
            centerline_data = np.atleast_2d(centerline_data)

            self.get_logger().info(
                f'Loaded {len(centerline_data)} centerline points from {centerline_path}')

            self.reference_path = centerline_data[:, :2]

            # Curvature-limited speeds, not a flat 3 m/s. The track is ~0.5 m
            # wide and the centerline peaks well above a curvature that 3 m/s
            # can hold, so a constant reference drives straight into the
            # outside wall at the first real corner.
            self.reference_speeds = self.mpc_controller.speed_profile(
                v_max=self.v_max, lat_accel=self.lat_accel, a_brake=self.a_brake,
                path=self.reference_path)
            self.get_logger().info(
                f'Speed profile: {self.reference_speeds.min():.2f}'
                f'-{self.reference_speeds.max():.2f} m/s')

            self.mpc_controller.set_reference_path(self.reference_path, self.reference_speeds)

        except Exception as e:
            self.get_logger().error(f'Error loading reference path: {e}')
            self.reference_path = None

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (about 40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """MPC-based planning function.

        Uses Model Predictive Control to compute optimal steering and speed
        based on vehicle dynamics, reference path, and current state.
        """
        # 1. Restrict view to forward field of view [-100 deg, +100 deg] (~ -1.75 to +1.75 rad)
        fov_mask = (angles >= -1.75) & (angles <= 1.75)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            # No valid LiDAR data - fallback to crawl speed
            return 0.0, self.crawl_speed

        # 2. Basic obstacle avoidance - keep existing safety logic for robustness
        # Locate nearest obstacle point inside FOV
        min_idx = np.argmin(fov_ranges)
        min_dist = fov_ranges[min_idx]

        # Create a safety bubble around nearest obstacle (radius ~0.35m)
        car_radius = 0.35
        bubble_rad = math.atan2(car_radius, max(min_dist, 0.1))

        proc_ranges = fov_ranges.copy()
        bubble_mask = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_rad
        proc_ranges[bubble_mask] = 0.0

        # 3. Find the largest continuous gap beyond threshold distance
        gap_threshold = 1.5  # meters
        free_space = proc_ranges > gap_threshold

        if not np.any(free_space):
            # Fallback: steer towards furthest visible point if no clear gap
            best_idx = np.argmax(proc_ranges)
            target_angle = fov_angles[best_idx]
        else:
            # Detect contiguous gap ranges
            bounded = np.pad(free_space.astype(int), (1, 1), 'constant')
            diffs = np.diff(bounded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            # Choose the widest gap
            gap_lengths = ends - starts
            max_gap_idx = np.argmax(gap_lengths)

            start_i = starts[max_gap_idx]
            end_i = ends[max_gap_idx]

            # Target the center of the largest gap
            gap_angles = fov_angles[start_i:end_i]
            target_angle = (gap_angles[0] + gap_angles[-1]) / 2.0

        # 4. Reactive steering: steer at the middle of the widest gap. Used as
        #    the MPC's warm-start guess and as the fallback if the MPC fails.
        reactive_steering = float(np.clip(0.85 * target_angle, -0.4, 0.4))

        # 5. Reactive speed: a crude clearance heuristic, kept as an upper
        #    bound and fallback only. The real speed target comes from the
        #    curvature profile once the MPC is in the loop.
        front_mask = (angles >= -0.2) & (angles <= 0.2)
        front_dist = np.mean(ranges[front_mask]) if np.any(front_mask) else 1.0

        if abs(reactive_steering) > 0.25 or front_dist < 2.0:
            reactive_speed = 2.0
        elif front_dist > 5.0 and abs(reactive_steering) < 0.1:
            reactive_speed = 4.5
        else:
            reactive_speed = 3.0

        # 6. Use MPC when we have both a path and a state. Without a reference
        #    path the MPC has nothing to track and would drive the car to the
        #    origin, so stay reactive in that case.
        if self.position is not None and self.reference_path is not None:
            try:
                # Current state: [x, y, yaw, velocity, beta_sideslip]
                # Beta (sideslip) is assumed 0 for simplified model
                current_state = np.array([
                    self.position[0],  # x
                    self.position[1],  # y
                    self.yaw,          # yaw
                    self.speed,        # velocity
                    0.0                # beta_sideslip (assumed zero)
                ])

                # Warm-start the optimizer from the reactive command.
                current_control = np.array([reactive_steering, 0.0])

                mpc_control, diagnostics = self.mpc_controller.get_control(
                    current_state, current_control)

                steering = float(mpc_control[0])
                accel = float(mpc_control[1])

                # /drive takes a SPEED, not an acceleration -- the simulator
                # closes that loop itself. The MPC plans in acceleration, so
                # project one step forward.
                speed = max(self.speed + accel * self.mpc_dt, 0.0)

                # Never let the MPC talk the car into a wall: the reactive
                # clearance heuristic is a genuine safety bound, so take the
                # lower of the two.
                speed = min(speed, reactive_speed)
                steering = float(np.clip(steering, -self.steer_limit, self.steer_limit))

                if hasattr(self, '_diag_counter'):
                    self._diag_counter += 1
                else:
                    self._diag_counter = 0

                if self._diag_counter % 50 == 0:  # Every ~2 s at 40 Hz
                    self.get_logger().info(
                        f'MPC: steer={steering:.3f}, speed={speed:.2f} '
                        f'(cur {self.speed:.2f}), solve={diagnostics["solve_time_ms"]:.1f}ms')

            except Exception as e:
                self.get_logger().warn(f'MPC failed, using reactive: {e}')
                steering = reactive_steering
                speed = reactive_speed
        else:
            # No odometry or no reference path - use reactive
            steering = reactive_steering
            speed = reactive_speed

        return steering, speed

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
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