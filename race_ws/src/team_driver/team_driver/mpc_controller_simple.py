#!/usr/bin/env python3
"""Simple Model Predictive Controller for F1TENTH autonomous racing.

This is a dependency-free version for initial testing that uses basic
control logic instead of optimization.
"""

import math
from typing import Tuple, Optional
import time


class SimpleMPCController:
    """Simple MPC-like controller for vehicle path following."""

    def __init__(self,
                 vehicle_model=None,
                 prediction_horizon: int = 8,
                 control_horizon: int = None,
                 dt: float = 0.025,
                 q_pos: float = 2.0,
                 q_yaw: float = 0.5,
                 q_vel: float = 0.1,
                 r_steer: float = 0.02,
                 r_accel: float = 0.01,
                 r_delta_steer: float = 0.01,
                 r_delta_accel: float = 0.005):
        """Initialize simple MPC controller.

        Args:
            vehicle_model: Vehicle model for predictions
            prediction_horizon: Number of steps to predict ahead
            control_horizon: Number of steps to optimize controls
            dt: Time step for discrete dynamics (seconds)
            q_pos: Weight on position tracking error
            q_yaw: Weight on yaw tracking error
            q_vel: Weight on velocity tracking error
            r_steer: Weight on steering control effort
            r_accel: Weight on acceleration control effort
            r_delta_steer: Weight on steering rate of change
            r_delta_accel: Weight on acceleration rate of change
        """
        self.vehicle_model = vehicle_model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon if control_horizon is not None else prediction_horizon
        self.dt = dt

        # Cost function weights
        self.q_pos = q_pos
        self.q_yaw = q_yaw
        self.q_vel = q_vel
        self.r_steer = r_steer
        self.r_accel = r_accel
        self.r_delta_steer = r_delta_steer
        self.r_delta_accel = r_delta_accel

        # State and control dimensions
        self.state_dim = 4  # [x, y, yaw, v]
        self.control_dim = 2  # [steering, acceleration]

        # Previous control for rate limiting
        self.prev_control = [0.0, 0.0]

        # Reference path
        self.reference_path = None
        self.reference_speeds = None

        print(f"Simple MPC Controller initialized:")
        print(f"  Prediction horizon: {self.prediction_horizon}")
        print(f"  Control horizon: {self.control_horizon}")
        print(f"  DT: {self.dt}s ({1/dt:.1f} Hz)")

    def set_reference_path(self, path, speeds=None):
        """Set reference path for tracking.

        Args:
            path: List of [x, y] waypoints
            speeds: Optional list of target speeds
        """
        self.reference_path = path
        self.reference_speeds = speeds

        # If no speeds provided, use constant
        if speeds is None and path is not None:
            self.reference_speeds = [3.0] * len(path)

    def _get_reference_at_index(self, idx):
        """Get reference state and speed at path index."""
        if self.reference_path is None or len(self.reference_path) == 0:
            return [0.0, 0.0, 0.0, 0.0], 0.0

        # Clamp index to valid range
        idx = max(0, min(idx, len(self.reference_path) - 1))

        # Get reference position
        ref_x, ref_y = self.reference_path[idx]

        # Estimate heading from path direction
        if idx < len(self.reference_path) - 1:
            next_x, next_y = self.reference_path[idx + 1]
            path_dx = next_x - ref_x
            path_dy = next_y - ref_y
            ref_yaw = math.atan2(path_dy, path_dx)
        elif idx > 0:
            prev_x, prev_y = self.reference_path[idx - 1]
            path_dx = ref_x - prev_x
            path_dy = ref_y - prev_y
            ref_yaw = math.atan2(path_dy, path_dx)
        else:
            ref_yaw = 0.0

        # Get reference speed
        if self.reference_speeds is not None:
            ref_speed = self.reference_speeds[idx]
        else:
            ref_speed = 3.0

        return [ref_x, ref_y, ref_yaw, ref_speed], ref_speed

    def _simple_predict(self, state, control, steps):
        """Simple prediction using bicycle model.

        Args:
            state: [x, y, yaw, v]
            control: [steering, acceleration]
            steps: Number of steps to predict

        Returns:
            List of predicted states
        """
        states = [state[:]]  # Copy initial state
        x, y, yaw, v = state
        steer, accel = control

        # Simple bicycle model parameters (approximate)
        wheelbase = 0.33  # Approximate for F1TENTH

        for _ in range(steps):
            # Update velocity
            v = v + accel * self.dt
            v = max(0.0, min(v, 8.0))  # Speed limits

            # Update position
            x = x + v * math.cos(yaw) * self.dt
            y = y + v * math.sin(yaw) * self.dt

            # Update yaw (steering)
            yaw = yaw + v * math.tan(steer) / wheelbase * self.dt

            # Normalize yaw
            while yaw > math.pi:
                yaw -= 2 * math.pi
            while yaw < -math.pi:
                yaw += 2 * math.pi

            states.append([x, y, yaw, v])

        return states

    def _compute_cost(self, states, controls, ref_states, ref_speeds):
        """Compute simple cost function.

        Args:
            states: Predicted states over horizon
            controls: Control sequence over horizon
            ref_states: Reference states over horizon
            ref_speeds: Reference speeds over horizon

        Returns:
            Total cost
        """
        cost = 0.0

        # State tracking cost
        for i, (state, ref_state) in enumerate(zip(states, ref_states)):
            # Position error
            dx = state[0] - ref_state[0]
            dy = state[1] - ref_state[1]
            pos_error = math.sqrt(dx*dx + dy*dy)

            # Yaw error
            yaw_error = state[2] - ref_state[2]
            # Normalize yaw error
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            # Velocity error
            vel_error = state[3] - ref_state[3]

            # Accumulate weighted cost
            cost += self.q_pos * (pos_error * pos_error)
            cost += self.q_yaw * (yaw_error * yaw_error)
            cost += self.q_vel * (vel_error * vel_error)

        # Control effort cost
        for control in controls:
            steer, accel = control
            cost += self.r_steer * (steer * steer)
            cost += self.r_accel * (accel * accel)

        # Control rate cost (vs previous)
        if len(controls) > 0:
            steer, accel = controls[0]
            dsteer = steer - self.prev_control[0]
            daccel = accel - self.prev_control[1]
            cost += self.r_delta_steer * (dsteer * dsteer)
            cost += self.r_delta_accel * (daccel * daccel)

        return cost

    def solve(self, current_state, current_control=None):
        """Solve MPC using simple grid search (for demonstration).

        Args:
            current_state: Current vehicle state [x, y, yaw, v]
            current_control: Current control input [steering, acceleration]

        Returns:
            (optimal_control, solve_time)
        """
        start_time = time.time()

        if current_control is not None:
            self.prev_control = current_control[:]

        # Generate reference trajectory
        ref_states = []
        ref_speeds = []
        for i in range(self.prediction_horizon + 1):
            ref_state, ref_speed = self._get_reference_at_index(i)
            ref_states.append(ref_state)
            ref_speeds.append(ref_speed)

        # Simple grid search over steering and acceleration
        best_cost = float('inf')
        best_control = [0.0, 0.0]

        # Define search ranges
        steer_range = [-0.3, -0.1, 0.0, 0.1, 0.3]  # radians
        accel_range = [-1.0, -0.5, 0.0, 0.5, 1.0]  # m/s^2

        for steer in steer_range:
            for accel in accel_range:
                control = [steer, accel]

                # Predict trajectory
                states = self._simple_predict(current_state, control, self.prediction_horizon)

                # Compute cost
                cost = self._compute_cost(states, [control] * self.prediction_horizon,
                                        ref_states, ref_speeds)

                if cost < best_cost:
                    best_cost = cost
                    best_control = control[:]

        solve_time = time.time() - start_time
        return best_control, solve_time

    def get_control(self, current_state, current_control=None):
        """Get MPC control command.

        Args:
            current_state: Current vehicle state [x, y, yaw, v]
            current_control: Current control input [steering, acceleration]

        Returns:
            (control_command, diagnostics)
        """
        control, solve_time = self.solve(current_state, current_control)

        diagnostics = {
            'solve_time_ms': solve_time * 1000,
            'control_steering': control[0],
            'control_acceleration': control[1],
            'feasible': True
        }

        return control, diagnostics


def test_simple_mpc():
    """Simple test of MPC controller."""
    print("Testing Simple MPC Controller...")

    # Create controller
    mpc = SimpleMPCController(prediction_horizon=5, dt=0.1)

    # Set simple reference path (straight line)
    path = [[0, 0], [5, 0], [10, 0]]  # Straight line along x-axis
    mpc.set_reference_path(path)

    # Test initial state at origin, zero velocity
    initial_state = [0.0, 0.0, 0.0, 0.0]  # [x, y, yaw, v]
    current_control = [0.0, 0.0]  # [steering, acceleration]

    # Get control command
    control, diagnostics = mpc.get_control(initial_state, current_control)

    print(f"Initial state: {initial_state}")
    print(f"Control command: steering={control[0]:.3f}, accel={control[1]:.3f}")
    print(f"Solve time: {diagnostics['solve_time_ms']:.1f} ms")
    print(f"Diagnostics: {diagnostics}")


if __name__ == "__main__":
    test_simple_mpc()