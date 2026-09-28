#!/usr/bin/env python3
"""Vehicle model wrapper for MPC controller.

Provides interface to F1TENTH vehicle dynamics for use in MPC prediction.
"""

import math
import numpy as np
from typing import Tuple

# Import vehicle parameters from f1tenth_gym
try:
    from external.f1tenth_gym.f1tenth_gym.envs.dynamic_models import (
        F1TENTH_VEHICLE_PARAMETERS,
        VehicleParameters,
        DynamicModel
    )
    from external.f1tenth_gym.f1tenth_gym.envs.dynamic_models.kinematic import (
        vehicle_dynamics_ks_cog
    )
except ImportError:
    # Fallback definitions if import fails
    from dataclasses import dataclass
    import math

    @dataclass(frozen=True)
    class VehicleParameters:
        """Typed view over the vehicle parameters used by the dynamics models."""
        # Core parameters required by the simplified single-track models
        mu: float = 1.0489
        C_Sf: float = 4.718
        C_Sr: float = 5.4562
        lf: float = 0.15875
        lr: float = 0.17145
        h: float = 0.074
        m: float = 3.74
        I: float = 0.04712
        s_min: float = -0.4189
        s_max: float = 0.4189
        sv_min: float = -3.2
        sv_max: float = 3.2
        v_switch: float = 7.319
        a_max: float = 9.51
        v_min: float = -5.0
        v_max: float = 20.0
        width: float = 0.31
        length: float = 0.58
        collision_body_center_x: float = 0.1651

    F1TENTH_VEHICLE_PARAMETERS = VehicleParameters()
    DynamicModel = type('DynamicModel', (), {'KS': 1})

    def vehicle_dynamics_ks_cog(state, u, params):
        """Simplified kinematic bicycle model."""
        x, y, psi, v, beta = state
        delta, a = u  # steering angle, acceleration

        # Kinematic bicycle model
        x_dot = v * math.cos(psi + beta)
        y_dot = v * math.sin(psi + beta)
        psi_dot = v * math.cos(beta) * math.tan(delta) / (params.lf + params.lr)
        v_dot = a
        # Simplified sideslip approximation
        beta_dot = 0.0  # Placeholder

        return np.array([x_dot, y_dot, psi_dot, v_dot, beta_dot])


class VehicleModel:
    """Vehicle model wrapper for MPC prediction."""

    def __init__(self, params: VehicleParameters = None, model_type: str = 'KS'):
        """Initialize vehicle model.

        Args:
            params: Vehicle parameters (defaults to F1TENTH)
            model_type: 'KS' for kinematic single track
        """
        self.params = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self.model_type = model_type

        # State dimensions
        if model_type == 'KS':
            self.state_dim = 5  # [x, y, yaw, velocity, sideslip]
            self.control_dim = 2  # [steering_angle, acceleration]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def dynamics_continuous(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Continuous-time dynamics function.

        Args:
            state: [x, y, yaw, v, beta]
            control: [steering_angle, acceleration]

        Returns:
            State derivative [x_dot, y_dot, yaw_dot, v_dot, beta_dot]
        """
        if self.model_type == 'KS':
            return vehicle_dynamics_ks_cog(state, control, self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def dynamics_discrete(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Discrete-time dynamics using Euler integration.

        Args:
            state: Current state [x, y, yaw, v, beta]
            control: Control input [steering_angle, acceleration]
            dt: Time step

        Returns:
            Next state [x, y, yaw, v, beta]
        """
        state_dot = self.dynamics_continuous(state, control)
        return state + state_dot * dt

    def get_initial_state(self, pose: Tuple[float, float, float] = None) -> np.ndarray:
        """Get initial vehicle state.

        Args:
            pose: (x, y, yaw) in meters and radians

        Returns:
            Initial state vector
        """
        state = np.zeros(self.state_dim)
        if pose is not None:
            state[0] = pose[0]  # x
            state[1] = pose[1]  # y
            state[2] = pose[2]  # yaw
            # velocity and sideslip default to zero
        return state

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds for MPC constraints.

        Returns:
            (min_bounds, max_bounds) arrays
        """
        # Rough bounds - can be refined based on vehicle limits
        min_bounds = np.array([
            -float('inf'),  # x
            -float('inf'),  # y
            -float('inf'),  # yaw
            self.params.v_min,  # velocity
            -0.5  # sideslip (rough estimate)
        ])

        max_bounds = np.array([
            float('inf'),   # x
            float('inf'),   # y
            float('inf'),   # yaw
            self.params.v_max,  # velocity
            0.5   # sideslip
        ])

        return min_bounds, max_bounds

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control input bounds.

        Returns:
            (min_control, max_control) arrays
        """
        min_control = np.array([
            self.params.s_min,  # steering angle
            self.params.a_max * -0.3  # acceleration (conservative reverse)
        ])

        max_control = np.array([
            self.params.s_max,  # steering angle
            self.params.a_max   # acceleration
        ])

        return min_control, max_control


def test_vehicle_model():
    """Simple test of vehicle model."""
    model = VehicleModel()

    # Test initial state
    state = model.get_initial_state((0.0, 0.0, 0.0))
    print(f"Initial state: {state}")

    # Test dynamics
    control = np.array([0.1, 1.0])  # 0.1 rad steering, 1.0 m/s^2 acceleration
    dt = 0.025  # 40Hz

    next_state = model.dynamics_discrete(state, control, dt)
    print(f"Next state: {next_state}")

    # Test bounds
    state_min, state_max = model.get_state_bounds()
    control_min, control_max = model.get_control_bounds()
    print(f"State bounds: [{state_min}, {state_max}]")
    print(f"Control bounds: [{control_min}, {control_max}]")


if __name__ == "__main__":
    test_vehicle_model()