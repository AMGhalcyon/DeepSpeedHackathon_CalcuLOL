#!/usr/bin/env python3
"""Vehicle model wrapper for MPC controller.

Provides interface to F1TENTH vehicle dynamics for use in MPC prediction.
"""

import math
from dataclasses import dataclass
import numpy as np
from typing import Tuple

# The judged steering servo over one MPC step, calibrated against GymPlant by
# least squares (test/test_steering_actuator.py pins it). A 25 ms command on the
# 10 ms physics grid is held for 2 then 3 steps, so the target is the mean move
# of those two holds. The continuous-time alpha, 1 - exp(-kp*dt) = 0.615, is
# wrong here: the plant's servo is a discrete Euler step, which closes more of
# the error per step than its continuous limit would.
SERVO_CAL_DT = 0.025      # [s] the step the constants below are fitted for
SERVO_ALPHA = 0.6903      # fraction of a small error closed per SERVO_CAL_DT
SERVO_D_MAX = 0.0801      # [rad] wheel travel per SERVO_CAL_DT at the rate limit
# Knee sharpness of the soft clip. tanh (the plan's first choice) fits no
# better than 6.7 % of d_max and overstates the small-signal rate by 26 %; this
# fits within 1.5 % and is exact in the linear regime.
SERVO_SHARPNESS = 8


@dataclass(frozen=True)
class ServoModel:
    """The wheel's move over one step toward a held command, smoothly saturated.

    move = d_max * x / (1 + |x|^p)^(1/p),  x = alpha * error / d_max

    Linear with slope alpha for small errors, tending to d_max in saturation,
    and strictly monotone: the derivative (1 + |x|^p)^-(1 + 1/p) is never zero.
    That matters because SLSQP differentiates by finite differences over the
    command; a hard clip reads a gradient of exactly zero the moment the rate
    limit binds. Here the gradient stays visible to SLSQP's 1.5e-8 step well
    past the knee (tested at 2.6x it). Past ~0.5 rad of error it does fade to
    numerically nothing -- honestly: every such command moves the wheel by
    d_max, and there the effort and rate costs carry the gradient, pulling the
    command back toward the knee. It never overshoots the command.
    """

    alpha: float = SERVO_ALPHA
    d_max_cal: float = SERVO_D_MAX
    sharpness: int = SERVO_SHARPNESS
    cal_dt: float = SERVO_CAL_DT

    def alpha_at(self, dt: float) -> float:
        """Linear fraction closed over dt, compounding the calibrated step."""
        return 1.0 - (1.0 - self.alpha) ** (dt / self.cal_dt)

    def d_max(self, dt: float) -> float:
        """[rad] Most the wheel can travel in dt."""
        return self.d_max_cal * dt / self.cal_dt

    def move(self, error: float, dt: float) -> float:
        """[rad] How far the wheel moves in dt toward a command `error` away."""
        if dt <= 0.0:
            return 0.0
        d = self.d_max(dt)
        x = error * self.alpha_at(dt) / d
        p = self.sharpness
        return d * x / (1.0 + abs(x) ** p) ** (1.0 / p)


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

    def __init__(self, params: VehicleParameters = None, model_type: str = 'KS',
                 servo: ServoModel = None):
        """Initialize vehicle model.

        Args:
            params: Vehicle parameters (defaults to F1TENTH)
            model_type: 'KS' for kinematic single track
            servo: Steering servo model for `actuated_step` (defaults to the
                one calibrated against the judged plant)
        """
        self.params = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self.model_type = model_type
        self.servo = servo if servo is not None else ServoModel()

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

    def actuated_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """One step with the steering servo in the loop.

        The wheel angle is the sixth state. The car turns with the wheel as it
        stands, not with the command: rows 0-4 are the Euler kinematic step
        driven by state[5], and the command only moves the wheel, through the
        servo model. So the rate limit is a property of every predicted
        trajectory, at no cost in decision variables or constraints.

        Args:
            state: [x, y, yaw, v, beta, delta]
            control: [delta_cmd, acceleration]
            dt: Time step

        Returns:
            Next state [x, y, yaw, v, beta, delta]
        """
        delta = state[5]
        nxt = np.empty(6)
        nxt[:5] = self.dynamics_discrete(state[:5], (delta, control[1]), dt)
        nxt[5] = delta + self.servo.move(control[0] - delta, dt)
        return nxt

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
            -0.5,  # sideslip (rough estimate)
            self.params.s_min,  # wheel angle, for actuated_step's sixth state
        ])

        max_bounds = np.array([
            float('inf'),   # x
            float('inf'),   # y
            float('inf'),   # yaw
            self.params.v_max,  # velocity
            0.5,   # sideslip
            self.params.s_max,  # wheel angle
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