"""
Comparative Cooling Actuation Strategies
----------------------------------------
Implements three distinct cooling control paradigms:
1. Reactive Hysteresis (Bang-Bang / Thermostatic On-Off)
2. Closed-Loop PID (Proportional-Integral-Derivative with Anti-Windup)
3. ML-Predictive Look-Ahead Optimizer (Anticipatory Pre-cooling minimizing pump power P ~ u^3)
"""

import numpy as np
from abc import ABC, abstractmethod

class BaseCoolingController(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute_action(
        self,
        current_temp: float,
        predicted_temp: float,
        current_load: float,
        target_temp: float = 30.0
    ) -> float:
        """
        Returns normalized cooling actuation signal u in [0.0, 1.0].
        """
        pass

# -------------------------------------------------------------
# Strategy 1: Reactive Hysteresis (Bang-Bang)
# -------------------------------------------------------------
class HysteresisCoolingController(BaseCoolingController):
    def __init__(self, t_on: float = 34.0, t_off: float = 29.0):
        self.t_on = t_on
        self.t_off = t_off
        self.state = 0.0 # 0.0 = OFF, 1.0 = FULL ON

    def reset(self):
        self.state = 0.0

    def compute_action(
        self,
        current_temp: float,
        predicted_temp: float,
        current_load: float,
        target_temp: float = 30.0
    ) -> float:
        if current_temp >= self.t_on:
            self.state = 1.0
        elif current_temp <= self.t_off:
            self.state = 0.0
        return self.state

# -------------------------------------------------------------
# Strategy 2: Proportional-Integral-Derivative (PID)
# -------------------------------------------------------------
class PIDCoolingController(BaseCoolingController):
    def __init__(
        self,
        kp: float = 0.18,
        ki: float = 0.015,
        kd: float = 0.45,
        integral_max: float = 20.0,
        dt: float = 1.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_action(
        self,
        current_temp: float,
        predicted_temp: float,
        current_load: float,
        target_temp: float = 30.0
    ) -> float:
        error = current_temp - target_temp
        
        # Accumulate integral only if temperature is above lower threshold to avoid negative windup
        if current_temp > (target_temp - 2.0):
            self.integral += error * self.dt
            self.integral = float(np.clip(self.integral, -self.integral_max, self.integral_max))
        else:
            self.integral *= 0.9 # Bleed down

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u_raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(u_raw, 0.0, 1.0))

# -------------------------------------------------------------
# Strategy 3: ML-Predictive Look-Ahead Optimizer
# -------------------------------------------------------------
class MLPredictiveCoolingController(BaseCoolingController):
    """
    Anticipatory Thermal Management:
    Uses ML multi-step temperature forecast T_pred(t + H) and instantaneous Joule heating proxy.
    Exploits the cubic pump affinity law (P ~ u^3):
    Pre-cooling early at moderate flow (u ~ 0.4) consumes ~93% LESS instantaneous power
    than waiting for overshoot and running at full throttle (u = 1.0).
    """
    def __init__(
        self,
        target_temp: float = 30.0,
        t_danger: float = 35.0,
        horizon_s: float = 10.0,
        lookahead_weight: float = 0.7
    ):
        self.target_temp = target_temp
        self.t_danger = t_danger
        self.horizon_s = horizon_s
        self.lookahead_weight = lookahead_weight
        self.prev_u = 0.0

    def reset(self):
        self.prev_u = 0.0

    def compute_action(
        self,
        current_temp: float,
        predicted_temp: float,
        current_load: float,
        target_temp: float = 30.0
    ) -> float:
        # Effective temperature metric blends current measurement with ML forecast
        t_effective = (1.0 - self.lookahead_weight) * current_temp + self.lookahead_weight * predicted_temp
        
        # Anticipatory current surge feedforward
        joule_load = (abs(current_load) ** 2) * 0.05
        
        if t_effective <= target_temp and current_temp <= target_temp:
            target_u = 0.0
        elif t_effective > self.t_danger:
            # Urgent cooling required to prevent thermal runaway risk
            severity = (t_effective - self.t_danger) / 5.0
            target_u = 0.65 + 0.35 * np.clip(severity, 0.0, 1.0)
        else:
            # Anticipatory proportional throttling: gentle cooling to arrest the temperature climb
            delta = t_effective - target_temp
            # Sub-linear power-optimal curve: avoids high u unless strictly necessary
            target_u = 0.25 * (delta ** 0.8) + 0.03 * np.sqrt(joule_load)

        # Rate limiter on pump actuation (reduces mechanical wear & fluid pressure hammer)
        u_clamped = np.clip(target_u, 0.0, 1.0)
        max_slew_rate = 0.15 # Max 15% change per step
        u_smooth = np.clip(u_clamped, self.prev_u - max_slew_rate, self.prev_u + max_slew_rate)
        self.prev_u = u_smooth

        return float(u_smooth)
