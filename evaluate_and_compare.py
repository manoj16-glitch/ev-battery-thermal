"""
Cooling Strategy Benchmark & Comparative Simulation Engine
----------------------------------------------------------
Simulates dynamic driving cycles across multiple cooling strategies:
1. No Active Cooling (Passive convection baseline)
2. Reactive Hysteresis (Bang-Bang)
3. Closed-Loop PID
4. ML-Predictive Look-Ahead Optimization

Quantifies:
- Peak Core and Surface Temperatures (deg C)
- Time above degradation threshold (> 35 deg C)
- Parasitic Pump Energy Consumption (Watt-hours)
- Actuator Fatigue / Total Variation (churn)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from thermal_model import BatteryThermalModel
from cooling_controllers import (
    HysteresisCoolingController,
    PIDCoolingController,
    MLPredictiveCoolingController
)

def run_thermal_cooling_benchmark(
    test_cycle_df: pd.DataFrame,
    ml_predictor,
    feature_names: List[str],
    t_target: float = 30.0,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Executes cycle-level comparative simulation across all cooling strategies.
    """
    strategies = {
        "Uncontrolled (No Cooling)": None,
        "Reactive Hysteresis (Bang-Bang)": HysteresisCoolingController(t_on=33.5, t_off=29.5),
        "Closed-Loop PID": PIDCoolingController(kp=0.20, ki=0.015, kd=0.35, dt=dt),
        "ML-Predictive Look-Ahead": MLPredictiveCoolingController(target_temp=t_target, horizon_s=10.0)
    }

    results = {}
    time_series_logs = {}

    currents = test_cycle_df["current"].values
    socs = test_cycle_df["soc"].values
    cycles = test_cycle_df["cycle"].values
    X_features = test_cycle_df[feature_names].values

    # Pre-generate ML predictions for speed
    y_preds = ml_predictor.predict(X_features)

    for name, controller in strategies.items():
        sim = BatteryThermalModel()
        sim.reset(initial_temp=24.0)
        if controller is not None:
            controller.reset()

        log_t_core = []
        log_t_surf = []
        log_u = []
        log_p_pump = []
        log_q_gen = []
        log_q_cool = []

        total_steps = len(currents)

        for step in range(total_steps):
            curr = currents[step]
            soc = socs[step]
            cycle_idx = cycles[step]
            pred_t = y_preds[step]

            # Current surface temperature measurement
            current_t = sim.t_surf

            # Actuator decision
            if controller is None:
                u = 0.0
            else:
                u = controller.compute_action(
                    current_temp=current_t,
                    predicted_temp=pred_t,
                    current_load=curr,
                    target_temp=t_target
                )

            # Advance physics simulation
            step_out = sim.step(current=curr, cooling_u=u, dt=dt, soc=soc, cycle=cycle_idx)

            log_t_core.append(step_out["t_core"])
            log_t_surf.append(step_out["t_surf"])
            log_u.append(u)
            log_p_pump.append(step_out["p_pump"])
            log_q_gen.append(step_out["q_gen"])
            log_q_cool.append(step_out["q_cool"])

        # Compute summary KPIs
        t_core_arr = np.array(log_t_core)
        t_surf_arr = np.array(log_t_surf)
        u_arr = np.array(log_u)
        p_pump_arr = np.array(log_p_pump)

        # Parasitic electrical energy: Joules -> Watt-hours (E = sum(P * dt) / 3600)
        total_energy_wh = float(np.sum(p_pump_arr * dt) / 3600.0)
        time_above_35 = float(np.sum(t_core_arr > 35.0) * dt)
        time_above_35_pct = float(100.0 * np.mean(t_core_arr > 35.0))
        actuator_wear = float(np.sum(np.abs(np.diff(u_arr))))

        results[name] = {
            "max_core_temp_c": float(np.max(t_core_arr)),
            "max_surf_temp_c": float(np.max(t_surf_arr)),
            "mean_core_temp_c": float(np.mean(t_core_arr)),
            "temp_std_c": float(np.std(t_core_arr)),
            "time_above_35c_s": time_above_35,
            "time_above_35c_pct": time_above_35_pct,
            "parasitic_energy_wh": total_energy_wh,
            "actuator_churn_score": actuator_wear
        }

        time_series_logs[name] = {
            "t_core": t_core_arr,
            "t_surf": t_surf_arr,
            "u": u_arr,
            "p_pump": p_pump_arr
        }

    return results, time_series_logs
