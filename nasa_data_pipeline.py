"""
NASA Battery Aging Dataset Parser & Dynamic EV Thermal Data Pipeline
--------------------------------------------------------------------
Handles:
1. Ingestion of NASA PCoE Battery Aging Dataset (.mat files: B0005, B0006, etc.)
2. High-fidelity synthetic EV dynamic load profile generator (WLTP/US06 pulse style)
3. Advanced feature engineering:
   - Electrical power (I * V), Joule heating proxy (I^2)
   - Dynamic rolling windows (mean, std, min, max)
   - Instantaneous thermal gradient (dT/dt)
   - Coulomb counting SOC estimation
   - Cycle number & cumulative energy throughput
   - Time-lagged historical state vectors
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class NASADataPipeline:
    def __init__(self, nominal_capacity_ah: float = 2.0):
        self.nominal_capacity_ah = nominal_capacity_ah
        self.capacity_coulombs = nominal_capacity_ah * 3600.0

    def load_nasa_mat_file(self, mat_filepath: str) -> pd.DataFrame:
        """
        Parses NASA Prognostics Center of Excellence (PCoE) battery .mat files.
        Extracts discharge cycles with voltage, current, temperature, and time.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required to load .mat files. Run: pip install scipy")

        if not os.path.exists(mat_filepath):
            raise FileNotFoundError(f"NASA file not found: {mat_filepath}")

        mat = loadmat(mat_filepath)
        key_name = [k for k in mat.keys() if not k.startswith("__")][0]
        battery_data = mat[key_name][0, 0]["cycle"][0]

        records = []
        cycle_idx = 0

        for cycle_entry in battery_data:
            c_type = cycle_entry["type"][0]
            if c_type == "discharge":
                cycle_idx += 1
                c_data = cycle_entry["data"][0, 0]
                
                voltage = c_data["Voltage_measured"][0].flatten()
                current = c_data["Current_measured"][0].flatten()
                temperature = c_data["Temperature_measured"][0].flatten()
                time_s = c_data["Time"][0].flatten()

                # Filter uniform sampling
                for t, v, i, temp in zip(time_s, voltage, current, temperature):
                    records.append({
                        "cycle": cycle_idx,
                        "time_s": float(t),
                        "voltage": float(v),
                        "current": float(abs(i)), # Magnitude of discharge current
                        "temperature": float(temp),
                    })

        df = pd.DataFrame(records)
        return self.engineer_features(df)

    def generate_synthetic_ev_dataset(
        self,
        num_cycles: int = 50,
        samples_per_cycle: int = 1200,
        dt: float = 1.0,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates realistic dynamic EV driving and fast-charge cycles.
        Combines stochastic acceleration pulses (0-15A), regenerative braking (-5A),
        cruise loads (3-6A), and rest periods, simulated through calibrated physics.
        """
        np.random.seed(random_seed)
        from thermal_model import BatteryThermalModel

        sim = BatteryThermalModel()
        records = []

        total_time = 0.0

        for cycle in range(1, num_cycles + 1):
            sim.reset(initial_temp=24.0 + np.random.uniform(-1.0, 1.0))
            soc = 1.0
            cycle_time = 0.0

            # Dynamic profile: mix of urban stop-and-go and highway acceleration bursts
            t = 0
            while t < samples_per_cycle and soc > 0.1:
                # Drive segments: idle (0A), cruise (2-4A), hard accel (8-14A), regen (-2 to -4A)
                segment_type = np.random.choice(["idle", "cruise", "accel", "regen"], p=[0.15, 0.50, 0.25, 0.10])
                segment_duration = int(np.random.uniform(5, 30))

                for _ in range(segment_duration):
                    if t >= samples_per_cycle or soc <= 0.1:
                        break

                    if segment_type == "idle":
                        curr = float(np.random.normal(0.05, 0.02))
                    elif segment_type == "cruise":
                        curr = float(np.random.normal(3.5, 0.5))
                    elif segment_type == "accel":
                        curr = float(np.random.normal(9.0, 2.0))
                    else: # regen
                        curr = float(np.random.normal(-3.0, 0.8))

                    # Update SOC
                    soc -= (curr * dt) / self.capacity_coulombs
                    soc = float(np.clip(soc, 0.05, 1.0))

                    # Cell terminal voltage approximation: Voc(SOC) - I * Rint
                    voc = 3.2 + 0.9 * soc + 0.1 * np.log(max(soc, 0.01))
                    r_int = sim.compute_internal_resistance(soc=soc, cycle=cycle)
                    v_terminal = voc - curr * r_int + float(np.random.normal(0, 0.005))

                    # Benchmark uncontrolled thermal state (u=0 for raw dataset)
                    out = sim.step(current=curr, cooling_u=0.0, dt=dt, soc=soc, cycle=cycle)

                    records.append({
                        "cycle": cycle,
                        "time_s": cycle_time,
                        "total_time_s": total_time,
                        "current": curr,
                        "voltage": v_terminal,
                        "temperature": out["t_surf"], # Sensor on surface
                        "t_core_actual": out["t_core"],
                        "soc": soc
                    })

                    t += 1
                    cycle_time += dt
                    total_time += dt

        df = pd.DataFrame(records)
        return self.engineer_features(df)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates domain-specific thermal and electrical engineering features.
        """
        df = df.copy()

        # Electrical power & Joule heat proxy
        df["power_w"] = df["voltage"] * df["current"]
        df["joule_heat_proxy"] = df["current"] ** 2

        # Instantaneous rate of temperature change: dT/dt
        df["dt_temp"] = df.groupby("cycle")["temperature"].diff().fillna(0.0)

        # Rolling window aggregations (capturing thermal capacitance lag)
        for w in [5, 15, 30]:
            df[f"current_mean_{w}s"] = df.groupby("cycle")["current"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"current_std_{w}s"] = df.groupby("cycle")["current"].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0.0)
            df[f"joule_mean_{w}s"] = df.groupby("cycle")["joule_heat_proxy"].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"temp_mean_{w}s"] = df.groupby("cycle")["temperature"].transform(lambda x: x.rolling(w, min_periods=1).mean())

        # Cumulative Joule energy (proxy for heat buildup in current cycle)
        df["cum_joule_heat"] = df.groupby("cycle")["joule_heat_proxy"].cumsum()

        # Lagged features (t-1, t-3, t-5)
        for lag in [1, 3, 5]:
            df[f"temp_lag_{lag}"] = df.groupby("cycle")["temperature"].shift(lag).bfill()
            df[f"current_lag_{lag}"] = df.groupby("cycle")["current"].shift(lag).bfill()

        # If SOC is not provided, estimate via Coulomb counting
        if "soc" not in df.columns:
            df["coulomb_delta"] = (df["current"] * 1.0) / self.capacity_coulombs
            df["soc"] = 1.0 - df.groupby("cycle")["coulomb_delta"].cumsum()
            df["soc"] = df["soc"].clip(0.0, 1.0)
            df.drop(columns=["coulomb_delta"], inplace=True)

        return df

    def create_forecasting_dataset(
        self,
        df: pd.DataFrame,
        horizon_steps: int = 10,
        train_cycle_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Prepares feature matrix X and target y (T at t + horizon_steps).
        Splits by cycle index to ensure no data leakage across time-series.
        """
        # Create target: temperature 'horizon_steps' ahead
        df = df.copy()
        df["target_temp"] = df.groupby("cycle")["temperature"].shift(-horizon_steps)
        df.dropna(subset=["target_temp"], inplace=True)

        feature_cols = [
            "current", "voltage", "power_w", "joule_heat_proxy", "dt_temp",
            "current_mean_5s", "current_mean_15s", "current_mean_30s",
            "current_std_15s", "joule_mean_15s", "temp_mean_15s",
            "cum_joule_heat", "temp_lag_1", "temp_lag_3", "temp_lag_5",
            "current_lag_1", "current_lag_3", "temperature", "cycle", "soc"
        ]

        cycles = df["cycle"].unique()
        split_idx = int(len(cycles) * train_cycle_ratio)
        train_cycles = cycles[:split_idx]
        test_cycles = cycles[split_idx:]

        train_mask = df["cycle"].isin(train_cycles)
        test_mask = df["cycle"].isin(test_cycles)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target_temp"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "target_temp"].values

        return X_train, y_train, X_test, y_test, feature_cols
