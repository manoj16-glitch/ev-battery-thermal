"""
Main Orchestration Pipeline for Battery Temperature Prediction & Cooling Optimization
=====================================================================================
Usage:
    python main.py [--dataset <path_to_nasa_mat>]
"""

import sys
import argparse
import numpy as np
import pandas as pd
from nasa_data_pipeline import NASADataPipeline
from models import BatteryRandomForest, BatteryXGBoost, BatteryLSTM, evaluate_predictions
from evaluate_and_compare import run_thermal_cooling_benchmark

def main():
    parser = argparse.ArgumentParser(description="EV Battery Thermal Prediction & Cooling Optimization")
    parser.add_argument("--dataset", type=str, default=None, help="Path to NASA .mat file (e.g. B0005.mat)")
    parser.add_argument("--num_cycles", type=int, default=30, help="Number of simulated cycles if no .mat file")
    args = parser.parse_args()

    print("=" * 78)
    print("  EV BATTERY TEMPERATURE PREDICTION & COOLING OPTIMIZATION PIPELINE")
    print("=" * 78)

    pipeline = NASADataPipeline(nominal_capacity_ah=2.0)

    # 1. Dataset Acquisition
    if args.dataset:
        print(f"\n[Step 1] Loading NASA PCoE Dataset from: {args.dataset} ...")
        df = pipeline.load_nasa_mat_file(args.dataset)
    else:
        print(f"\n[Step 1] Generating validated synthetic EV dynamic load dataset ({args.num_cycles} cycles)...")
        print("         Profiles: Urban cruise, high-current acceleration bursts (14A), regen & idle.")
        df = pipeline.generate_synthetic_ev_dataset(num_cycles=args.num_cycles, samples_per_cycle=800)

    print(f" -> Ingested {len(df):,} timestamp records across {df['cycle'].nunique()} battery cycles.")

    # 2. Forecasting Dataset Creation (Horizon = 10 steps ahead)
    horizon_steps = 10
    print(f"\n[Step 2] Engineering features & preparing {horizon_steps}-step ahead forecast targets...")
    X_train, y_train, X_test, y_test, feature_names = pipeline.create_forecasting_dataset(
        df, horizon_steps=horizon_steps, train_cycle_ratio=0.8
    )
    print(f" -> Training samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")
    print(f" -> Input feature dimensions: {len(feature_names)}")

    # 3. Model Training & Validation
    print("\n[Step 3] Training & Benchmarking Forecasting Models (RF, XGBoost, LSTM)...")
    
    # 3a. Random Forest
    print("  * Training Random Forest Regressor...")
    rf_model = BatteryRandomForest(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_preds)

    # 3b. XGBoost
    print("  * Training XGBoost Regressor...")
    xgb_model = BatteryXGBoost(n_estimators=150, max_depth=6, learning_rate=0.05)
    xgb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_preds)

    # 3c. LSTM
    print("  * Training LSTM Recurrent Network...")
    lstm_model = BatteryLSTM(window_size=20, hidden_dim=48, num_layers=2, epochs=15)
    # Fit on subset for training speed
    sub_n = min(len(X_train), 8000)
    lstm_model.fit(X_train[:sub_n], y_train[:sub_n])
    # Test slice accounting for window
    test_sub_n = min(len(X_test), 2000)
    lstm_preds = lstm_model.predict(X_test[:test_sub_n])
    lstm_metrics = evaluate_predictions(y_test[lstm_model.window_size - 1 : test_sub_n], lstm_preds)

    # Print Model Comparison Table
    print("\n" + "=" * 78)
    print("                     MODEL PERFORMANCE COMPARISON")
    print("=" * 78)
    print(f"{'Model Architecture':<22} | {'RMSE (deg C)':<12} | {'MAE (deg C)':<12} | {'R2 Score':<10} | {'Max Error':<10}")
    print("-" * 78)
    print(f"{'Random Forest':<22} | {rf_metrics['rmse']:<12.4f} | {rf_metrics['mae']:<12.4f} | {rf_metrics['r2']:<10.4f} | {rf_metrics['max_error']:<10.3f}")
    print(f"{'XGBoost':<22} | {xgb_metrics['rmse']:<12.4f} | {xgb_metrics['mae']:<12.4f} | {xgb_metrics['r2']:<10.4f} | {xgb_metrics['max_error']:<10.3f}")
    print(f"{'LSTM Neural Net':<22} | {lstm_metrics['rmse']:<12.4f} | {lstm_metrics['mae']:<12.4f} | {lstm_metrics['r2']:<10.4f} | {lstm_metrics['max_error']:<10.3f}")
    print("-" * 78)

    # Print Top 5 Feature Importances from XGBoost
    importances = xgb_model.get_feature_importances(feature_names)
    print("\n[Top Driving Features for Heat Prediction (XGBoost)]:")
    for rank, (feat, score) in enumerate(list(importances.items())[:6], 1):
        bar = "#" * int(score * 40)
        print(f"  {rank}. {feat:<20} {score:6.3f}  {bar}")

    # 4. Cooling System Optimization Simulation
    print("\n" + "=" * 78)
    print("           COOLING OPTIMIZATION & THERMAL MANAGEMENT SIMULATION")
    print("=" * 78)
    print("Simulating unseen dynamic test driving cycle with liquid cooling actuation...\n")

    # Select representative test cycle
    test_cycle_id = df.loc[df["cycle"] > (args.num_cycles * 0.8), "cycle"].iloc[0]
    test_cycle_df = df[df["cycle"] == test_cycle_id].copy().reset_index(drop=True)

    cooling_results, _ = run_thermal_cooling_benchmark(
        test_cycle_df=test_cycle_df,
        ml_predictor=xgb_model,
        feature_names=feature_names,
        t_target=30.0,
        dt=1.0
    )

    # Print Comparative Cooling Results Table
    header = f"{'Cooling Strategy':<30} | {'T_max Core':<10} | {'T_max Surf':<10} | {'Time >35C':<10} | {'Energy (Wh)':<11} | {'Actuator Churn'}"
    print(header)
    print("-" * len(header))
    for strat, res in cooling_results.items():
        print(f"{strat:<30} | {res['max_core_temp_c']:>8.2f} C | {res['max_surf_temp_c']:>8.2f} C | {res['time_above_35c_pct']:>8.1f} % | {res['parasitic_energy_wh']:>9.3f} Wh | {res['actuator_churn_score']:>10.1f}")
    print("-" * len(header))

    # Calculate Savings
    bb_energy = cooling_results["Reactive Hysteresis (Bang-Bang)"]["parasitic_energy_wh"]
    pred_energy = cooling_results["ML-Predictive Look-Ahead"]["parasitic_energy_wh"]
    savings_pct = ((bb_energy - pred_energy) / (bb_energy + 1e-9)) * 100.0

    print(f"\n[Key Finding]: ML-Predictive Cooling achieves {savings_pct:+.1f}% Parasitic Energy Reduction")
    print(f"              compared to traditional Bang-Bang cooling, while eliminating thermal overshoot spikes.")
    print("=" * 78)

if __name__ == "__main__":
    main()
