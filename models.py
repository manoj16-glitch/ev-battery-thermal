"""
Machine Learning & Deep Learning Temperature Forecasting Models
--------------------------------------------------------------
Includes:
1. Random Forest Regressor (Nonlinear tree ensemble)
2. XGBoost Regressor (Gradient boosted decision trees)
3. Long Short-Term Memory (LSTM) Recurrent Neural Network (PyTorch / PyTorch-like)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

# Evaluation metrics
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
    max_err = float(np.max(np.abs(y_true - y_pred)))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_error": max_err
    }

# -------------------------------------------------------------
# 1. Random Forest Model
# -------------------------------------------------------------
class BatteryRandomForest:
    def __init__(self, n_estimators: int = 100, max_depth: int = 14, min_samples_split: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        if self.model is None:
            return {}
        return dict(sorted(zip(feature_names, self.model.feature_importances_), key=lambda x: x[1], reverse=True))

# -------------------------------------------------------------
# 2. XGBoost Model
# -------------------------------------------------------------
class BatteryXGBoost:
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1
        }
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.params)
            if eval_set is not None:
                self.model.fit(X, y, eval_set=[eval_set], verbose=False)
            else:
                self.model.fit(X, y)
        except ImportError:
            # Fallback to GradientBoostingRegressor if xgboost isn't installed
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["random_state"]
            )
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        if self.model is None:
            return {}
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}
        return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))

# -------------------------------------------------------------
# 3. LSTM Recurrent Neural Network
# -------------------------------------------------------------
class BatteryLSTM:
    """
    Sequence-to-One LSTM for Multi-Step Ahead Temperature Forecasting.
    Uses sliding history window of length W to predict T at t + H.
    """
    def __init__(
        self,
        window_size: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 25
    ):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates [num_samples, window_size, num_features] sequences."""
        num_samples = len(X) - self.window_size + 1
        if num_samples <= 0:
            raise ValueError("Data length must be greater than window_size.")
        
        X_seq = np.empty((num_samples, self.window_size, X.shape[1]), dtype=np.float32)
        y_seq = np.empty((num_samples,), dtype=np.float32)

        for i in range(num_samples):
            X_seq[i] = X[i : i + self.window_size]
            y_seq[i] = y[i + self.window_size - 1]

        return X_seq, y_seq

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader

            # Standardization
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_scale = np.std(X, axis=0) + 1e-8
            X_scaled = (X - self.scaler_mean) / self.scaler_scale

            X_seq, y_seq = self.create_sequences(X_scaled, y)

            class LSTMModule(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )

                def forward(self, x):
                    out, _ = self.lstm(x)
                    # Take last time step
                    last_step = out[:, -1, :]
                    return self.fc(last_step).squeeze(-1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = LSTMModule(X.shape[1], self.hidden_dim, self.num_layers, self.dropout).to(device)

            dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            self.model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()

        except ImportError:
            # High-performance lightweight recurrent/ridge regressor fallback if PyTorch is not installed
            from sklearn.linear_model import Ridge
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_scale = np.std(X, axis=0) + 1e-8
            X_scaled = (X - self.scaler_mean) / self.scaler_scale
            # Flatten lag window
            X_seq, y_seq = self.create_sequences(X_scaled, y)
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_flat, y_seq)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        dummy_y = np.zeros(len(X_scaled))
        X_seq, _ = self.create_sequences(X_scaled, dummy_y)

        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                device = next(self.model.parameters()).device
                self.model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(X_seq, dtype=torch.float32).to(device)
                    preds = self.model(inputs).cpu().numpy()
                return preds
        except Exception:
            pass

        # Ridge fallback
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        return self.model.predict(X_flat)
