"""
Deep Learning Sequence Modeling Module for Transit Delay Prediction
Implements Sliding Window Dataset creation, PyTorch Stacked LSTM, GRU, and Bidirectional LSTM models.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Dict, Any, Optional


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across numpy, python, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesSequenceDataset(Dataset):
    """
    PyTorch Dataset for Sliding Window Sequences.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SlidingWindowProcessor:
    """
    Handles feature scaling and sliding window sequence generation for Deep Learning.
    Ensures zero data leakage by fitting scalers strictly on training data.
    """
    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "mean_delay",
        lookback_window: int = 24,
        scaler_type: str = "minmax"
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback_window = lookback_window
        self.scaler_type = scaler_type
        
        if scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            
        self.is_fitted = False
        
    def fit(self, train_df: pd.DataFrame):
        """Fits scalers on the training dataframe."""
        self.feature_scaler.fit(train_df[self.feature_cols].values)
        self.target_scaler.fit(train_df[[self.target_col]].values)
        self.is_fitted = True
        return self
        
    def transform_and_create_windows(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Transforms features and builds 3D sliding window tensors (N, lookback, features) and targets.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transforming data.")
            
        feat_scaled = self.feature_scaler.transform(df[self.feature_cols].values)
        target_scaled = self.target_scaler.transform(df[[self.target_col]].values).flatten()
        
        X, y, target_timestamps = [], [], []
        n_samples = len(df)
        
        for i in range(self.lookback_window, n_samples):
            X.append(feat_scaled[i - self.lookback_window : i])
            y.append(target_scaled[i])
            target_timestamps.append(df.index[i])
            
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            pd.DatetimeIndex(target_timestamps)
        )
        
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverses target predictions to original unit scale (delay in minutes)."""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()


class LSTMDelayModel(nn.Module):
    """
    Stacked Multi-Layer LSTM with Dropout and Dense Projection Head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take representation from last time step
        last_step = lstm_out[:, -1, :]
        normed = self.layer_norm(last_step)
        dropped = self.dropout(normed)
        out = self.fc1(dropped)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GRUDelayModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) Model for Sequential Delay Modeling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        gru_out, hn = self.gru(x)
        last_step = gru_out[:, -1, :]
        normed = self.layer_norm(last_step)
        dropped = self.dropout(normed)
        out = self.fc1(dropped)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DeepLearningTrainer:
    """
    Training harness for PyTorch Sequence Models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
            
        return total_loss / len(dataloader.dataset)
        
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(dataloader.dataset)
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                break
                
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        return history
        
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch)
                predictions.append(preds.cpu().numpy())
        return np.vstack(predictions).flatten()
