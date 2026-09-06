#!/usr/bin/env python3
"""Simple TWSE index tracker using LSTM or RNN.

Features:
- Fetch TWSE (^TWII) daily data from Yahoo Finance or load from CSV.
- Build sliding-window sequences for multi-step forecasting.
- Train either LSTM or SimpleRNN (RNN) in TensorFlow/Keras.
- Optional grid search for model hyperparameters.
- Compare against persistence and MA5 baselines.
- Export model, metrics, predictions, comparison report, and optional plot.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TWSE tracker with LSTM or RNN.")
    parser.add_argument("--csv", type=str, default=None, help="Optional local CSV path.")
    parser.add_argument("--symbol", type=str, default="^TWII", help="Yahoo ticker, default ^TWII.")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "rnn"],
        default="lstm",
        help="Model type: lstm or rnn.",
    )
    parser.add_argument("--lookback", type=int, default=30, help="Sequence length in days.")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden units for RNN/LSTM.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grid-search", action="store_true", help="Enable hyperparameter grid search.")
    parser.add_argument(
        "--grid-model-types",
        type=str,
        default="lstm,rnn",
        help="Comma-separated model types for grid search.",
    )
    parser.add_argument(
        "--grid-lookbacks",
        type=str,
        default="20,30,60",
        help="Comma-separated lookback values for grid search.",
    )
    parser.add_argument(
        "--grid-hiddens",
        type=str,
        default="32,64",
        help="Comma-separated hidden-unit values for grid search.",
    )
    parser.add_argument(
        "--grid-lrs",
        type=str,
        default="0.001,0.0005",
        help="Comma-separated learning rates for grid search.",
    )
    parser.add_argument(
        "--grid-dropouts",
        type=str,
        default="0.0,0.1",
        help="Comma-separated dropout values for grid search.",
    )
    parser.add_argument(
        "--grid-max-runs",
        type=int,
        default=24,
        help="Maximum number of grid trials to run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="twse_tracker_output",
        help="Output directory.",
    )
    parser.add_argument("--plot", action="store_true", help="Save prediction plot.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_list_int(values: str) -> List[int]:
    return [int(x.strip()) for x in values.split(",") if x.strip()]


def parse_list_float(values: str) -> List[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def parse_list_str(values: str) -> List[str]:
    return [x.strip().lower() for x in values.split(",") if x.strip()]


def load_market_data(
    csv_path: str | None,
    symbol: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for online download. Install with: pip install yfinance"
            ) from exc

        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()

    # Normalize common column names.
    col_map = {c.lower().strip(): c for c in df.columns}
    date_col = col_map.get("date")
    close_col = col_map.get("close")

    if date_col is None or close_col is None:
        raise ValueError("Input data must contain Date and Close columns.")

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in col_map:
            if c == "volume":
                df["Volume"] = 0.0
                col_map["volume"] = "Volume"
            else:
                raise ValueError(f"Input data missing required column: {c}")

    use_cols = [
        col_map["date"],
        col_map["open"],
        col_map["high"],
        col_map["low"],
        col_map["close"],
        col_map["volume"],
    ]
    data = df[use_cols].copy()
    data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data = data.dropna()
    if len(data) < 200:
        raise ValueError("Not enough rows. Need at least 200 rows for stable training/testing.")
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Ret1"] = df["Close"].pct_change()
    df["HLRange"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["OCMove"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    # A missing/zero-volume previous day has no defined percentage change.
    # Encode the change as zero and retain the raw Volume feature.
    previous = df['Volume'].shift(1)
    df['VolChg'] = ((df['Volume'] - previous) / previous.where(previous > 0)).fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def split_index(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def fit_standardizer(x_train_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train_2d.mean(axis=0)
    sigma = x_train_2d.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def standardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lookback < 1 or horizon < 1 or len(features) != len(target):
        raise ValueError("lookback/horizon must be positive and feature/target lengths equal")
    xs = []
    ys = []
    target_starts = []
    for i in range(lookback, len(features) - horizon + 1):
        xs.append(features[i - lookback : i])
        ys.append(target[i : i + horizon])
        target_starts.append(i)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(target_starts, dtype=np.int32),
    )


def sequence_split_masks(target_starts, horizon, train_end, val_end):
    """Assign only labels wholly contained within one chronological split."""
    if horizon < 1 or not 0 < train_end < val_end:
        raise ValueError('invalid horizon or chronological split bounds')
    starts = np.asarray(target_starts)
    ends = starts + horizon
    return (ends <= train_end,
            (starts >= train_end) & (ends <= val_end),
            starts >= val_end)


def build_model(
    model_type: str,
    lookback: int,
    n_features: int,
    hidden: int,
    dropout: float,
    lr: float,
    horizon: int,
):
    model = Sequential()
    if model_type == "lstm":
        model.add(LSTM(hidden, input_shape=(lookback, n_features)))
    else:
        model.add(SimpleRNN(hidden, input_shape=(lookback, n_features)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(horizon))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def inverse_scale_close(y_scaled: np.ndarray, y_mu: float, y_sigma: float) -> np.ndarray:
    return y_scaled * y_sigma + y_mu


def direction_accuracy_with_prev(y_true_h1: np.ndarray, y_pred_h1: np.ndarray, prev_close: np.ndarray) -> float:
    if len(y_true_h1) == 0:
        return float("nan")
    true_diff = np.sign(y_true_h1 - prev_close)
    pred_diff = np.sign(y_pred_h1 - prev_close)
    return float((true_diff == pred_diff).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rmse_all"] = rmse(y_true, y_pred)
    out["mae_all"] = float(np.mean(np.abs(y_true - y_pred)))

    horizon = y_true.shape[1]
    rmse_by_h = []
    mae_by_h = []
    for h in range(horizon):
        rmse_by_h.append(rmse(y_true[:, h], y_pred[:, h]))
        mae_by_h.append(float(np.mean(np.abs(y_true[:, h] - y_pred[:, h]))))
    out["rmse_by_horizon"] = rmse_by_h
    out["mae_by_horizon"] = mae_by_h
    out["direction_accuracy_h1"] = direction_accuracy_with_prev(y_true[:, 0], y_pred[:, 0], prev_close)
    return out


def build_baseline_predictions(
    close_values: np.ndarray,
    target_start_idx: np.ndarray,
    horizon: int,
) -> Dict[str, np.ndarray]:
    persistence = np.zeros((len(target_start_idx), horizon), dtype=np.float32)
    ma5 = np.zeros((len(target_start_idx), horizon), dtype=np.float32)

    for row, i in enumerate(target_start_idx):
        last_close = close_values[i - 1]
        persistence[row, :] = last_close

        left = max(0, i - 5)
        ma5_val = float(np.mean(close_values[left:i]))
        ma5[row, :] = ma5_val

    return {
        "persistence": persistence,
        "ma5": ma5,
    }


def run_one_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    model_type: str,
    lookback: int,
    hidden: int,
    dropout: float,
    lr: float,
    horizon: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    model = build_model(
        model_type=model_type,
        lookback=lookback,
        n_features=x_train.shape[-1],
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        horizon=horizon,
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    val_pred = model.predict(x_val, verbose=0)
    test_pred = model.predict(x_test, verbose=0)
    return model, hist, val_pred, test_pred


def format_grid_trials(trials: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(trials)


def save_integrated_comparison_plot(
    outdir: Path,
    pred_df: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_label: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    ax0 = axes[0]
    ax0.plot(pred_df["Date"], pred_df["TrueClose_t+1"], label="True t+1", lw=2.0)
    ax0.plot(pred_df["Date"], pred_df["PredClose_t+1"], label=f"{model_label} Pred t+1", lw=1.4)
    if "persistence_pred_t+1" in baseline_pred_df.columns:
        ax0.plot(
            baseline_pred_df["Date"],
            baseline_pred_df["persistence_pred_t+1"],
            label="Persistence t+1",
            lw=1.2,
            alpha=0.9,
        )
    if "ma5_pred_t+1" in baseline_pred_df.columns:
        ax0.plot(
            baseline_pred_df["Date"],
            baseline_pred_df["ma5_pred_t+1"],
            label="MA5 t+1",
            lw=1.2,
            alpha=0.9,
        )
    ax0.set_title("TWSE Integrated Comparison (t+1)")
    ax0.set_xlabel("Date")
    ax0.set_ylabel("Index")
    ax0.legend(loc="best")

    ax1 = axes[1]
    labels = comparison_df["model"].astype(str).tolist()
    x = np.arange(len(labels))
    rmse_vals = comparison_df["rmse_h1"].to_numpy(dtype=float)
    mae_vals = comparison_df["mae_h1"].to_numpy(dtype=float)
    acc_vals = comparison_df["direction_accuracy_h1"].to_numpy(dtype=float)

    width = 0.35
    ax1.bar(x - width / 2, rmse_vals, width=width, label="RMSE h1")
    ax1.bar(x + width / 2, mae_vals, width=width, label="MAE h1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel("Error")

    ax1r = ax1.twinx()
    ax1r.plot(x, acc_vals, marker="o", linewidth=1.6, label="Direction Acc h1")
    ax1r.set_ylim(0.0, 1.0)
    ax1r.set_ylabel("Direction Accuracy")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.set_title("Model vs Baselines Metrics")

    fig.tight_layout()
    out_path = outdir / "integrated_comparison_plot.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_market_data(args.csv, args.symbol, args.start, args.end)
    df = add_features(raw)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Ret1",
        "HLRange",
        "OCMove",
        "MA5",
        "MA20",
        "STD20",
        "VolChg",
    ]

    if len(df) < 100:
        raise ValueError("Too few finite feature rows after preprocessing")

    all_features = df[feature_cols].to_numpy(dtype=np.float32)
    all_close = df["Close"].to_numpy(dtype=np.float32)
    all_dates = df["Date"].to_numpy()

    train_end, val_end = split_index(len(df), 0.7, 0.15)

    feat_mu, feat_sigma = fit_standardizer(all_features[:train_end])
    tar_mu = float(all_close[:train_end].mean())
    tar_sigma = float(all_close[:train_end].std())
    if tar_sigma < 1e-8:
        tar_sigma = 1.0

    features_scaled = standardize(all_features, feat_mu, feat_sigma)
    target_scaled = (all_close - tar_mu) / tar_sigma

    x_seq, y_seq, target_start_idx = build_sequences(
        features_scaled,
        target_scaled,
        lookback=args.lookback,
        horizon=args.horizon,
    )

    train_mask, val_mask, test_mask = sequence_split_masks(
        target_start_idx, args.horizon, train_end, val_end)

    x_train, y_train = x_seq[train_mask], y_seq[train_mask]
    x_val, y_val = x_seq[val_mask], y_seq[val_mask]
    x_test, y_test = x_seq[test_mask], y_seq[test_mask]

    idx_test = target_start_idx[test_mask]
    d_test = all_dates[idx_test]
    prev_close_test = all_close[idx_test - 1]

    if len(x_train) < 50 or len(x_test) < 20:
        raise ValueError(
            "Effective sequence samples are too small. Try a shorter lookback or longer data period."
        )

    if len(x_val) < 10:
        raise ValueError("Validation split too small. Use longer data period or smaller lookback/horizon.")

    selected_params = {
        "model_type": args.model_type,
        "lookback": args.lookback,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
    }

    trial_rows: List[Dict[str, Any]] = []

    if args.grid_search:
        grid_model_types = [m for m in parse_list_str(args.grid_model_types) if m in {"lstm", "rnn"}]
        grid_lookbacks = parse_list_int(args.grid_lookbacks)
        grid_hiddens = parse_list_int(args.grid_hiddens)
        grid_lrs = parse_list_float(args.grid_lrs)
        grid_dropouts = parse_list_float(args.grid_dropouts)

        combos = list(itertools.product(grid_model_types, grid_lookbacks, grid_hiddens, grid_lrs, grid_dropouts))
        if len(combos) > args.grid_max_runs:
            combos = combos[: args.grid_max_runs]

        best_key = None
        best_score = float("inf")
        best_model = None
        best_hist = None
        best_test_pred_scaled = None

        for trial_id, (mtype, lb, hid, lr, dr) in enumerate(combos, start=1):
            if lb != args.lookback:
                x_seq_t, y_seq_t, idx_t = build_sequences(
                    features_scaled,
                    target_scaled,
                    lookback=lb,
                    horizon=args.horizon,
                )
                train_m, val_m, test_m = sequence_split_masks(
                    idx_t, args.horizon, train_end, val_end)

                x_train_t, y_train_t = x_seq_t[train_m], y_seq_t[train_m]
                x_val_t, y_val_t = x_seq_t[val_m], y_seq_t[val_m]
                x_test_t = x_seq_t[test_m]
            else:
                x_train_t, y_train_t = x_train, y_train
                x_val_t, y_val_t = x_val, y_val
                x_test_t = x_test

            if len(x_train_t) < 50 or len(x_val_t) < 10 or len(x_test_t) < 20:
                continue

            model_t, hist_t, val_pred_t, test_pred_t = run_one_training(
                x_train=x_train_t,
                y_train=y_train_t,
                x_val=x_val_t,
                y_val=y_val_t,
                x_test=x_test_t,
                model_type=mtype,
                lookback=lb,
                hidden=hid,
                dropout=dr,
                lr=lr,
                horizon=args.horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            val_pred_real = inverse_scale_close(val_pred_t, tar_mu, tar_sigma)
            y_val_real = inverse_scale_close(y_val_t, tar_mu, tar_sigma)
            val_rmse_all = rmse(y_val_real, val_pred_real)
            best_val_loss = float(np.min(hist_t.history.get("val_loss", [math.nan])))

            trial_rows.append(
                {
                    "trial": trial_id,
                    "model_type": mtype,
                    "lookback": lb,
                    "hidden": hid,
                    "lr": lr,
                    "dropout": dr,
                    "val_rmse_all": val_rmse_all,
                    "best_val_loss": best_val_loss,
                    "n_train": len(x_train_t),
                    "n_val": len(x_val_t),
                    "n_test": len(x_test_t),
                }
            )

            if val_rmse_all < best_score:
                best_score = val_rmse_all
                best_key = {
                    "model_type": mtype,
                    "lookback": lb,
                    "hidden": hid,
                    "dropout": dr,
                    "lr": lr,
                }
                best_model = model_t
                best_hist = hist_t
                best_test_pred_scaled = test_pred_t

        if best_key is None or best_model is None or best_hist is None or best_test_pred_scaled is None:
            raise RuntimeError("Grid search found no valid trials. Adjust ranges or data period.")

        selected_params = best_key
        model = best_model
        hist = best_hist
        y_pred_scaled = best_test_pred_scaled

        # Rebuild test references if lookback differs from original args.
        if int(selected_params["lookback"]) != args.lookback:
            best_lb = int(selected_params["lookback"])
            x_seq_b, y_seq_b, idx_b = build_sequences(
                features_scaled,
                target_scaled,
                lookback=best_lb,
                horizon=args.horizon,
            )
            test_mask_b = idx_b >= val_end
            y_test = y_seq_b[test_mask_b]
            idx_test = idx_b[test_mask_b]
            d_test = all_dates[idx_test]
            prev_close_test = all_close[idx_test - 1]
    else:
        model, hist, _val_pred_unused, y_pred_scaled = run_one_training(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            model_type=args.model_type,
            lookback=args.lookback,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    y_pred = inverse_scale_close(y_pred_scaled, tar_mu, tar_sigma)
    y_true = inverse_scale_close(y_test, tar_mu, tar_sigma)

    model_metrics = evaluate_forecasts(y_true, y_pred, prev_close_test)

    baselines = build_baseline_predictions(all_close, idx_test, args.horizon)
    baseline_eval = {
        name: evaluate_forecasts(y_true, pred, prev_close_test) for name, pred in baselines.items()
    }

    comparison_rows = []
    model_label = str(selected_params["model_type"]).upper()
    comparison_rows.append(
        {
            "model": model_label,
            "rmse_all": model_metrics["rmse_all"],
            "mae_all": model_metrics["mae_all"],
            "rmse_h1": model_metrics["rmse_by_horizon"][0],
            "mae_h1": model_metrics["mae_by_horizon"][0],
            "direction_accuracy_h1": model_metrics["direction_accuracy_h1"],
        }
    )
    for baseline_name in ["persistence", "ma5"]:
        m = baseline_eval[baseline_name]
        comparison_rows.append(
            {
                "model": baseline_name,
                "rmse_all": m["rmse_all"],
                "mae_all": m["mae_all"],
                "rmse_h1": m["rmse_by_horizon"][0],
                "mae_h1": m["mae_by_horizon"][0],
                "direction_accuracy_h1": m["direction_accuracy_h1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(outdir / "baseline_comparison.csv", index=False)

    metrics = {
        "selected_params": selected_params,
        "grid_search": bool(args.grid_search),
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "n_train": int(sum(sequence_split_masks(
            np.arange(int(selected_params['lookback']), len(df) - args.horizon + 1),
            args.horizon, train_end, val_end)[0])),
        "n_val": int(sum(sequence_split_masks(
            np.arange(int(selected_params['lookback']), len(df) - args.horizon + 1),
            args.horizon, train_end, val_end)[1])),
        "n_test": int(len(y_true)),
        "test_rmse_all": model_metrics["rmse_all"],
        "test_mae_all": model_metrics["mae_all"],
        "test_rmse_by_horizon": model_metrics["rmse_by_horizon"],
        "test_mae_by_horizon": model_metrics["mae_by_horizon"],
        "test_direction_accuracy_h1": model_metrics["direction_accuracy_h1"],
        "best_val_loss": float(np.min(hist.history.get("val_loss", [math.nan]))),
        "baseline_comparison_csv": str((outdir / "baseline_comparison.csv").resolve()),
    }

    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if trial_rows:
        trial_df = format_grid_trials(trial_rows)
        trial_df = trial_df.sort_values("val_rmse_all", ascending=True).reset_index(drop=True)
        trial_df.to_csv(outdir / "grid_search_results.csv", index=False)

    pred_data: Dict[str, Any] = {"Date": pd.to_datetime(d_test)}
    for h in range(args.horizon):
        step = h + 1
        pred_data[f"TrueClose_t+{step}"] = y_true[:, h]
        pred_data[f"PredClose_t+{step}"] = y_pred[:, h]
        pred_data[f"AbsError_t+{step}"] = np.abs(y_true[:, h] - y_pred[:, h])
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    baseline_pred_df = pd.DataFrame({"Date": pd.to_datetime(d_test)})
    for baseline_name, arr in baselines.items():
        for h in range(args.horizon):
            step = h + 1
            baseline_pred_df[f"{baseline_name}_pred_t+{step}"] = arr[:, h]
    baseline_pred_df.to_csv(outdir / "baseline_predictions.csv", index=False)

    model.save(outdir / "model.keras")
    preprocessing = {
        'schema_version': 1, 'feature_columns': feature_cols,
        'feature_mean': feat_mu.tolist(), 'feature_std': feat_sigma.tolist(),
        'target_mean': tar_mu, 'target_std': tar_sigma,
        'selected_params': selected_params, 'horizon': args.horizon,
        'volume_change_policy': 'zero when previous volume is missing or nonpositive',
        'split_policy': 'complete target horizon within split',
        'train_end': train_end, 'validation_end': val_end,
    }
    (outdir / 'preprocessing.json').write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(11, 4))
            plt.plot(pred_df["Date"], pred_df["TrueClose_t+1"], label="True t+1", lw=1.8)
            plt.plot(pred_df["Date"], pred_df["PredClose_t+1"], label="Pred t+1", lw=1.2)
            plt.title("TWSE Tracker (Selected Model)")
            plt.xlabel("Date")
            plt.ylabel("Index")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "prediction_plot.png", dpi=140)
            plt.close()

            integrated_path = save_integrated_comparison_plot(
                outdir=outdir,
                pred_df=pred_df,
                baseline_pred_df=baseline_pred_df,
                comparison_df=comparison_df,
                model_label=model_label,
            )
            metrics["integrated_comparison_plot"] = str(integrated_path.resolve())
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[Warn] Plot failed: {exc}")

    print("Training done.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()