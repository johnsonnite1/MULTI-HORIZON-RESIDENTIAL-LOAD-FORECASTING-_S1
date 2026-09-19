from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cchs")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260824
HORIZONS = (1, 2, 4, 8)  # 15, 30, 60 and 120 minutes after resampling
APPLIANCES = ("Fan", "PC", "AC", "Lamp", "TV")
METHOD_ORDER = (
    "Persistence",
    "Daily seasonal",
    "Ridge",
    "Extra Trees",
    "Direct HGB",
    "Bottom-up HGB",
    "MinT HGB",
)
COLORS = {
    "Persistence": "#6B7280",
    "Daily seasonal": "#9CA3AF",
    "Ridge": "#60A5FA",
    "Extra Trees": "#2563EB",
    "Direct HGB": "#F59E0B",
    "Bottom-up HGB": "#10B981",
    "MinT HGB": "#0F766E",
    "CCHS": "#B91C1C",
    "CCHS without bottom-up": "#7C3AED",
}


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    results: Path
    figures: Path
    predictions: Path


def resolve_paths() -> Paths:
    script = Path(__file__).resolve()
    study_root = script.parents[2]
    data = study_root / "reproducibility" / "data" / "raw" / "residential_energy_1min.csv"
    results = study_root / "reproducibility" / "results"
    figures = study_root / "reproducibility" / "figures"
    predictions = results / "predictions"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    return Paths(study_root, data, results, figures, predictions)


def load_and_resample(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], dayfirst=True, errors="raise")
    raw = raw.sort_values("Timestamp").set_index("Timestamp")
    all_cols = list(APPLIANCES) + ["Total_Power"]

    component_sum = raw[list(APPLIANCES)].sum(axis=1)
    balance_error = raw["Total_Power"] - component_sum

    # Values are treated as interval energy in kWh per minute. Fifteen successive
    # interval energies are summed and divided by 0.25 h to obtain interval-average kW.
    energy_15_kwh = raw[all_cols].resample("15min").sum()
    power_15_kw = energy_15_kwh / 0.25
    power_15_kw.index.name = "Timestamp"

    audit = {
        "raw_rows": int(len(raw)),
        "resampled_rows": int(len(power_15_kw)),
        "raw_start": raw.index.min().isoformat(),
        "raw_end": raw.index.max().isoformat(),
        "missing_values": int(raw[all_cols].isna().sum().sum()),
        "duplicate_timestamps": int(raw.index.duplicated().sum()),
        "non_one_minute_steps": int((raw.index.to_series().diff().dropna().dt.total_seconds() != 60).sum()),
        "max_abs_balance_error_raw": float(balance_error.abs().max()),
        "all_appliance_values_strictly_positive": bool((raw[list(APPLIANCES)] > 0).all().all()),
        "interpretation": "Input values interpreted as kWh per one-minute interval; resampled values are interval-average kW.",
        "mean_total_kw_after_conversion": float(power_15_kw["Total_Power"].mean()),
        "max_total_kw_after_conversion": float(power_15_kw["Total_Power"].max()),
        "mean_daily_energy_kwh": float(raw["Total_Power"].resample("1D").sum().mean()),
    }
    return power_15_kw, audit


def build_features(power: pd.DataFrame) -> pd.DataFrame:
    idx = power.index
    feat = pd.DataFrame(index=idx)
    minute_of_day = idx.hour * 60 + idx.minute
    day_of_week = idx.dayofweek

    feat["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    feat["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    feat["tod2_sin"] = np.sin(4 * np.pi * minute_of_day / 1440)
    feat["tod2_cos"] = np.cos(4 * np.pi * minute_of_day / 1440)
    feat["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    feat["is_weekend"] = (day_of_week >= 5).astype(int)

    lag_steps = (0, 1, 2, 4, 8, 12, 24, 48, 96, 192, 672)
    for col in list(APPLIANCES) + ["Total_Power"]:
        for lag in lag_steps:
            feat[f"{col}_lag_{lag}"] = power[col].shift(lag)

    for window in (4, 8, 16, 32, 96):
        roll = power["Total_Power"].rolling(window=window, min_periods=window)
        feat[f"total_mean_{window}"] = roll.mean()
        feat[f"total_std_{window}"] = roll.std(ddof=0)
        feat[f"total_min_{window}"] = roll.min()
        feat[f"total_max_{window}"] = roll.max()

    feat["total_slope_1"] = power["Total_Power"] - power["Total_Power"].shift(1)
    feat["total_slope_4"] = power["Total_Power"] - power["Total_Power"].shift(4)
    feat["total_daily_change"] = power["Total_Power"] - power["Total_Power"].shift(96)
    return feat


def segment_masks(target_indices: np.ndarray, n_rows: int) -> tuple[dict[str, np.ndarray], dict]:
    train_end = int(n_rows * 0.55)
    val_end = int(n_rows * 0.70)
    cal_end = int(n_rows * 0.85)
    masks = {
        "train": target_indices < train_end,
        "validation": (target_indices >= train_end) & (target_indices < val_end),
        "calibration": (target_indices >= val_end) & (target_indices < cal_end),
        "test": target_indices >= cal_end,
    }
    boundaries = {
        "train_end_index_exclusive": train_end,
        "validation_end_index_exclusive": val_end,
        "calibration_end_index_exclusive": cal_end,
        "test_end_index_exclusive": n_rows,
    }
    return masks, boundaries


def fit_hgb(x: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=280,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=0.08,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(x, y)
    return model


def fit_convex_stack(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_models = pred.shape[1]
    start = np.full(n_models, 1 / n_models)

    def objective(w: np.ndarray) -> float:
        err = y - pred @ w
        return float(np.mean(err**2) + 1e-7 * np.sum(w**2))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(f"Stack optimization failed: {result.message}")
    w = np.clip(result.x, 0, 1)
    return w / w.sum()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    denom = np.maximum(np.abs(y) + np.abs(pred), 1e-8)
    smape = float(np.mean(2 * np.abs(pred - y) / denom) * 100)
    return {
        "RMSE_kW": float(rmse),
        "MAE_kW": float(mae),
        "nRMSE_pct_mean": float(rmse / np.mean(y) * 100),
        "sMAPE_pct": smape,
        "R2": float(r2_score(y, pred)),
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def regime_key(ts: pd.DatetimeIndex) -> np.ndarray:
    six_hour_period = ts.hour // 6
    weekend = (ts.dayofweek >= 5).astype(int)
    return np.array([f"{p}_{w}" for p, w in zip(six_hour_period, weekend)], dtype=object)


def stratified_half_widths(
    cal_y: np.ndarray,
    cal_pred: np.ndarray,
    cal_ts: pd.DatetimeIndex,
    target_ts: pd.DatetimeIndex,
    alpha: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    scores = np.abs(cal_y - cal_pred)
    global_q = conformal_quantile(scores, alpha)
    cal_keys = regime_key(cal_ts)
    target_keys = regime_key(target_ts)
    q_by_group: dict[str, float] = {}
    for key in np.unique(cal_keys):
        group_scores = scores[cal_keys == key]
        q_by_group[str(key)] = (
            conformal_quantile(group_scores, alpha) if len(group_scores) >= 30 else global_q
        )
    widths = np.array([q_by_group.get(str(key), global_q) for key in target_keys])
    return widths, global_q, q_by_group


def interval_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    half_width: np.ndarray,
    alpha: float,
    scale: float,
    ts: pd.DatetimeIndex,
) -> dict[str, float]:
    lower = np.maximum(0.0, pred - half_width)
    upper = pred + half_width
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    penalty_low = (2 / alpha) * (lower - y) * (y < lower)
    penalty_high = (2 / alpha) * (y - upper) * (y > upper)
    winkler = width + penalty_low + penalty_high
    keys = regime_key(ts)
    group_coverages = [float(covered[keys == key].mean()) for key in np.unique(keys)]
    return {
        "PICP_pct": float(covered.mean() * 100),
        "MPIW_kW": float(width.mean()),
        "PINAW_pct": float(width.mean() / scale * 100),
        "Mean_interval_score": float(winkler.mean()),
        "Worst_regime_coverage_pct": float(min(group_coverages) * 100),
        "Best_regime_coverage_pct": float(max(group_coverages) * 100),
    }


def peak_metrics(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    actual = y >= threshold
    forecast = pred >= threshold
    return {
        "Peak_threshold_kW": float(threshold),
        "Precision_pct": float(precision_score(actual, forecast, zero_division=0) * 100),
        "Recall_pct": float(recall_score(actual, forecast, zero_division=0) * 100),
        "F1_pct": float(f1_score(actual, forecast, zero_division=0) * 100),
        "Balanced_accuracy_pct": float(balanced_accuracy_score(actual, forecast) * 100),
        "Peak_MAE_kW": float(mean_absolute_error(y[actual], pred[actual])) if actual.any() else float("nan"),
        "Actual_peak_count": int(actual.sum()),
        "Predicted_peak_count": int(forecast.sum()),
    }


def reconcile_components(component_pred: np.ndarray, total_pred: np.ndarray, mean_shares: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(component_pred, 0.0)
    sums = nonnegative.sum(axis=1)
    reconciled = np.empty_like(nonnegative)
    valid = sums > 1e-12
    reconciled[valid] = nonnegative[valid] * (total_pred[valid] / sums[valid])[:, None]
    reconciled[~valid] = total_pred[~valid, None] * mean_shares[None, :]
    return reconciled


def mint_projection(validation_errors: np.ndarray) -> np.ndarray:
    """Return a shrinkage-MinT projection for [total, five bottom series]."""
    covariance = np.cov(validation_errors, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.40 * covariance + 0.60 * diagonal
    covariance += np.eye(covariance.shape[0]) * 1e-9
    inverse = np.linalg.pinv(covariance)
    summing = np.vstack([np.ones((1, len(APPLIANCES))), np.eye(len(APPLIANCES))])
    middle = np.linalg.pinv(summing.T @ inverse @ summing)
    return summing @ middle @ summing.T @ inverse


def moving_block_bootstrap_mae_difference(
    y: np.ndarray,
    proposed: np.ndarray,
    comparator: np.ndarray,
    block_length: int = 96,
    replications: int = 2000,
) -> tuple[float, float, float]:
    diff = np.abs(proposed - y) - np.abs(comparator - y)
    n = len(diff)
    rng = np.random.default_rng(SEED)
    starts = np.arange(0, n - block_length + 1)
    estimates = np.empty(replications)
    blocks_needed = math.ceil(n / block_length)
    for b in range(replications):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([diff[s : s + block_length] for s in sampled_starts])[:n]
        estimates[b] = sample.mean()
    return float(diff.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def plot_framework(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.25, 1.8, 1.55, 1.35, "1-min appliance\nenergy records", "#E0F2FE"),
        (2.15, 1.8, 1.55, 1.35, "15-min conversion\nand lag features", "#DBEAFE"),
        (4.05, 2.8, 1.65, 1.15, "Direct total\nforecasters", "#FEF3C7"),
        (4.05, 1.0, 1.65, 1.15, "Appliance HGB\nforecasters", "#D1FAE5"),
        (6.15, 1.8, 1.55, 1.35, "Convex stacking\n(validation)", "#FCE7F3"),
        (8.15, 2.8, 1.55, 1.15, "Coherent hierarchy\nreconciliation", "#EDE9FE"),
        (8.15, 1.0, 1.55, 1.15, "Regime-stratified\nconformal intervals", "#FEE2E2"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#334155", linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")

    arrows = [
        ((1.80, 2.48), (2.15, 2.48)),
        ((3.70, 2.48), (4.05, 3.38)),
        ((3.70, 2.48), (4.05, 1.58)),
        ((5.70, 3.38), (6.15, 2.68)),
        ((5.70, 1.58), (6.15, 2.28)),
        ((7.70, 2.68), (8.15, 3.38)),
        ((7.70, 2.28), (8.15, 1.58)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#475569", linewidth=1.2))
    ax.text(5, 4.65, "Coherent Conformal Hierarchical Stacking (CCHS)", ha="center", va="center", fontsize=14, weight="bold", color="#7F1D1D")
    ax.text(5, 0.30, "Strict chronological train / validation / calibration / test protocol", ha="center", color="#475569")
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_data_characteristics(power: pd.DataFrame, path: Path) -> None:
    daily_matrix = power["Total_Power"].to_numpy().reshape(-1, 96)
    duration = np.sort(power["Total_Power"].to_numpy())[::-1]
    exceedance = np.arange(1, len(duration) + 1) / len(duration) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im = axes[0].imshow(daily_matrix, aspect="auto", cmap="magma", interpolation="nearest")
    axes[0].set_title("Fifteen-minute demand heatmap")
    axes[0].set_xlabel("15-minute interval of day")
    axes[0].set_ylabel("Day index")
    axes[0].set_xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"])
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Mean power (kW)")

    axes[1].plot(exceedance, duration, color="#B91C1C", linewidth=1.8)
    axes[1].axvline(10, color="#F59E0B", linestyle="--", linewidth=1, label="Top 10% demand")
    axes[1].set_title("Load-duration curve")
    axes[1].set_xlabel("Time demand is equalled or exceeded (%)")
    axes[1].set_ylabel("Mean power (kW)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(point: pd.DataFrame, path: Path) -> None:
    selected = ["Persistence", "Daily seasonal", "Ridge", "Extra Trees", "Direct HGB", "MinT HGB", "CCHS"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for method in selected:
        part = point[point["Method"] == method].sort_values("Horizon_min")
        ax.plot(part["Horizon_min"], part["RMSE_kW"], marker="o", linewidth=2 if method == "CCHS" else 1.3, label=method, color=COLORS[method])
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("RMSE (kW)")
    ax.set_title("Point-forecast error across horizons")
    ax.set_xticks([15, 30, 60, 120])
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_interval(pred_df: pd.DataFrame, path: Path) -> None:
    # Show a continuous three-day window selected from the middle of the test period.
    start = max(0, len(pred_df) // 2 - 144)
    view = pred_df.iloc[start : start + 288]
    ts = pd.to_datetime(view["Target_Timestamp"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(ts, view["CCHS_Lower_90"], view["CCHS_Upper_90"], color="#FCA5A5", alpha=0.42, label="90% conformal interval")
    ax.plot(ts, view["Actual_kW"], color="#111827", linewidth=1.25, label="Observed")
    ax.plot(ts, view["CCHS_kW"], color="#B91C1C", linewidth=1.1, label="CCHS forecast")
    ax.set_title("Sixty-minute-ahead CCHS forecast with calibrated uncertainty")
    ax.set_ylabel("Mean power (kW)")
    ax.set_xlabel("Target time")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(reliability: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for method, color, marker in [("Global", "#2563EB", "o"), ("Regime-stratified", "#B91C1C", "s")]:
        part = reliability[(reliability["Interval_method"] == method)].groupby("Nominal_coverage_pct", as_index=False)["PICP_pct"].mean()
        ax.plot(part["Nominal_coverage_pct"], part["PICP_pct"], marker=marker, linewidth=1.8, color=color, label=method)
    ax.plot([75, 97], [75, 97], linestyle="--", color="#4B5563", label="Ideal")
    ax.set_xlim(78, 97)
    ax.set_ylim(78, 97)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Prediction-interval reliability across horizons")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    pivot = weights.pivot(index="Horizon_min", columns="Method", values="Weight").fillna(0)
    pivot = pivot[list(METHOD_ORDER)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bottom = np.zeros(len(pivot))
    for method in METHOD_ORDER:
        values = pivot[method].to_numpy()
        ax.bar(pivot.index.astype(str), values, bottom=bottom, label=method, color=COLORS[method])
        bottom += values
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Convex ensemble weight")
    ax.set_title("Validation-learned CCHS weights")
    ax.legend(ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.18), loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_peak(peak: pd.DataFrame, path: Path) -> None:
    methods = ["Persistence", "Daily seasonal", "Direct HGB", "MinT HGB", "CCHS"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)
    for method in methods:
        part = peak[peak["Method"] == method].sort_values("Horizon_min")
        axes[0].plot(part["Horizon_min"], part["Recall_pct"], marker="o", label=method, color=COLORS[method])
        axes[1].plot(part["Horizon_min"], part["F1_pct"], marker="o", label=method, color=COLORS[method])
    axes[0].set_title("Peak recall")
    axes[1].set_title("Peak F1 score")
    for ax in axes:
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel("Score (%)")
        ax.set_xticks([15, 30, 60, 120])
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.03, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_components(component: pd.DataFrame, coherence: pd.DataFrame, path: Path) -> None:
    part = component[component["Horizon_min"] == 60].copy()
    pivot = part.pivot(index="Appliance", columns="Variant", values="RMSE_kW").reindex(APPLIANCES)
    x = np.arange(len(pivot))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(x - width, pivot["Unreconciled bottom-up"], width, label="Base HGB", color="#10B981")
    axes[0].bar(x, pivot["MinT reconciled"], width, label="MinT", color="#0F766E")
    axes[0].bar(x + width, pivot["CCHS reconciled"], width, label="CCHS", color="#7C3AED")
    axes[0].set_xticks(x, pivot.index)
    axes[0].set_ylabel("RMSE (kW)")
    axes[0].set_title("Appliance forecasts at 60 minutes")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    coh = coherence.pivot(index="Horizon_min", columns="Variant", values="Mean_abs_coherence_error_kW")
    axes[1].plot(coh.index, coh["Direct total vs component sum"], marker="o", color="#F59E0B", label="Before reconciliation")
    axes[1].plot(coh.index, coh["MinT HGB reconciled"], marker="o", color="#0F766E", label="MinT reconciliation")
    axes[1].plot(coh.index, coh["CCHS reconciled"], marker="o", color="#7C3AED", label="Final CCHS")
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].set_xlabel("Forecast horizon (minutes)")
    axes[1].set_ylabel("Mean absolute coherence error (kW)")
    axes[1].set_title("Hierarchical coherence")
    axes[1].set_xticks([15, 30, 60, 120])
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    paths = resolve_paths()
    setup_plot_style()
    power, audit = load_and_resample(paths.data)
    features = build_features(power)
    max_lag = 672

    point_rows: list[dict] = []
    interval_rows: list[dict] = []
    reliability_rows: list[dict] = []
    weight_rows: list[dict] = []
    peak_rows: list[dict] = []
    component_rows: list[dict] = []
    coherence_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    split_rows: list[dict] = []
    pred_frames: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        origins = np.arange(max_lag, len(power) - horizon)
        targets = origins + horizon
        valid = ~features.iloc[origins].isna().any(axis=1).to_numpy()
        origins = origins[valid]
        targets = targets[valid]
        x_all = features.iloc[origins]
        y_total = power["Total_Power"].iloc[targets].to_numpy()
        y_components = power[list(APPLIANCES)].iloc[targets].to_numpy()
        target_ts = power.index[targets]

        masks, boundaries = segment_masks(targets, len(power))
        split_counts = {name: int(mask.sum()) for name, mask in masks.items()}
        for split, count in split_counts.items():
            split_ts = target_ts[masks[split]]
            split_rows.append(
                {
                    "Horizon_min": horizon * 15,
                    "Split": split,
                    "Samples": count,
                    "Target_start": split_ts.min().isoformat(),
                    "Target_end": split_ts.max().isoformat(),
                }
            )

        x_train = x_all.iloc[masks["train"]]
        y_train = y_total[masks["train"]]

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
        ridge.fit(x_train, y_train)
        extra = ExtraTreesRegressor(
            n_estimators=320,
            max_features=0.80,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )
        extra.fit(x_train, y_train)
        direct_hgb = fit_hgb(x_train, y_train)

        component_models = []
        for j, _ in enumerate(APPLIANCES):
            model = fit_hgb(x_train, y_components[masks["train"], j])
            component_models.append(model)

        split_predictions: dict[str, dict[str, np.ndarray]] = {}
        split_components: dict[str, np.ndarray] = {}
        split_mint_components: dict[str, np.ndarray] = {}
        for split, mask in masks.items():
            x = x_all.iloc[mask]
            split_origins = origins[mask]
            split_targets = targets[mask]
            persistence = power["Total_Power"].iloc[split_origins].to_numpy()
            seasonal = power["Total_Power"].iloc[split_targets - 96].to_numpy()
            ridge_pred = ridge.predict(x)
            extra_pred = extra.predict(x)
            hgb_pred = direct_hgb.predict(x)
            component_pred = np.column_stack([np.maximum(0, model.predict(x)) for model in component_models])
            bottom_up = component_pred.sum(axis=1)
            split_components[split] = component_pred
            split_predictions[split] = {
                "Persistence": persistence,
                "Daily seasonal": seasonal,
                "Ridge": ridge_pred,
                "Extra Trees": extra_pred,
                "Direct HGB": hgb_pred,
                "Bottom-up HGB": bottom_up,
            }

        validation_actual_all = np.column_stack(
            [y_total[masks["validation"]], y_components[masks["validation"]]]
        )
        validation_base_all = np.column_stack(
            [
                split_predictions["validation"]["Direct HGB"],
                split_components["validation"],
            ]
        )
        mint = mint_projection(validation_actual_all - validation_base_all)
        for split in masks:
            base_all = np.column_stack(
                [split_predictions[split]["Direct HGB"], split_components[split]]
            )
            reconciled_all = base_all @ mint.T
            split_predictions[split]["MinT HGB"] = reconciled_all[:, 0]
            split_mint_components[split] = reconciled_all[:, 1:]

        val_matrix = np.column_stack([split_predictions["validation"][m] for m in METHOD_ORDER])
        val_y = y_total[masks["validation"]]
        weights = fit_convex_stack(val_matrix, val_y)
        for method, weight in zip(METHOD_ORDER, weights):
            weight_rows.append({"Horizon_min": horizon * 15, "Method": method, "Weight": float(weight)})

        no_bu_methods = tuple(
            method for method in METHOD_ORDER if method not in ("Bottom-up HGB", "MinT HGB")
        )
        val_no_bu = np.column_stack([split_predictions["validation"][m] for m in no_bu_methods])
        weights_no_bu = fit_convex_stack(val_no_bu, val_y)

        for split in masks:
            matrix = np.column_stack([split_predictions[split][m] for m in METHOD_ORDER])
            split_predictions[split]["CCHS"] = matrix @ weights
            matrix_no_bu = np.column_stack([split_predictions[split][m] for m in no_bu_methods])
            split_predictions[split]["CCHS without bottom-up"] = matrix_no_bu @ weights_no_bu

        test_mask = masks["test"]
        test_y = y_total[test_mask]
        test_components_y = y_components[test_mask]
        test_ts = target_ts[test_mask]

        for method, pred in split_predictions["test"].items():
            metrics = regression_metrics(test_y, pred)
            point_rows.append({"Horizon_min": horizon * 15, "Method": method, **metrics})

        cchs_test = split_predictions["test"]["CCHS"]
        train_component_means = y_components[masks["train"]].mean(axis=0)
        mean_shares = train_component_means / train_component_means.sum()
        reconciled = reconcile_components(split_mint_components["test"], cchs_test, mean_shares)
        for j, appliance in enumerate(APPLIANCES):
            for variant, pred in [
                ("Unreconciled bottom-up", split_components["test"][:, j]),
                ("MinT reconciled", split_mint_components["test"][:, j]),
                ("CCHS reconciled", reconciled[:, j]),
            ]:
                component_rows.append(
                    {
                        "Horizon_min": horizon * 15,
                        "Appliance": appliance,
                        "Variant": variant,
                        **regression_metrics(test_components_y[:, j], pred),
                    }
                )

        direct_sum_error = np.abs(split_predictions["test"]["Direct HGB"] - split_components["test"].sum(axis=1))
        mint_sum_error = np.abs(split_predictions["test"]["MinT HGB"] - split_mint_components["test"].sum(axis=1))
        reconciled_sum_error = np.abs(cchs_test - reconciled.sum(axis=1))
        coherence_rows.extend(
            [
                {
                    "Horizon_min": horizon * 15,
                    "Variant": "Direct total vs component sum",
                    "Mean_abs_coherence_error_kW": float(direct_sum_error.mean()),
                    "Max_abs_coherence_error_kW": float(direct_sum_error.max()),
                },
                {
                    "Horizon_min": horizon * 15,
                    "Variant": "MinT HGB reconciled",
                    "Mean_abs_coherence_error_kW": float(mint_sum_error.mean()),
                    "Max_abs_coherence_error_kW": float(mint_sum_error.max()),
                },
                {
                    "Horizon_min": horizon * 15,
                    "Variant": "CCHS reconciled",
                    "Mean_abs_coherence_error_kW": float(reconciled_sum_error.mean()),
                    "Max_abs_coherence_error_kW": float(reconciled_sum_error.max()),
                },
            ]
        )

        cal_y = y_total[masks["calibration"]]
        cal_pred = split_predictions["calibration"]["CCHS"]
        cal_ts = target_ts[masks["calibration"]]
        interval_scale = float(np.quantile(y_train, 0.95) - np.quantile(y_train, 0.05))
        primary_lower = primary_upper = None
        for nominal in (0.80, 0.90, 0.95):
            alpha = 1 - nominal
            strat_test_q, global_q, q_by_group = stratified_half_widths(cal_y, cal_pred, cal_ts, test_ts, alpha)
            global_width = np.full(len(test_y), global_q)
            for interval_method, half_width in [("Global", global_width), ("Regime-stratified", strat_test_q)]:
                metrics = interval_metrics(test_y, cchs_test, half_width, alpha, interval_scale, test_ts)
                row = {
                    "Horizon_min": horizon * 15,
                    "Nominal_coverage_pct": nominal * 100,
                    "Interval_method": interval_method,
                    **metrics,
                }
                reliability_rows.append(row)
                if nominal == 0.90:
                    interval_rows.append(row)
            if nominal == 0.90:
                primary_lower = np.maximum(0.0, cchs_test - strat_test_q)
                primary_upper = cchs_test + strat_test_q

        threshold = float(np.quantile(y_train, 0.90))
        peak_methods = ["Persistence", "Daily seasonal", "Direct HGB", "Bottom-up HGB", "MinT HGB", "CCHS"]
        for method in peak_methods:
            peak_rows.append(
                {
                    "Horizon_min": horizon * 15,
                    "Method": method,
                    **peak_metrics(test_y, split_predictions["test"][method], threshold),
                }
            )

        for comparator in ("Persistence", "Daily seasonal", "Direct HGB", "Bottom-up HGB", "MinT HGB"):
            estimate, lower, upper = moving_block_bootstrap_mae_difference(
                test_y, cchs_test, split_predictions["test"][comparator]
            )
            bootstrap_rows.append(
                {
                    "Horizon_min": horizon * 15,
                    "Comparator": comparator,
                    "CCHS_minus_comparator_MAE_kW": estimate,
                    "CI95_lower_kW": lower,
                    "CI95_upper_kW": upper,
                    "CCHS_significantly_better": bool(upper < 0),
                }
            )

        pred_df = pd.DataFrame(
            {
                "Target_Timestamp": test_ts,
                "Actual_kW": test_y,
                **{f"{method}_kW": pred for method, pred in split_predictions["test"].items()},
                "CCHS_Lower_90": primary_lower,
                "CCHS_Upper_90": primary_upper,
            }
        )
        for j, appliance in enumerate(APPLIANCES):
            pred_df[f"Actual_{appliance}_kW"] = test_components_y[:, j]
            pred_df[f"Reconciled_{appliance}_kW"] = reconciled[:, j]
        pred_df.to_csv(paths.predictions / f"test_predictions_h{horizon * 15:03d}min.csv", index=False)
        pred_frames[horizon * 15] = pred_df

    point = pd.DataFrame(point_rows).sort_values(["Horizon_min", "Method"])
    intervals = pd.DataFrame(interval_rows).sort_values(["Horizon_min", "Interval_method"])
    reliability = pd.DataFrame(reliability_rows).sort_values(["Horizon_min", "Nominal_coverage_pct", "Interval_method"])
    weights_df = pd.DataFrame(weight_rows).sort_values(["Horizon_min", "Method"])
    peak_df = pd.DataFrame(peak_rows).sort_values(["Horizon_min", "Method"])
    component_df = pd.DataFrame(component_rows).sort_values(["Horizon_min", "Appliance", "Variant"])
    coherence_df = pd.DataFrame(coherence_rows).sort_values(["Horizon_min", "Variant"])
    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(["Horizon_min", "Comparator"])
    split_df = pd.DataFrame(split_rows).sort_values(["Horizon_min", "Split"])

    point.to_csv(paths.results / "point_forecast_metrics.csv", index=False)
    intervals.to_csv(paths.results / "interval_metrics_90pct.csv", index=False)
    reliability.to_csv(paths.results / "interval_reliability.csv", index=False)
    weights_df.to_csv(paths.results / "ensemble_weights.csv", index=False)
    peak_df.to_csv(paths.results / "peak_event_metrics.csv", index=False)
    component_df.to_csv(paths.results / "component_forecast_metrics.csv", index=False)
    coherence_df.to_csv(paths.results / "coherence_metrics.csv", index=False)
    bootstrap_df.to_csv(paths.results / "bootstrap_mae_comparisons.csv", index=False)
    split_df.to_csv(paths.results / "split_manifest.csv", index=False)
    with (paths.results / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    plot_framework(paths.figures / "Fig01_CCHS_framework.png")
    plot_data_characteristics(power, paths.figures / "Fig02_demand_heatmap_and_duration_curve.png")
    plot_rmse(point, paths.figures / "Fig03_RMSE_across_horizons.png")
    plot_interval(pred_frames[60], paths.figures / "Fig04_forecast_intervals_60min.png")
    plot_reliability(reliability, paths.figures / "Fig05_interval_reliability.png")
    plot_weights(weights_df, paths.figures / "Fig06_ensemble_weights.png")
    plot_peak(peak_df, paths.figures / "Fig07_peak_event_performance.png")
    plot_components(component_df, coherence_df, paths.figures / "Fig08_component_reconciliation.png")

    summary = {
        "method": "Coherent Conformal Hierarchical Stacking (CCHS)",
        "horizons_minutes": [h * 15 for h in HORIZONS],
        "point_results": point[point["Method"] == "CCHS"].to_dict(orient="records"),
        "interval_results_90pct": intervals[intervals["Interval_method"] == "Regime-stratified"].to_dict(orient="records"),
        "peak_results": peak_df[peak_df["Method"] == "CCHS"].to_dict(orient="records"),
        "audit": audit,
        "split_boundaries": boundaries,
    }
    with (paths.results / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
