#!/usr/bin/env python3
"""Quick EDA for per-component runtime and memory modeling.

The script loads a CSV of profiling features, runs simple linear/non-linear
baselines per component, prints correlation tables, and saves scatter/residual
plots for log-transformed runtime.
"""

import argparse
import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


MIN_ROWS = 20
RANDOM_STATE = 0
TEST_SIZE = 0.2
MIN_NONNULL_FRAC = 0.5
TARGETS_TO_MODEL = [
    ("log_runtime_sec", "log runtime", "log_runtime_sec"),
    ("runtime_sec", "runtime", "runtime_sec"),
    (
        "mean_temporal_util_percent",
        "mean temporal util",
        "mean_temporal_util_percent",
    ),
]
PRIORITIZED_FEATURES = [
    "input_batch_size",
    "output_samples",
    "scaffold_length",
    "ligand_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick EDA for runtime/memory vs features per component."
    )
    parser.add_argument(
        "--csv",
        default="modeling/features.csv",
        help="Path to the features CSV file.",
    )
    parser.add_argument(
        "--outdir",
        default="modeling/results/quick_eda",
        help="Directory to write plots (PNG).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_ROWS,
        help="Minimum rows per component to analyze.",
    )
    parser.add_argument(
        "--min-nonnull-frac",
        type=float,
        default=MIN_NONNULL_FRAC,
        help=(
            "Drop feature columns with a non-null fraction below this per component "
            "to avoid discarding rows for irrelevant knobs."
        ),
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["runtime_sec", "peak_memory_mib"])
    df = df[(df["runtime_sec"] > 0) & (df["peak_memory_mib"] > 0)]
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    target_cols = [
        "runtime_sec",
        "peak_memory_mib",
        "mean_temporal_util_percent",
        "log_runtime_sec",
        "log_peak_memory_mib",
    ]
    return [col for col in numeric_cols if col not in target_cols]


def filter_features_by_nonnull(
    df: pd.DataFrame, feature_cols: List[str], min_nonnull_frac: float
) -> List[str]:
    """Keep only features with enough non-null coverage and some variance."""
    kept: List[str] = []
    for col in feature_cols:
        frac = df[col].notna().mean()
        if frac < min_nonnull_frac:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        kept.append(col)
    return kept


def print_correlations(
    df: pd.DataFrame, component: str, feature_cols: List[str]
) -> None:
    if not feature_cols:
        print(f"[{component}] No numeric features to correlate.")
        return

    corr_df = df[feature_cols + ["log_runtime_sec", "log_peak_memory_mib"]].dropna()
    if corr_df.empty:
        print(f"[{component}] Not enough data for correlation after dropping NA.")
        return

    for target in ["log_runtime_sec", "log_peak_memory_mib"]:
        series = corr_df[feature_cols].corrwith(corr_df[target])
        series = series.dropna()
        series = series.reindex(series.abs().sort_values(ascending=False).index)
        print(f"\n=== Component: {component} | Target: {target} correlations ===")
        for feat, val in series.items():
            print(f"{feat:35s} {val:+.3f}")


def choose_plot_features(
    df: pd.DataFrame, feature_cols: List[str], max_plots: int = 6
) -> List[str]:
    selected: List[str] = []
    for feat in PRIORITIZED_FEATURES:
        if feat in feature_cols and df[feat].var(skipna=True) > 0:
            selected.append(feat)

    remaining = [f for f in feature_cols if f not in selected]
    if remaining:
        variances = df[remaining].var(skipna=True)
        candidates = [
            feat for feat in variances.index if pd.notna(variances.loc[feat]) and variances.loc[feat] > 0
        ]
        candidates = sorted(candidates, key=lambda f: variances.loc[f], reverse=True)
        for feat in candidates:
            if len(selected) >= max_plots:
                break
            selected.append(feat)

    return selected[:max_plots]


def interpret_models(r2_ridge: float, r2_rf: float) -> str:
    diff = r2_rf - r2_ridge
    if diff > 0.1:
        return "Non-linear relationships or interactions likely matter (Random Forest wins)."
    if diff < -0.1:
        return "Linear model outperforms; non-linear additions may not be necessary."
    return "Scores are similar; linear model may already capture most structure."


def plot_scatter_features(
    df: pd.DataFrame, component: str, features: List[str], outdir: str
) -> None:
    if not features:
        print(f"[{component}] Skipping scatter plot: no features selected.")
        return

    n_feats = len(features)
    cols = 3 if n_feats > 4 else 2 if n_feats > 1 else 1
    rows = math.ceil(n_feats / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for idx, feat in enumerate(features):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.scatter(df[feat], df["log_runtime_sec"], alpha=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("log_runtime_sec")
        ax.set_title(f"{component} vs {feat}")

    # Hide any unused subplots
    total_axes = rows * cols
    if total_axes > n_feats:
        for idx in range(n_feats, total_axes):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis("off")

    plt.tight_layout()
    outfile = os.path.join(outdir, f"{component}_scatter_log_runtime.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[{component}] Saved scatter plot to {outfile}")


def plot_residuals(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    component: str,
    features_for_resid: List[str],
    outdir: str,
    filename: str,
    y_label: str,
) -> None:
    residuals = y_test - y_pred
    extra_feats = [f for f in features_for_resid if f in X_test.columns][:2]
    n_plots = 1 + len(extra_feats)
    cols = 2 if n_plots > 1 else 1
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes_flat = axes.ravel()

    axes_flat[0].scatter(y_pred, residuals, alpha=0.6)
    axes_flat[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes_flat[0].set_xlabel(f"Predicted {y_label} (Ridge)")
    axes_flat[0].set_ylabel("Residuals")
    axes_flat[0].set_title(f"{component} residuals vs prediction")

    for idx, feat in enumerate(extra_feats, start=1):
        axes_flat[idx].scatter(X_test[feat], residuals, alpha=0.6)
        axes_flat[idx].axhline(0, color="red", linestyle="--", linewidth=1)
        axes_flat[idx].set_xlabel(feat)
        axes_flat[idx].set_ylabel("Residuals")
        axes_flat[idx].set_title(f"{component} residuals vs {feat}")

    if n_plots < len(axes_flat):
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].axis("off")

    plt.tight_layout()
    outfile = filename
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[{component}] Saved residual plot to {outfile}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.csv)
    components = sorted(df["component"].dropna().unique())
    print(f"Loaded {len(df)} rows across {len(components)} components.")

    for component in components:
        df_comp = df[df["component"] == component].copy()
        if len(df_comp) < args.min_rows:
            print(
                f"[{component}] Skipping: only {len(df_comp)} rows (< {args.min_rows})."
            )
            continue

        df_comp["log_runtime_sec"] = np.log(df_comp["runtime_sec"])
        df_comp["log_peak_memory_mib"] = np.log(df_comp["peak_memory_mib"])

        feature_cols_raw = select_feature_columns(df_comp)
        feature_cols = filter_features_by_nonnull(
            df_comp, feature_cols_raw, min_nonnull_frac=args.min_nonnull_frac
        )
        if not feature_cols:
            print(f"[{component}] Skipping: no numeric features after filtering.")
            continue

        before_drop = len(df_comp)
        df_comp = df_comp.dropna(subset=feature_cols)
        after_drop = len(df_comp)
        if after_drop < args.min_rows:
            print(
                f"[{component}] Skipping: insufficient rows ({after_drop}) after dropping NA "
                f"(started with {before_drop}, features kept: {len(feature_cols)})."
            )
            continue

        print_correlations(df_comp, component, feature_cols)

        # Model each target separately to avoid dropping rows for unrelated targets
        print(f"\n[{component}] Samples after feature NA drop: {len(df_comp)}")
        plot_features = choose_plot_features(df_comp, feature_cols, max_plots=6)
        plot_scatter_features(df_comp, component, plot_features, args.outdir)

        for target_col, target_label, file_suffix in TARGETS_TO_MODEL:
            if target_col not in df_comp.columns:
                print(f"[{component}] Skipping target {target_col}: column missing.")
                continue

            df_target = df_comp.dropna(subset=[target_col])
            if len(df_target) < args.min_rows:
                print(
                    f"[{component}] Skipping target {target_col}: only {len(df_target)} rows."
                )
                continue

            X = df_target[feature_cols]
            y = df_target[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            r2_ridge = ridge.score(X_test, y_test)

            rf = RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            r2_rf = rf.score(X_test, y_test)

            interpretation = interpret_models(r2_ridge, r2_rf)
            print(
                f"[{component}] Target: {target_label} | rows: {len(df_target)} "
                f"| Ridge R^2: {r2_ridge:.3f} | RandomForest R^2: {r2_rf:.3f}"
            )
            print(f"[{component}] Target {target_label} interpretation: {interpretation}")

            y_pred_lin = ridge.predict(X_test)
            residual_suffix = (
                "residuals.png" if target_col == "log_runtime_sec" else f"residuals_{file_suffix}.png"
            )
            plot_residuals(
                X_test,
                y_test,
                y_pred_lin,
                component,
                plot_features,
                args.outdir,
                filename=os.path.join(args.outdir, f"{component}_{residual_suffix}"),
                y_label=target_col,
            )


if __name__ == "__main__":
    main()
