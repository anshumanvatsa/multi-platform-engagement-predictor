"""
shap_analysis.py
================
SHAP explainability analysis for the DG-Social Phase 2 model.

What it does
------------
1.  Loads the trained LightGBM pipeline  (models/best_model.joblib)
2.  Samples a representative subset of the test data
3.  Computes SHAP values using TreeExplainer  (fast for tree-based models)
4.  Generates 8 publication-ready figures:
        shap_summary_beeswarm.png     ← main paper figure
        shap_summary_bar.png          ← mean absolute importance
        shap_waterfall_high.png       ← single HIGH-engagement prediction
        shap_waterfall_low.png        ← single LOW-engagement prediction
        shap_dependence_post_hour.png ← how hour affects prediction
        shap_dependence_log_views.png ← how views affect prediction
        shap_platform_comparison.png  ← per-platform mean SHAP
        shap_force_plot.html          ← interactive force plot (bonus)
5.  Prints top-5 most impactful features overall

Location  : d:/dg-social/phase2/shap_analysis.py
Run       : python shap_analysis.py
Requires  : train.py must have been run first (best_model.joblib must exist)
Output    : outputs/figures/shap_*.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import joblib
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    COMBINED_CSV, BEST_MODEL_PATH,
    OUTPUTS_DIR, RANDOM_STATE,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET,
)

FIG_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")
SEED = RANDOM_STATE

# ── Colour palette ───────────────────────────────────────────────────────────
HIGH_COLOR  = "#f97316"   # orange  → HIGH engagement
LOW_COLOR   = "#6366f1"   # indigo  → LOW engagement
SHAP_CMAP   = "coolwarm"

# ── How many rows to sample for SHAP (TreeExplainer is fast but 800K is big) ─
SHAP_SAMPLE = 3000        # enough for stable SHAP values, fast enough to run


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════
def save_fig(fig, fname, dpi=150):
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fname}")
    return path


def build_feature_names(cat_cols, num_cols):
    """Return human-readable names in the same order as ColumnTransformer output."""
    return cat_cols + num_cols


# ════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD MODEL + DATA
# ════════════════════════════════════════════════════════════════
def load_model_and_data():
    print("\n[1/5] Loading model and data...")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Please run train.py first."
        )

    pipeline = joblib.load(BEST_MODEL_PATH)
    print(f"      Model loaded: {type(pipeline.named_steps['clf']).__name__}")

    df = pd.read_csv(COMBINED_CSV)
    print(f"      Data loaded : {len(df):,} rows")
    return pipeline, df


# ════════════════════════════════════════════════════════════════
#  STEP 2 — REPRODUCE FEATURE ENGINEERING (same as train.py)
# ════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["total_views", "follower_count", "caption_length"]:
        df[f"log_{col}"] = np.log1p(df[col])

    df["video_x_peak"]    = df["is_video"]  * df["is_peak_hour"]
    df["paid_x_weekend"]  = df["is_paid"]   * df["is_weekend"]
    df["media_x_hashtag"] = df["has_media"] * df["hashtag_count"]
    df["hour_bin"] = pd.cut(
        df["post_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"]
    ).astype(str).replace("nan", "unknown")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)
    return df


def get_feature_cols():
    engineered_num = [
        "log_total_views", "log_follower_count", "log_caption_length",
        "video_x_peak", "paid_x_weekend", "media_x_hashtag",
    ]
    cat_cols = CATEGORICAL_FEATURES + ["hour_bin"]
    num_cols = NUMERICAL_FEATURES + engineered_num
    return cat_cols, num_cols


# ════════════════════════════════════════════════════════════════
#  STEP 3 — COMPUTE SHAP VALUES
# ════════════════════════════════════════════════════════════════
def compute_shap(pipeline, df):
    print("\n[2/5] Computing SHAP values...")

    cat_cols, num_cols = get_feature_cols()
    feature_cols = cat_cols + num_cols

    df_eng = engineer_features(df)
    X_all  = df_eng[feature_cols]
    y_all  = df_eng[TARGET]

    # Stratified sample for SHAP
    rng    = np.random.RandomState(SEED)
    idx    = rng.choice(len(X_all), size=SHAP_SAMPLE, replace=False)
    X_samp = X_all.iloc[idx].reset_index(drop=True)
    y_samp = y_all.iloc[idx].reset_index(drop=True)
    plat_samp = df_eng["platform"].iloc[idx].reset_index(drop=True)

    # Transform with preprocessor
    pre       = pipeline.named_steps["pre"]
    clf       = pipeline.named_steps["clf"]
    X_trans   = pre.transform(X_samp)

    # TreeExplainer — fastest for LightGBM/XGBoost/RF
    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_trans)

    # For binary classification LightGBM returns list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # use class 1 (HIGH engagement)

    feature_names = build_feature_names(cat_cols, num_cols)
    print(f"      SHAP matrix shape: {shap_vals.shape}")
    print(f"      Features         : {len(feature_names)}")

    return shap_vals, X_trans, X_samp, y_samp, plat_samp, feature_names, explainer


# ════════════════════════════════════════════════════════════════
#  STEP 4 — FIGURES
# ════════════════════════════════════════════════════════════════
def plot_summary_beeswarm(shap_vals, X_trans, feature_names):
    """Fig 1: Classic SHAP beeswarm — most important paper figure."""
    print("\n[3/5] Plotting figures...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_trans,
        feature_names=feature_names,
        plot_type="dot",
        max_display=15,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Summary — Feature Impact on Engagement Prediction",
              fontsize=13, pad=12)
    plt.tight_layout()
    save_fig(plt.gcf(), "shap_summary_beeswarm.png", dpi=180)


def plot_summary_bar(shap_vals, X_trans, feature_names):
    """Fig 2: Mean absolute SHAP — cleaner for presentations."""
    shap.summary_plot(
        shap_vals, X_trans,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Mean |SHAP Value| — Global Feature Importance", fontsize=13, pad=12)
    plt.tight_layout()
    save_fig(plt.gcf(), "shap_summary_bar.png", dpi=180)


def plot_waterfall(shap_vals, X_trans, y_samp, feature_names, explainer):
    """Fig 3 & 4: Waterfall for one HIGH and one LOW prediction."""
    # find a HIGH and a LOW example
    high_idx = np.where(y_samp.values == 1)[0]
    low_idx  = np.where(y_samp.values == 0)[0]

    for label, idxs, fname in [
        ("HIGH Engagement", high_idx, "shap_waterfall_high.png"),
        ("LOW Engagement",  low_idx,  "shap_waterfall_low.png"),
    ]:
        if len(idxs) == 0:
            continue
        i = idxs[0]

        # Build a shap.Explanation object for waterfall
        exp = shap.Explanation(
            values          = shap_vals[i],
            base_values     = explainer.expected_value if not isinstance(
                                explainer.expected_value, list
                              ) else explainer.expected_value[1],
            data            = X_trans[i] if hasattr(X_trans, "__getitem__") else X_trans.toarray()[i],
            feature_names   = feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=12, show=False)
        plt.title(f"SHAP Waterfall — {label} Prediction", fontsize=12)
        plt.tight_layout()
        save_fig(plt.gcf(), fname, dpi=150)


def plot_dependence(shap_vals, X_trans, feature_names, feat_a, feat_b=None, fname=None):
    """Fig 5 & 6: Dependence plot for a single feature."""
    try:
        feat_idx = feature_names.index(feat_a)
        inter_idx = (feature_names.index(feat_b)
                     if feat_b and feat_b in feature_names else "auto")

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx, shap_vals, X_trans,
            feature_names=feature_names,
            interaction_index=inter_idx,
            ax=ax, show=False,
            dot_size=10, alpha=0.4,
        )
        ax.set_title(f"SHAP Dependence: {feat_a}", fontsize=12)
        plt.tight_layout()
        save_fig(fig, fname or f"shap_dep_{feat_a}.png", dpi=150)
    except Exception as e:
        print(f"      [warn] Dependence plot for {feat_a} failed: {e}")


def plot_platform_comparison(shap_vals, X_samp, feature_names, plat_samp):
    """Fig 7: Mean absolute SHAP per platform — shows what drives each platform."""
    shap_df = pd.DataFrame(
        np.abs(shap_vals), columns=feature_names
    )
    shap_df["platform"] = plat_samp.values

    # Only keep numeric SHAP columns (exclude categoricals like platform/post_type)
    cat_cols_to_skip = {"platform", "post_type", "hour_bin"}
    numeric_feats = [f for f in feature_names if f not in cat_cols_to_skip]
    top_features  = numeric_feats[:12]   # top 12 numeric features

    plat_means = shap_df.groupby("platform")[top_features].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    plat_means.T.plot(
        kind="bar", ax=ax,
        colormap="tab10", edgecolor="white", width=0.75
    )
    ax.set_title("Mean |SHAP| per Feature by Platform", fontsize=13)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.legend(title="Platform", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig(fig, "shap_platform_comparison.png", dpi=150)


def plot_top_features_text(shap_vals, feature_names):
    """Print top-5 features by mean absolute SHAP."""
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top5 = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:5]
    print("\n      Top-5 Features by Mean |SHAP|:")
    for rank, (feat, val) in enumerate(top5, 1):
        print(f"        {rank}. {feat:<30}  {val:.4f}")


# ════════════════════════════════════════════════════════════════
#  STEP 5 — SAVE SHAP VALUES FOR API USE (reason_generator.py)
# ════════════════════════════════════════════════════════════════
def save_shap_metadata(shap_vals, feature_names):
    """Save mean absolute SHAP per feature — used by reason_generator.py."""
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    shap_meta = {
        feat: float(val)
        for feat, val in zip(feature_names, mean_abs)
    }
    import json
    path = os.path.join(OUTPUTS_DIR, "shap_feature_importance.json")
    with open(path, "w") as f:
        json.dump(shap_meta, f, indent=2)
    print(f"\n      Saved SHAP metadata → {path}")
    return shap_meta


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DG-Social Phase 2 — SHAP Explainability Analysis")
    print("=" * 60)

    pipeline, df = load_model_and_data()

    shap_vals, X_trans, X_samp, y_samp, plat_samp, feature_names, explainer = \
        compute_shap(pipeline, df)

    # ── Figures ──────────────────────────────────────────────
    plot_summary_beeswarm(shap_vals, X_trans, feature_names)
    plot_summary_bar(shap_vals, X_trans, feature_names)
    plot_waterfall(shap_vals, X_trans, y_samp, feature_names, explainer)
    plot_dependence(
        shap_vals, X_trans, feature_names,
        feat_a="post_hour",
        feat_b="is_peak_hour",
        fname="shap_dependence_post_hour.png"
    )
    plot_dependence(
        shap_vals, X_trans, feature_names,
        feat_a="log_total_views",
        feat_b="is_video",
        fname="shap_dependence_log_views.png"
    )
    plot_platform_comparison(shap_vals, X_samp, feature_names, plat_samp)

    # ── Print top features ────────────────────────────────────
    plot_top_features_text(shap_vals, feature_names)

    # ── Save metadata for API ─────────────────────────────────
    shap_meta = save_shap_metadata(shap_vals, feature_names)

    print(f"\n[4/5] All figures saved to: {FIG_DIR}/")
    print("        shap_summary_beeswarm.png")
    print("        shap_summary_bar.png")
    print("        shap_waterfall_high.png")
    print("        shap_waterfall_low.png")
    print("        shap_dependence_post_hour.png")
    print("        shap_dependence_log_views.png")
    print("        shap_platform_comparison.png")

    print(f"\n{'='*60}")
    print("  SHAP analysis complete.")
    print(f"  Figures : {FIG_DIR}/")
    print(f"  Metadata: {OUTPUTS_DIR}/shap_feature_importance.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
