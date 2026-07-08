"""
train.py
========
Full ML pipeline for DG-Social Phase 2 — Pre-Publication Engagement Predictor.

What it does
------------
1.  Loads data/combined_dataset.csv  (800 K rows, 5 platforms)
2.  Feature engineering  (interaction terms, log transforms)
3.  Builds a preprocessing pipeline  (OrdinalEncoder + StandardScaler)
4.  Trains 5 models with 5-fold stratified cross-validation:
        - Logistic Regression   (baseline)
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - LightGBM
5.  Picks the best model by mean CV F1-score
6.  Evaluates on held-out 20% test set
7.  Saves:
        models/best_model.joblib
        models/preprocessor.joblib
        models/label_encoders.joblib
        outputs/model_results.json
        outputs/figures/  (confusion matrix, ROC, feature importance,
                           platform breakdown, CV comparison)

Location : d:/dg-social/phase2/train.py
Run      : python train.py
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing    import OrdinalEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
)
from sklearn.inspection       import permutation_importance
import joblib

# XGBoost / LightGBM  — optional but strongly recommended
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [warn] xgboost not installed — skipping XGBoost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [warn] lightgbm not installed — skipping LightGBM")

warnings.filterwarnings("ignore")

# ── bring config into scope ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    COMBINED_CSV, MODELS_DIR, OUTPUTS_DIR,
    BEST_MODEL_PATH, PREPROCESSOR_PATH, MODEL_RESULTS_PATH, LABEL_ENCODERS_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET,
)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
SEED = RANDOM_STATE


# ════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ════════════════════════════════════════════════════════════════
def load_data():
    print("\n[1/7] Loading data...")
    df = pd.read_csv(COMBINED_CSV)
    print(f"      Rows: {len(df):,}   Cols: {df.shape[1]}")
    print(f"      Platforms: {df['platform'].value_counts().to_dict()}")
    print(f"      Class balance  0={( df[TARGET]==0).sum():,}  "
          f"1={(df[TARGET]==1).sum():,}")
    return df


# ════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/7] Engineering features...")
    df = df.copy()

    # log-transform heavy-tailed raw counts
    for col in ["total_views", "follower_count", "caption_length"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # interaction: video posted in peak hour
    df["video_x_peak"]    = df["is_video"]    * df["is_peak_hour"]
    # interaction: paid post on weekend
    df["paid_x_weekend"]  = df["is_paid"]     * df["is_weekend"]
    # interaction: media with hashtags
    df["media_x_hashtag"] = df["has_media"]   * df["hashtag_count"]
    # hour bins: morning / afternoon / evening / night
    df["hour_bin"] = pd.cut(
        df["post_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"]
    ).astype(str).replace("nan", "unknown")

    # fill any remaining NaNs introduced by engineering
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    print(f"      Features after engineering: "
          f"{df.shape[1] - 1} (excl. target)")
    return df


# ════════════════════════════════════════════════════════════════
#  STEP 3 — SPLIT
# ════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    print("\n[3/7] Splitting train / test (80/20, stratified)...")

    engineered_num = [
        "log_total_views", "log_follower_count", "log_caption_length",
        "video_x_peak", "paid_x_weekend", "media_x_hashtag",
    ]
    all_num = NUMERICAL_FEATURES + engineered_num
    all_cat = CATEGORICAL_FEATURES + ["hour_bin"]

    feature_cols = all_cat + all_num
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"      Train: {len(X_train):,}   Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, all_cat, all_num


# ════════════════════════════════════════════════════════════════
#  STEP 4 — PREPROCESSOR
# ════════════════════════════════════════════════════════════════
def build_preprocessor(cat_cols, num_cols):
    cat_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, cat_cols),
        ("num", num_pipeline, num_cols),
    ], remainder="drop")
    return preprocessor


# ════════════════════════════════════════════════════════════════
#  STEP 5 — DEFINE MODELS
# ════════════════════════════════════════════════════════════════
def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            random_state=SEED, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=SEED
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=SEED, n_jobs=-1,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    return models


# ════════════════════════════════════════════════════════════════
#  STEP 6 — CROSS-VALIDATE & PICK BEST
# ════════════════════════════════════════════════════════════════
def cross_validate_models(models, preprocessor, X_train, y_train, cat_cols, num_cols):
    print(f"\n[4/7] Cross-validating {len(models)} models ({CV_FOLDS}-fold)...")
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", model),
        ])
        t0  = time.time()
        res = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=["f1", "accuracy", "roc_auc"],
            n_jobs=-1,
            return_train_score=True,
        )
        elapsed = time.time() - t0
        cv_results[name] = {
            "cv_f1_mean":       float(np.mean(res["test_f1"])),
            "cv_f1_std":        float(np.std(res["test_f1"])),
            "cv_acc_mean":      float(np.mean(res["test_accuracy"])),
            "cv_roc_mean":      float(np.mean(res["test_roc_auc"])),
            "train_f1_mean":    float(np.mean(res["train_f1"])),
            "elapsed_sec":      round(elapsed, 2),
        }
        print(f"      {name:<22}  "
              f"F1={cv_results[name]['cv_f1_mean']:.4f} "
              f"(+/-{cv_results[name]['cv_f1_std']:.4f})  "
              f"AUC={cv_results[name]['cv_roc_mean']:.4f}  "
              f"[{elapsed:.1f}s]")

    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_f1_mean"])
    print(f"\n      >>> Best model: {best_name}  "
          f"(CV F1={cv_results[best_name]['cv_f1_mean']:.4f})")
    return cv_results, best_name


# ════════════════════════════════════════════════════════════════
#  STEP 7 — FINAL TRAIN & EVALUATE ON TEST SET
# ════════════════════════════════════════════════════════════════
def final_train_and_evaluate(
    models, best_name, preprocessor,
    X_train, X_test, y_train, y_test
):
    print(f"\n[5/7] Final training: {best_name} on full train set...")
    best_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", models[best_name]),
    ])
    best_pipe.fit(X_train, y_train)

    y_pred     = best_pipe.predict(X_test)
    y_proba    = best_pipe.predict_proba(X_test)[:, 1]

    test_f1    = f1_score(y_test, y_pred)
    test_acc   = accuracy_score(y_test, y_pred)
    test_auc   = roc_auc_score(y_test, y_proba)
    report     = classification_report(y_test, y_pred, output_dict=True)

    print(f"      Test F1       : {test_f1:.4f}")
    print(f"      Test Accuracy : {test_acc:.4f}")
    print(f"      Test AUC-ROC  : {test_auc:.4f}")
    print(f"\n      Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["LOW (0)", "HIGH (1)"]))

    return best_pipe, y_pred, y_proba, {
        "test_f1":   test_f1,
        "test_acc":  test_acc,
        "test_auc":  test_auc,
        "report":    report,
    }


# ════════════════════════════════════════════════════════════════
#  STEP 8 — SAVE ARTEFACTS
# ════════════════════════════════════════════════════════════════
def save_artefacts(best_pipe, preprocessor, cv_results, best_name, test_metrics):
    print("\n[6/7] Saving model artefacts...")

    # save full pipeline (preprocessor + model)
    joblib.dump(best_pipe,   BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"      Saved: {BEST_MODEL_PATH}")
    print(f"      Saved: {PREPROCESSOR_PATH}")

    # save results JSON
    results = {
        "best_model":    best_name,
        "cv_results":    cv_results,
        "test_metrics":  {
            "f1":       round(test_metrics["test_f1"],  4),
            "accuracy": round(test_metrics["test_acc"], 4),
            "auc_roc":  round(test_metrics["test_auc"], 4),
        },
        "classification_report": test_metrics["report"],
    }
    with open(MODEL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved: {MODEL_RESULTS_PATH}")


# ════════════════════════════════════════════════════════════════
#  STEP 9 — FIGURES
# ════════════════════════════════════════════════════════════════
def plot_all(
    best_pipe, best_name, cv_results,
    X_test, y_test, y_pred, y_proba,
    X_train, y_train, cat_cols, num_cols, df
):
    print("\n[7/7] Generating figures...")
    feature_names = cat_cols + num_cols

    # ── Fig 1: Confusion Matrix ──────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["LOW", "HIGH"],
        yticklabels=["LOW", "HIGH"], ax=ax
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ── Fig 2: ROC Curve ─────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {best_name}", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # ── Fig 3: CV F1 Comparison ──────────────────────────────
    names  = list(cv_results.keys())
    means  = [cv_results[n]["cv_f1_mean"] for n in names]
    stds   = [cv_results[n]["cv_f1_std"]  for n in names]
    colors = ["#f97316" if n == best_name else "#6366f1" for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="white")
    ax.set_ylim(0, 1)
    ax.set_ylabel("CV F1-Score")
    ax.set_title("Model Comparison — Cross-Validated F1", fontsize=13)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cv_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Fig 4: Feature Importance (permutation) ───────────────
    print("      Computing permutation importance (may take ~30s)...")
    try:
        # Use a sample to speed this up
        sample_idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
        X_samp = X_test.iloc[sample_idx]
        y_samp = y_test.iloc[sample_idx]

        perm = permutation_importance(
            best_pipe, X_samp, y_samp,
            n_repeats=10, random_state=SEED, n_jobs=-1, scoring="f1"
        )
        imp_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": perm.importances_mean,
            "std":        perm.importances_std,
        }).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            imp_df["feature"][::-1],
            imp_df["importance"][::-1],
            xerr=imp_df["std"][::-1],
            color="#6366f1", edgecolor="white", capsize=3
        )
        ax.set_xlabel("Mean decrease in F1")
        ax.set_title("Top 15 Feature Importances (Permutation)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"      [warn] Feature importance failed: {e}")

    # ── Fig 5: Per-Platform F1 Breakdown ─────────────────────
    X_test_c = X_test.copy()
    X_test_c["y_true"] = y_test.values
    X_test_c["y_pred"] = y_pred

    plat_f1 = {}
    for p in X_test_c["platform"].unique():
        sub = X_test_c[X_test_c["platform"] == p]
        if len(sub) > 10:
            plat_f1[p] = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        list(plat_f1.keys()), list(plat_f1.values()),
        color=["#f97316", "#06b6d4", "#8b5cf6", "#10b981"][:len(plat_f1)],
        edgecolor="white"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Platform F1 Score on Test Set", fontsize=13)
    for i, (p, v) in enumerate(plat_f1.items()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "platform_f1_breakdown.png"), dpi=150)
    plt.close(fig)

    # ── Fig 6: Class distribution per platform ───────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    dist = df.groupby(["platform", "engagement_class"]).size().unstack(fill_value=0)
    dist.plot(kind="bar", ax=ax, color=["#f43f5e", "#22d3ee"], edgecolor="white")
    ax.set_title("Engagement Class Distribution by Platform", fontsize=13)
    ax.set_xlabel("Platform"); ax.set_ylabel("Count")
    ax.legend(["LOW (0)", "HIGH (1)"]); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"      Figures saved to: {FIG_DIR}/")
    print("        confusion_matrix.png")
    print("        roc_curve.png")
    print("        cv_comparison.png")
    print("        feature_importance.png")
    print("        platform_f1_breakdown.png")
    print("        class_distribution.png")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DG-Social Phase 2 — ML Training Pipeline")
    print("=" * 60)
    t_start = time.time()

    df                                              = load_data()
    df                                              = engineer_features(df)
    X_train, X_test, y_train, y_test, cat_cols, num_cols = split_data(df)

    preprocessor                                    = build_preprocessor(cat_cols, num_cols)
    models                                          = get_models()
    cv_results, best_name                           = cross_validate_models(
                                                        models, preprocessor,
                                                        X_train, y_train,
                                                        cat_cols, num_cols
                                                    )
    best_pipe, y_pred, y_proba, test_metrics        = final_train_and_evaluate(
                                                        models, best_name, preprocessor,
                                                        X_train, X_test, y_train, y_test
                                                    )
    save_artefacts(best_pipe, preprocessor, cv_results, best_name, test_metrics)
    plot_all(
        best_pipe, best_name, cv_results,
        X_test, y_test, y_pred, y_proba,
        X_train, y_train, cat_cols, num_cols, df
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Best model : {best_name}")
    print(f"  Test F1    : {test_metrics['test_f1']:.4f}")
    print(f"  Test AUC   : {test_metrics['test_auc']:.4f}")
    print(f"  Saved to   : {MODELS_DIR}/  and  {OUTPUTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
