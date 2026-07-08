"""
config.py — Central constants for the Phase 2 pipeline.
"""
import os

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── Data files ───────────────────────────────────────────────────────────────
COMBINED_CSV    = os.path.join(DATA_DIR, "combined_dataset.csv")
RAW_FACEBOOK    = os.path.join(DATA_DIR, "dataset.csv")

# ── Model artefacts ───────────────────────────────────────────────────────────
BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_PATH    = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_RESULTS_PATH   = os.path.join(OUTPUTS_DIR, "model_results.json")
LABEL_ENCODERS_PATH  = os.path.join(MODELS_DIR, "label_encoders.joblib")

# ── Training ──────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20          # 80/20 train-test split
CV_FOLDS       = 5             # stratified k-fold cross-validation

# ── Feature columns ───────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["platform", "post_type"]
NUMERICAL_FEATURES   = [
    "post_hour", "post_weekday", "is_weekend", "is_peak_hour",
    "has_media", "is_video", "is_paid", "caption_length",
    "hashtag_count", "follower_count", "total_views",
]
TARGET = "engagement_class"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
